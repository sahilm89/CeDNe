'''
This module contains the implementations for simulating neural networks
using different neuron models. It includes classes for defining neurons,
inputs, and models, as well as methods for simulating the dynamics of the network.
The main classes are:
- `Input`: Represents an input to a neuron.
- `StepInput`: Represents a step input to a neuron.
- `TimeDependentInput`: Represents a time-dependent input to a neuron.
- `Neuron`: Represents a neuron in the network.
- `Model`: Represents the base class for a neural network model.
- `RateModel`: Represents a rate model for a neural network.
- `JaxNeuron`: Represents a JAX-compatible neuron.
- `JaxRateModel`: Represents a JAX-compatible rate model for a neural network.
'''
__author__ = "Sahil Moza"
__date__ = "2025-04-06"
__license__ = "MIT"

import jax
import jax.numpy as jnp
import equinox as eqx
import diffrax as dfx
import numpy as np
import networkx as nx
import copy
import logging
from cedne import Neuron 

logging.basicConfig(
    filename="debug_log.txt",  # Save logs to a file
    filemode="w",  # Overwrite each run
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO  # Set to INFO (change to DEBUG for more details)
)

class Input:
    """ 
    A class representing an input to a neuron."""
    def __init__(self, input_neurons):
        """
        Initialize the input.

        Args:
            input_neurons (list): A list of neurons that receive this input.
        """
        self.input_neurons = input_neurons

class StepInput(Input):
    """ 
    A class representing a step input to a neuron."""
    def __init__(self, input_neurons, tstart, tend, value):
        """
        Initialize the input.

        Args:
            input_neurons (list): A list of neurons that receive this input.
            tstart (float): The start time of the input.
            tend (float): The end time of the input.
            value (float): The value of the input.
        """
        super().__init__(input_neurons)
        self.value = value
        self.tstart = tstart
        self.tend = tend

    def process_input(self, t):
        """
        Process the input at a given time.

        Args:
            t (float): The current time.

        Returns:
            float: The processed input.
        """
        return self.value if t > self.tstart and t<self.tend else 0
    
class StepInputWithAdaptation(StepInput):
    """ 
    A class representing a step input to a neuron."""
    def __init__(self, input_neurons, tstart, tend, value, decay_rate):
        """
        Initialize the input.

        Args:
            input_neurons (list): A list of neurons that receive this input.
            tstart (float): The start time of the input.
            tend (float): The end time of the input.
            value (float): The value of the input.
            decay_rate (float): The decay rate of the input.
        """
        super().__init__(input_neurons, tstart, tend, value)
        self.decay_rate = decay_rate

    def process_input(self, t):
        """
        Process the input at a given time.

        Args:
            t (float): The current time.

        Returns:
            float: The processed input.
        """
        return self.value * np.exp(-(t-self.tstart)/self.decay_rate) if t > self.tstart and t<self.tend else 0
        
class TimeDependentInput(Input):
    """ 
    A class representing a time-dependent input to a neuron."""
    def __init__(self, input_neurons, function):
        """
        Initialize the input.

        Args:
            input_neurons (list): A list of neurons that receive this input.
            function (function): A function that takes a time as input and returns the value of the input.
        """
        super().__init__(input_neurons)
        self.function = function

    def process_input(self, t):
        """
        Process the input at a given time.

        Args:
            t (float): The current time.

        Returns:
            float: The processed input.
        """
        return self.function(t)

class TimeSeriesInput(TimeDependentInput):
    """ 
    A class representing a time series input to a neuron."""
    def __init__(self, input_neurons, values):
        """
        Initialize the input.

        Args:
            input_neurons (list): A list of neurons that receive this input.
            values (list): A list of values of the input.
        """
        self.values = values
        super().__init__(input_neurons, lambda t: values[t])

class Neuron:
    """ 
    A class representing a neuron in a neural network."""
    def __init__(self, node, model, gain=0, time_constant=1, baseline=0., static=False, activation='linear', **kwargs):
        """
        Initialize the neuron.

        Args:
            node (Node): The node associated with the neuron.
            model (Model): The model the neuron belongs to.
        """
        self.node = node
        self.name = node.name if isinstance(node, Neuron) else node
        self.model = model
        self.gain = gain
        self.time_constant = time_constant
        self.baseline = baseline
        self.static = static
        self.set_activation(activation)

        for key, value in kwargs.items():
            setattr(self, key, value)

        self.neuron_parameters = {
            'gain': gain,
            'time_constant': time_constant,
            'baseline': baseline,
            'static': static,
            'activation': activation,
            **kwargs,
        }
        
        self.model.add_node(self, gain=gain, time_constant=time_constant, baseline=baseline, static=static, activation=activation, **kwargs)
    
    def set_timeconstant(self, time_constant):
        """ 
        Set the time constant of the neuron."""
        self.time_constant = time_constant
        self.neuron_parameters['time_constant'] = time_constant

    def set_baseline(self, baseline):
        """ 
        Set the baseline of the neuron."""
        self.baseline = baseline
        self.neuron_parameters['baseline'] = baseline

    def set_parameter(self, param, value):
        """ 
        Set a parameter of the neuron."""
        setattr(self, param, value)
        self.neuron_parameters[param] = value
    
    def set_activation(self, activation):
        """ 
        Set the activation function of the neuron."""
        if activation == 'linear':
            self.activation = lambda x:x
        elif activation == 'sigmoid':
            self.activation = lambda x: 1/(1+np.exp(-x))
        elif activation == 'tanh':
            self.activation = np.tanh
        elif activation == 'relu':
            self.activation = lambda x: np.maximum(0, x)
        else:
            raise ValueError("Activation function not supported")

class JaxNeuron(eqx.Module):
    """
    A class representing a JAX neuron in a neural network.
    """
    name: str
    gain: float
    time_constant: float
    baseline: float
    static: bool
    activation: callable

    def __init__(self, node, gain=0, time_constant=1, baseline=0., static=False, activation='linear'):
        self.name = node.name
        self.node = node
        self.gain = gain
        self.time_constant = time_constant
        self.baseline = baseline
        self.static = static
        self.set_activation(activation)

    def set_activation(self, activation):
        if activation == 'linear':
            self.activation = lambda x: x
        elif activation == 'sigmoid':
            self.activation = lambda x: 1 / (1 + jnp.exp(-x))
        elif activation == 'tanh':
            self.activation = jnp.tanh
        elif activation == 'relu':
            self.activation = lambda x: jnp.maximum(0, x)
        else:
            raise ValueError("Activation function not supported")
        
class InputNeuron(Neuron):
    """ 
    A class representing an input neuron in a neural network."""
    def __init__(self, node, model, gain=0, time_constant=1, baseline=0., static=False, **kwargs):
        """
        Initialize the input neuron.

        Args:
            name (str): The name of the neuron.
        """
        super().__init__(node, model, gain, time_constant, baseline, static, **kwargs)
        self.inputs = []
        self.baseline = baseline

    def set_input(self, inp):
        """ 
        Set the inputs of the input neuron."""
        if not isinstance(inp, Input):
            raise ValueError("Inputs must be of type Input")
        self.inputs+= [inp]
    
    def process_inputs(self, t):
        """
        Process the inputs at a given time.

        Args:
            t (float): The current time.

        Returns:
            float: The processed inputs.
        """
        gain = 1.0 if self.gain is None else self.gain
        return sum([gain * inp.process_input(t) for inp in self.inputs])

class Model(nx.MultiDiGraph):
    """ 
    A class representing a model for a neural network."""
    def __init__(self, graph, input_neurons, neuron_parameters=None, edge_parameters=None, static_neurons=None, inputs=None, time_points=None):
        """
        Initialize the model.

        Args:
            graph (networkx.DiGraph): The graph representing the neural network.
            weights (dict): A dictionary where the keys are the edges and the values are the weights.
            input_neurons (list): A list of neurons that receive external input.
            gains (dict): A dictionary where the keys are the neurons and the values are the list of gain terms for each neuron.
            time_constants (dict): A dictionary where the keys are the neurons and the values are the list of time constants for each neuron.
        """
        super().__init__()
        self.source_graph = graph
        self.neurons = {}
        self.input_neurons = {}
        if static_neurons is None:
            static_neurons = []

        for node, data in graph.nodes(data=True):
            neuron_args = {param: neuron_params.get(node) for param, neuron_params in neuron_parameters.items()}
            for param in neuron_args:
                if param in data:
                    data.pop(param)
            if not node in input_neurons:
                self.neurons[node] = Neuron(node, self, static=node in static_neurons, **neuron_args, **data) #gain=gains[node], time_constant=time_constants[node], 
            else:
                self.input_neurons[node] = InputNeuron(node, self, static=node in static_neurons, **neuron_args, **data)
                #self.input_neurons[node] = InputNeuron(node, self, gain=gains[node], time_constant=time_constants[node], static=node in static_neurons, **data)
        
        self.neurons.update(self.input_neurons)
        self.dynamic_neurons = [self.neurons[neuron] for neuron in self.neurons if not self.neurons[neuron].static]
        self.static_neurons = [self.neurons[neuron] for neuron in self.neurons if self.neurons[neuron].static]

        for edge_0, edge_1, data in graph.edges(data=True):
            edge_args = {param: edge_params.get((edge_0, edge_1)) for param, edge_params in edge_parameters.items()}
            for param in edge_args:
                if param in data:
                    data.pop(param)
            self.add_edge(self.neurons[edge_0], self.neurons[edge_1], **edge_args, **data)
            #self.add_edge(self.neurons[edge_0], self.neurons[edge_1], weight=weights[(edge_0, edge_1)] if weights is not None else data['weight'], **data)
        self.time_points = time_points
        self.neuron_parameters = {par: {self.neurons[neuron]: neuron_parameters[par][neuron] for neuron in neuron_parameters[par]} for par in neuron_parameters}
        # print({par: {(self.neurons[edge[0]], self.neurons[edge[1]], 0): edge_parameters[par][edge] for edge in edge_parameters[par]} for par in edge_parameters})
        self.edge_parameters = {par: {(self.neurons[edge[0]], self.neurons[edge[1]], 0): edge_parameters[par][edge] for edge in edge_parameters[par]} for par in edge_parameters}
        self.inputs = inputs
        if inputs is not None:
            self.set_inputs(inputs)

    def set_inputs(self, inputs):
        """Set the inputs to the input neurons."""
        assert all(isinstance(inp, Input) for inp in inputs)
        for inp in inputs:
            for neuron in inp.input_neurons:
                self.neurons[neuron].set_input(inp)
        self.inputs = inputs

    def remove_inputs(self):
        """Remove the inputs from the input neurons."""
        for neuron in self.input_neurons:
            self.neurons[neuron].inputs = []
        self.inputs = None

    def _ensure_time_points(self, time_points=None):
        """Normalize and validate the model time axis."""
        if time_points is not None:
            self.time_points = time_points
        if self.time_points is None:
            raise ValueError("Time points must be set")
        return self.time_points

    def _empty_static_rates(self):
        """Allocate the static-neuron trajectory array for the current time axis."""
        return np.zeros((len(self.time_points), len(self.static_neurons)), dtype=np.float32)

    def _initialize_static_rates(self, static_rates):
        """Populate static neurons at the first time point."""
        if len(self.static_neurons) == 0:
            return
        static_rates[0] = [neuron.process_inputs(self.time_points[0]) for neuron in self.static_neurons]
        for i, neuron in enumerate(self.static_neurons):
            neuron.rate = static_rates[0, i]

    def _advance_static_rates(self, static_rates, t_idx):
        """Advance static-neuron inputs to the next time step."""
        if len(self.static_neurons) == 0:
            return
        static_rates[t_idx + 1, :] = [
            neuron.process_inputs(self.time_points[t_idx + 1]) for neuron in self.static_neurons
        ]
        for i, neuron in enumerate(self.static_neurons):
            neuron.rate = static_rates[t_idx + 1, i]

    def _pack_observable_trajectories(self, dynamic_values, static_rates):
        """Return the standard dict[neuron -> trajectory] observable contract."""
        rates = {neuron: [] for neuron in self.static_neurons + self.dynamic_neurons}
        for i, neuron in enumerate(self.static_neurons):
            rates[neuron] = static_rates[:, i]
        for i, neuron in enumerate(self.dynamic_neurons):
            rates[neuron] = dynamic_values[:, i]
        return rates

    def _dynamic_weight_matrix(self, connection_type=None):
        """Return the weighted adjacency over dynamic neurons in simulator order."""
        ordered_neurons = list(self.dynamic_neurons)
        index_map = {neuron: idx for idx, neuron in enumerate(ordered_neurons)}
        matrix = np.zeros((len(ordered_neurons), len(ordered_neurons)), dtype=np.float32)

        selectors = None
        if connection_type is not None:
            selectors = [connection_type] if isinstance(connection_type, str) else list(connection_type)

        for pre, post, edge_data in self.edges(data=True):
            if pre not in index_map or post not in index_map:
                continue
            if selectors is not None:
                edge_type = edge_data.get("connection_type")
                include_edge = False
                for selector in selectors:
                    if selector == "chemical" and edge_type == "chemical-synapse":
                        include_edge = True
                    elif selector in {"gap", "gap-junction"} and edge_type == "gap-junction":
                        include_edge = True
                    elif selector == "bulk" and edge_type not in {"chemical-synapse", "gap-junction"}:
                        include_edge = True
                    elif edge_type == selector:
                        include_edge = True
                if not include_edge:
                    continue
            matrix[index_map[pre], index_map[post]] += edge_data.get("weight", 1.0)
        return matrix
    
    def set_neuron_parameters(self, neuron_parameters):
        """Set the parameters of the neurons.""" 
        for par in self.neuron_parameters:
            updated_values = neuron_parameters.get(par, {})
            for neuron in self.neuron_parameters[par]:
                if not neuron in self.neurons.values():
                    raise ValueError(f"Neuron {neuron} not found in the model")
                if neuron in updated_values:
                    neuron.set_parameter(par, updated_values[neuron])
        self.update_neuron_parameters()

    def set_edge_parameters(self, edge_parameters):
        """Set the parameters of the edges."""
        for par in self.edge_parameters:
            updated_values = edge_parameters.get(par, {})
            for edge in self.edge_parameters[par]:
                if not edge in self.edges:
                    raise ValueError(f"Edge {edge} not found in the model")
                if edge in updated_values:
                    nx.set_edge_attributes(self, {edge: {par: updated_values[edge]}})
        self.update_edge_parameters()
    
    def update_neuron_parameters(self):
        for n,neuron in self.neurons.items():
            for par in self.neuron_parameters:
                self.neuron_parameters[par][neuron] = neuron.neuron_parameters[par]

    def update_edge_parameters(self):
        for edge in self.edges:
            for par in self.edge_parameters:
                self.edge_parameters[par][edge] = self.edges[edge][par]
    
    def copy(self):
        """Deepcopy the model."""
        return copy.deepcopy(self)

class RateModel(Model):
    """ 
    A class representing a rate model for a neural network."""
    def __init__(self, graph, input_neurons, weights=None, gains=None, time_constants=None, baseline=0., static_neurons=None, time_points=None, inputs=None) -> None:
        """
        Initialize the rate model.

        Args:
            graph (networkx.DiGraph): The graph representing the neural network.
            weights (dict): A dictionary where the keys are the edges and the values are the weights.
            input_neurons (list): A list of neurons that receive external input.
            gains (dict): A dictionary where the keys are the neurons and the values are the list of gain terms for each neuron.
            time_constants (dict): A dictionary where the keys are the neurons and the values are the list of time constants for each neuron.
            static_neurons (list): A list of neurons that are static.
            time_points (list): A list of time points.
        """
        super().__init__(graph, input_neurons, neuron_parameters={'gain':gains, 'time_constant':time_constants, 'baseline':baseline}, edge_parameters={'weight':weights},  static_neurons=static_neurons, time_points=time_points, inputs=inputs)
        
    def rate_equations(self, t):
        """
        Compute the derivatives of the rates with respect to time.

        Args:
            t (float): The current time.

        Returns:
            list: The derivatives of the rates with respect to time.
        """
        num_dynamic_neurons = len(self.dynamic_neurons)
        external_inputs = np.zeros(num_dynamic_neurons, dtype=np.float32)
        rates = np.array([neuron.rate for neuron in self.dynamic_neurons], dtype=np.float32)

        # Process external inputs efficiently
        input_neurons_mask = np.array([neuron.name in self.input_neurons for neuron in self.dynamic_neurons])
        external_inputs[input_neurons_mask] = np.array([neuron.process_inputs(t) for neuron, mask in zip(self.dynamic_neurons, input_neurons_mask) if mask])

        synaptic_inputs = self._dynamic_weight_matrix().T @ rates
        synaptic_inputs[input_neurons_mask] = 0.0
        baselines = np.array([neuron.baseline for neuron in self.dynamic_neurons])
        total_input = synaptic_inputs + external_inputs + baselines

        activations = np.array([neuron.activation(x) for neuron, x in zip(self.dynamic_neurons, total_input)])

        time_constants = np.array([neuron.time_constant for neuron in self.dynamic_neurons])
        gains = np.array([neuron.gain for neuron in self.dynamic_neurons])

        if np.isnan(time_constants).any():
            logging.warning(f"NaN in time_constants at t={t}. Values: {time_constants}")
        if np.isnan(gains).any():
            logging.warning(f"NaN in gains at t={t}. Values: {gains}")
        if np.isnan(rates).any():
            logging.warning(f"NaN in rates at t={t}. Values: {rates}")
        if np.isnan(activations).any():
            logging.warning(f"NaN in activations at t={t}. Inputs: {total_input}")

        derivatives = (1 / time_constants) * (-rates + gains * activations)

        if np.isnan(derivatives).any():
            logging.error(f"NaN in derivatives at t={t}! Full values: {derivatives}")

        return derivatives

    def simulate(self, time_points=None, initial_rates=None):
        """
        Simulate the rates over time.

        Args:
            time_points (list): The time points at which to simulate the rates.
            initial_rates (list): The initial rates of the neurons.

        Returns:
            list: The simulated rates at each time point.
        """
        self._ensure_time_points(time_points)
        if initial_rates is None:
            initial_rates = np.zeros(len(self.dynamic_neurons))
        simulated_rates = np.zeros((len(self.time_points), len(self.dynamic_neurons)))
        static_rates = self._empty_static_rates()
        # ## Set inputs
        # assert all(isinstance(inp, Input) for inp in inputs)
        # self.set_inputs(inputs)

        ## Initial conditions
        simulated_rates[0] = initial_rates
        self._initialize_static_rates(static_rates)

        for i, neuron in enumerate(self.dynamic_neurons):
            neuron.rate = simulated_rates[0, i]

        for t in range(len(self.time_points)-1):
            derivatives = self.rate_equations(self.time_points[t])
            simulated_rates[t+1] = simulated_rates[t] + derivatives * (self.time_points[t+1] - self.time_points[t])
            self._advance_static_rates(static_rates, t)

            for i, neuron in enumerate(self.dynamic_neurons):
                neuron.rate = simulated_rates[t+1,i]
        return self._pack_observable_trajectories(simulated_rates, static_rates)

    def reinitialize(self):
        pass


class LDSModel(Model):
    """
    Linear dynamical system over neural activity.

    Dynamics:
        x(t + dt) = x(t) + dt * (A x(t) + B u(t) + baseline)

    This forward simulator evolves the neuron activity vector directly.
    """

    def __init__(
        self,
        graph,
        input_neurons,
        weights=None,
        baseline=0.0,
        input_weight=1.0,
        static_neurons=None,
        time_points=None,
        inputs=None,
    ) -> None:
        def _coerce_node_param(value):
            if isinstance(value, dict):
                return value
            return {node: value for node in graph.nodes}

        super().__init__(
            graph,
            input_neurons,
            neuron_parameters={
                'gain': _coerce_node_param(1.0),
                'baseline': _coerce_node_param(baseline),
                'input_weight': _coerce_node_param(input_weight),
            },
            edge_parameters={'weight': weights},
            static_neurons=static_neurons,
            time_points=time_points,
            inputs=inputs,
        )

    def lds_equations(self, t):
        """
        Compute linear state derivatives at time ``t``.
        """
        num_dynamic_neurons = len(self.dynamic_neurons)
        external_inputs = np.zeros(num_dynamic_neurons, dtype=np.float32)
        states = np.array([neuron.rate for neuron in self.dynamic_neurons], dtype=np.float32)

        for i, neuron in enumerate(self.dynamic_neurons):
            if neuron.name in self.input_neurons:
                external_inputs[i] = neuron.input_weight * neuron.process_inputs(t)

        recurrent_inputs = self._dynamic_weight_matrix().T @ states
        baselines = np.array([neuron.baseline for neuron in self.dynamic_neurons], dtype=np.float32)
        derivatives = recurrent_inputs + external_inputs + baselines
        return derivatives

    def simulate(self, time_points=None, initial_states=None):
        """
        Simulate the linear observable state over time.
        """
        self._ensure_time_points(time_points)
        if initial_states is None:
            initial_states = np.zeros(len(self.dynamic_neurons), dtype=np.float32)

        simulated_states = np.zeros((len(self.time_points), len(self.dynamic_neurons)), dtype=np.float32)
        static_rates = self._empty_static_rates()
        simulated_states[0] = initial_states
        self._initialize_static_rates(static_rates)

        for i, neuron in enumerate(self.dynamic_neurons):
            neuron.rate = simulated_states[0, i]

        for t in range(len(self.time_points) - 1):
            dt = self.time_points[t + 1] - self.time_points[t]
            derivatives = self.lds_equations(self.time_points[t])
            simulated_states[t + 1] = simulated_states[t] + derivatives * dt
            self._advance_static_rates(static_rates, t)

            for i, neuron in enumerate(self.dynamic_neurons):
                neuron.rate = simulated_states[t + 1, i]

        return self._pack_observable_trajectories(simulated_states, static_rates)

    def reinitialize(self):
        pass


class CTRNNModel(Model):
    """
    Continuous-time recurrent neural network model.

    Dynamics:
        tau * dx/dt = -x + gain * activation(Wx + input + baseline)

    This keeps the same graph/input/parameter plumbing as ``RateModel`` while
    exposing a separate simulator family with explicit non-linear recurrent
    dynamics.
    """
    def __init__(
        self,
        graph,
        input_neurons,
        weights=None,
        tau=None,
        gains=None,
        baseline=0.,
        activation='tanh',
        static_neurons=None,
        time_points=None,
        inputs=None,
    ) -> None:
        neuron_activation = activation
        if isinstance(activation, dict):
            neuron_activation = activation
        else:
            neuron_activation = {
                node: activation for node in graph.nodes
            }
        super().__init__(
            graph,
            input_neurons,
            neuron_parameters={
                'gain': gains,
                'time_constant': tau,
                'baseline': baseline,
                'activation': neuron_activation,
            },
            edge_parameters={'weight': weights},
            static_neurons=static_neurons,
            time_points=time_points,
            inputs=inputs,
        )

    def ctrnn_equations(self, t):
        """
        Compute the state derivatives at time ``t``.
        """
        num_dynamic_neurons = len(self.dynamic_neurons)
        external_inputs = np.zeros(num_dynamic_neurons, dtype=np.float32)
        states = np.array([neuron.rate for neuron in self.dynamic_neurons], dtype=np.float32)

        input_neurons_mask = np.array(
            [neuron.name in self.input_neurons for neuron in self.dynamic_neurons]
        )
        external_inputs[input_neurons_mask] = np.array(
            [neuron.process_inputs(t) for neuron, mask in zip(self.dynamic_neurons, input_neurons_mask) if mask]
        )

        synaptic_inputs = self._dynamic_weight_matrix().T @ states
        synaptic_inputs[input_neurons_mask] = 0.0

        baselines = np.array([neuron.baseline for neuron in self.dynamic_neurons], dtype=np.float32)
        total_input = synaptic_inputs + external_inputs + baselines
        activations = np.array([neuron.activation(x) for neuron, x in zip(self.dynamic_neurons, total_input)], dtype=np.float32)
        taus = np.array([neuron.time_constant for neuron in self.dynamic_neurons], dtype=np.float32)
        gains = np.array([neuron.gain for neuron in self.dynamic_neurons], dtype=np.float32)

        derivatives = (1 / taus) * (-states + gains * activations)
        return derivatives

    def simulate(self, time_points=None, initial_states=None):
        """
        Simulate the CTRNN states over time.
        """
        self._ensure_time_points(time_points)
        if initial_states is None:
            initial_states = np.zeros(len(self.dynamic_neurons), dtype=np.float32)

        simulated_states = np.zeros((len(self.time_points), len(self.dynamic_neurons)), dtype=np.float32)
        static_rates = self._empty_static_rates()
        simulated_states[0] = initial_states
        self._initialize_static_rates(static_rates)

        for i, neuron in enumerate(self.dynamic_neurons):
            neuron.rate = simulated_states[0, i]

        for t in range(len(self.time_points) - 1):
            derivatives = self.ctrnn_equations(self.time_points[t])
            dt = self.time_points[t + 1] - self.time_points[t]
            simulated_states[t + 1] = simulated_states[t] + derivatives * dt
            self._advance_static_rates(static_rates, t)

            for i, neuron in enumerate(self.dynamic_neurons):
                neuron.rate = simulated_states[t + 1, i]
        return self._pack_observable_trajectories(simulated_states, static_rates)

    def reinitialize(self):
        pass


class DKBModel(Model):
    """
    Damped second-order network model.

    Dynamics:
        dx/dt = v
        dv/dt = -damping * v - stiffness * (x - target)
                + recurrent_weighted_input + input_weight * external_input
                + baseline

    The observable trajectory returned by ``simulate`` is the position-like
    state ``x``. The auxiliary velocity state is retained on
    ``self.last_velocities`` for downstream inspection.
    """
    def __init__(
        self,
        graph,
        input_neurons,
        weights=None,
        damping=None,
        stiffness=None,
        baseline=0.,
        input_weight=1.0,
        target=0.0,
        static_neurons=None,
        time_points=None,
        inputs=None,
    ) -> None:
        def _coerce_node_param(value):
            if isinstance(value, dict):
                return value
            return {node: value for node in graph.nodes}

        super().__init__(
            graph,
            input_neurons,
            neuron_parameters={
                'gain': _coerce_node_param(1.0),
                'damping': _coerce_node_param(damping),
                'stiffness': _coerce_node_param(stiffness),
                'baseline': _coerce_node_param(baseline),
                'input_weight': _coerce_node_param(input_weight),
                'target': _coerce_node_param(target),
            },
            edge_parameters={'weight': weights},
            static_neurons=static_neurons,
            time_points=time_points,
            inputs=inputs,
        )
        self.last_velocities = None

    def dkb_equations(self, t, positions, velocities):
        """
        Compute the second-order dynamics at time ``t``.
        """
        num_dynamic_neurons = len(self.dynamic_neurons)
        position_derivatives = velocities.copy()
        velocity_derivatives = np.zeros(num_dynamic_neurons, dtype=np.float32)
        external_inputs = np.zeros(num_dynamic_neurons, dtype=np.float32)

        for i, neuron in enumerate(self.dynamic_neurons):
            if neuron.name in self.input_neurons:
                external_inputs[i] = neuron.input_weight * sum(
                    inp.process_input(t) for inp in neuron.inputs
                )

        recurrent_inputs = self._dynamic_weight_matrix().T @ positions
        damping = np.array([neuron.damping for neuron in self.dynamic_neurons], dtype=np.float32)
        stiffness = np.array([neuron.stiffness for neuron in self.dynamic_neurons], dtype=np.float32)
        baseline = np.array([neuron.baseline for neuron in self.dynamic_neurons], dtype=np.float32)
        target = np.array([neuron.target for neuron in self.dynamic_neurons], dtype=np.float32)

        velocity_derivatives = (
            -damping * velocities
            - stiffness * (positions - target)
            + recurrent_inputs
            + external_inputs
            + baseline
        )
        return position_derivatives, velocity_derivatives

    def simulate(self, time_points=None, initial_states=None, initial_velocities=None):
        """
        Simulate the observable ``x`` state over time.
        """
        self._ensure_time_points(time_points)
        if initial_states is None:
            initial_states = np.zeros(len(self.dynamic_neurons), dtype=np.float32)
        if initial_velocities is None:
            initial_velocities = np.zeros(len(self.dynamic_neurons), dtype=np.float32)

        simulated_states = np.zeros((len(self.time_points), len(self.dynamic_neurons)), dtype=np.float32)
        simulated_velocities = np.zeros((len(self.time_points), len(self.dynamic_neurons)), dtype=np.float32)
        static_rates = self._empty_static_rates()
        simulated_states[0] = initial_states
        simulated_velocities[0] = initial_velocities
        self._initialize_static_rates(static_rates)

        for i, neuron in enumerate(self.dynamic_neurons):
            neuron.rate = simulated_states[0, i]
            neuron.velocity = simulated_velocities[0, i]

        for t in range(len(self.time_points) - 1):
            dt = self.time_points[t + 1] - self.time_points[t]
            dx, dv = self.dkb_equations(
                self.time_points[t],
                simulated_states[t],
                simulated_velocities[t],
            )
            simulated_states[t + 1] = simulated_states[t] + dx * dt
            simulated_velocities[t + 1] = simulated_velocities[t] + dv * dt
            self._advance_static_rates(static_rates, t)

            for i, neuron in enumerate(self.dynamic_neurons):
                neuron.rate = simulated_states[t + 1, i]
                neuron.velocity = simulated_velocities[t + 1, i]

        self.last_velocities = {
            neuron: simulated_velocities[:, i]
            for i, neuron in enumerate(self.dynamic_neurons)
        }
        return self._pack_observable_trajectories(simulated_states, static_rates)

    def reinitialize(self):
        self.last_velocities = None


class CalciumObservation:
    """
    Simple calcium/fluorescence observation layer for latent neural activity.

    This applies a first-order rise/decay filter to each trajectory and returns
    a new dictionary keyed by the same neuron objects used in the latent traces.
    """

    def __init__(self, rise_tau=0.2, decay_tau=1.0, scale=1.0, baseline=0.0, rectify=True):
        if rise_tau <= 0 or decay_tau <= 0:
            raise ValueError("rise_tau and decay_tau must be positive")
        self.rise_tau = float(rise_tau)
        self.decay_tau = float(decay_tau)
        self.scale = float(scale)
        self.baseline = float(baseline)
        self.rectify = bool(rectify)

    def transform(self, trajectories, time_points):
        time_points = np.asarray(time_points, dtype=np.float32)
        if time_points.ndim != 1 or len(time_points) < 2:
            raise ValueError("time_points must be a one-dimensional array with at least two entries")

        calcium = {}
        for neuron, values in trajectories.items():
            latent = np.asarray(values, dtype=np.float32)
            if latent.shape[0] != time_points.shape[0]:
                raise ValueError("Each trajectory must match the length of time_points")
            if self.rectify:
                latent = np.maximum(latent, 0.0)

            bound = np.zeros_like(latent, dtype=np.float32)
            observed = np.zeros_like(latent, dtype=np.float32)
            bound[0] = latent[0]
            observed[0] = self.baseline + self.scale * bound[0]

            for i in range(len(time_points) - 1):
                dt = time_points[i + 1] - time_points[i]
                rise_drive = (latent[i] - bound[i]) / self.rise_tau
                bound[i + 1] = bound[i] + dt * rise_drive
                decay_drive = (bound[i] - (observed[i] - self.baseline) / self.scale) / self.decay_tau
                observed_signal = (observed[i] - self.baseline) / self.scale + dt * decay_drive
                observed[i + 1] = self.baseline + self.scale * observed_signal

            calcium[neuron] = observed

        return calcium


class JaxRateModel(eqx.Module):
    """
    A class representing a JAX rate model for a neural network.
    """
    neurons: dict
    edges: dict
    dynamic_neurons: list
    static_neurons: list
    input_neurons: list
    time_points: jnp.ndarray

    def __init__(self, graph, input_neurons, neuron_parameters, edge_parameters, static_neurons=None, time_points=None):
        self.neurons = {}
        self.edges = {}
        self.input_neurons = input_neurons
        self.dynamic_neurons = []
        self.static_neurons = []
        self.time_points = jnp.array(time_points) if time_points is not None else None

        for node in graph.nodes:
            neuron_args = {param: neuron_parameters[param][node] for param in neuron_parameters}
            self.neurons[node] = JaxNeuron(node, **neuron_args)
            if node in static_neurons:
                self.static_neurons.append(self.neurons[node])
            else:
                self.dynamic_neurons.append(self.neurons[node])

        for edge_0, edge_1, k in graph.edges:
            edge_args = {param: edge_parameters[param][(edge_0, edge_1, k)] for param in edge_parameters}
            self.edges[(edge_0, edge_1, k)] = edge_args

    @eqx.filter_jit
    def rate_equations(self, t, rates):
        def compute_derivative(neuron, rate):
            synaptic_input = 0.
            external_inputs = 0.
            if neuron.name in self.input_neurons:
                external_inputs += neuron.process_inputs(t)
            else:
                for in_neuron, _, data in self.in_edges(neuron, data=True): ## this is the problem, since there's no in_edges.
                    synaptic_input += in_neuron.rate * data['weight']
            return (1 / neuron.time_constant) * (-rate +  neuron.gain* neuron.activation(synaptic_input + external_inputs + neuron.baseline))

        derivatives = jax.vmap(compute_derivative)(self.dynamic_neurons, rates)
        return derivatives

    def simulate(self, initial_rates=None):
        if self.time_points is None:
            raise ValueError("Time points must be set")
        if initial_rates is None:
            initial_rates = jnp.zeros(len(self.dynamic_neurons))

        def ode_func(t, y, args):
            return self.rate_equations(t, y)

        solver = dfx.Tsit5()
        term = dfx.ODETerm(ode_func)
        sol = dfx.diffeqsolve(term, solver, t0=self.time_points[0], t1=self.time_points[-1], dt0=0.1, y0=initial_rates, saveat=dfx.SaveAt(ts=self.time_points))
        return sol.ys
