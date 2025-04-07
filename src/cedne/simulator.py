import jax
import jax.numpy as jnp
import equinox as eqx
import diffrax as dfx
import numpy as np
import networkx as nx
import copy
import logging
from cedne import cedne

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
        self.name = node.name if isinstance(node, cedne.Neuron) else node
        self.model = model
        self.gain = gain
        self.time_constant = time_constant
        self.baseline = baseline
        self.static = static
        self.set_activation(activation)

        self.neuron_parameters = {'gain':gain, 'time_constant':time_constant, 'baseline':baseline, 'static':static, 'activation':activation}

        for key, value in kwargs.items():
            setattr(self, key, value)
        
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
        return sum([self.gain*inp.process_input(t) for inp in self.inputs])

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
    
    def set_neuron_parameters(self, neuron_parameters):
        """Set the parameters of the neurons.""" 
        for par in self.neuron_parameters:
            for neuron in self.neuron_parameters[par]:
                if not neuron in self.neurons.values():
                    raise ValueError(f"Neuron {neuron} not found in the model")
                if neuron in neuron_parameters[par]:
                    neuron.set_parameter(par, neuron_parameters[par][neuron])
        self.update_neuron_parameters()

    def set_edge_parameters(self, edge_parameters):
        """Set the parameters of the edges."""
        for par in self.edge_parameters:
            for edge in self.edge_parameters[par]:
                if not edge in self.edges:
                    raise ValueError(f"Edge {edge} not found in the model")
                if edge in edge_parameters[par]:
                    nx.set_edge_attributes(self, {edge: {par: edge_parameters[par][edge]}})
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
        derivatives = np.zeros(num_dynamic_neurons, dtype=np.float32)
        synaptic_inputs = np.zeros(num_dynamic_neurons, dtype=np.float32)
        external_inputs = np.zeros(num_dynamic_neurons, dtype=np.float32)

        # Process external inputs efficiently
        input_neurons_mask = np.array([neuron.name in self.input_neurons for neuron in self.dynamic_neurons])
        external_inputs[input_neurons_mask] = np.array([neuron.process_inputs(t) for neuron, mask in zip(self.dynamic_neurons, input_neurons_mask) if mask])

        for i, neuron in enumerate(self.dynamic_neurons):
            if not input_neurons_mask[i]:  # Skip input neurons
                edges = [(in_neuron.rate, data['weight']) for in_neuron, _, data in self.in_edges(neuron, data=True)]
                if edges:
                    in_neurons, weights = zip(*edges)
                    synaptic_inputs[i] = np.dot(in_neurons, weights)  # Use NumPy dot product
                else:
                    in_neurons, weights = [], []  # Set empty lists
                    synaptic_inputs[i] = 0  # No input contribution
                # in_neurons, weights = zip(*[(in_neuron.rate, data['weight'])
                #                       for in_neuron, _, data in self.in_edges(neuron, data=True)])
                # synaptic_inputs[i] = np.dot(in_neurons, weights)  # Matrix multiplication for efficiency
                # synaptic_inputs[i] = sum(in_neuron.rate * data['weight'] for in_neuron, _, data in self.in_edges(neuron, data=True))

        # synaptic_inputs = np.array([
        #     sum(in_neuron.rate * data['weight'] for in_neuron, _, data in self.in_edges(neuron, data=True))
        #     if not input_neurons_mask[i] else 0.0
        #     for i, neuron in enumerate(self.dynamic_neurons)
        # ], dtype=np.float32)
        baselines = np.array([neuron.baseline for neuron in self.dynamic_neurons])
        total_input = synaptic_inputs + external_inputs + baselines

        activations = np.array([neuron.activation(x) for neuron, x in zip(self.dynamic_neurons, total_input)])

        time_constants = np.array([neuron.time_constant for neuron in self.dynamic_neurons])
        gains = np.array([neuron.gain for neuron in self.dynamic_neurons])
        rates = np.array([neuron.rate for neuron in self.dynamic_neurons])

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
        if time_points is not None:
            self.time_points = time_points
        if self.time_points is None:
            raise ValueError("Time points must be set")
        if initial_rates is None:
            initial_rates = np.zeros(len(self.dynamic_neurons))
        simulated_rates = np.zeros((len(self.time_points), len(self.dynamic_neurons)))
        static_rates = np.zeros((len(self.time_points), len(self.static_neurons)))
        
        rates = {neuron: [] for neuron in self.static_neurons + self.dynamic_neurons}
        # ## Set inputs
        # assert all(isinstance(inp, Input) for inp in inputs)
        # self.set_inputs(inputs)

        ## Initial conditions
        t = 0
        static_rates[t]= [neuron.process_inputs(self.time_points[t]) for neuron in self.static_neurons]

        simulated_rates[t] = initial_rates

        for i, neuron in enumerate(self.dynamic_neurons):
            neuron.rate = simulated_rates[t,i]
        for i, neuron in enumerate(self.static_neurons):
            neuron.rate = static_rates[t,i]

        for t in range(len(self.time_points)-1):
            derivatives = self.rate_equations(self.time_points[t])
            simulated_rates[t+1] = simulated_rates[t] + derivatives * (self.time_points[t+1] - self.time_points[t])
            static_rates[t+1,:] = [neuron.process_inputs(self.time_points[t+1]) for neuron in self.static_neurons]

            for i, neuron in enumerate(self.dynamic_neurons):
                neuron.rate = simulated_rates[t+1,i]
            for i, neuron in enumerate(self.static_neurons):
                neuron.rate = static_rates[t+1,i]
        
        for i, neuron in enumerate(self.static_neurons):
            rates[neuron] = static_rates[:,i]
        for i, neuron in enumerate(self.dynamic_neurons):
            rates[neuron] = simulated_rates[:,i]

        # rates = np.concatenate((static_rates, simulated_rates), axis=1)
        
        return rates

    def reinitialize(self):
        pass

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