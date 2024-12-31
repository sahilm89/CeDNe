import numpy as np
import networkx as nx
import copy

class Input:
    """ 
    A class representing an input to a neuron."""
    def __init__(self, input_nodes):
        """
        Initialize the input.

        Args:
            input_nodes (list): A list of nodes that receive this input.
        """
        self.input_nodes = input_nodes

class StepInput(Input):
    """ 
    A class representing a step input to a neuron."""
    def __init__(self, input_nodes, tstart, tend, value):
        """
        Initialize the input.

        Args:
            input_nodes (list): A list of nodes that receive this input.
            tstart (float): The start time of the input.
            tend (float): The end time of the input.
            value (float): The value of the input.
        """
        super().__init__(input_nodes)
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
    def __init__(self, input_nodes, tstart, tend, value, decay_rate):
        """
        Initialize the input.

        Args:
            input_nodes (list): A list of nodes that receive this input.
            tstart (float): The start time of the input.
            tend (float): The end time of the input.
            value (float): The value of the input.
            decay_rate (float): The decay rate of the input.
        """
        super().__init__(input_nodes, tstart, tend, value)
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
    def __init__(self, input_nodes, function):
        """
        Initialize the input.

        Args:
            input_nodes (list): A list of nodes that receive this input.
            function (function): A function that takes a time as input and returns the value of the input.
        """
        super().__init__(input_nodes)
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
    def __init__(self, input_nodes, values):
        """
        Initialize the input.

        Args:
            input_nodes (list): A list of nodes that receive this input.
            values (list): A list of values of the input.
        """
        self.values = values
        super().__init__(input_nodes, lambda t: values[t])

class Node:
    """ 
    A class representing a node in a neural network."""
    def __init__(self, label, model, gain=1, time_constant=1,  static=False, **kwargs):
        """
        Initialize the node.

        Args:
            label (str): The name of the node.
        """
        self.label = label
        self.model = model
        self.gain = gain
        self.time_constant = time_constant
        self.static = static
        self.node_parameters = {'gain':gain, 'time_constant':time_constant, 'static':static}

        for key, value in kwargs.items():
            setattr(self, key, value)
        
        self.model.add_node(self, gain=gain, time_constant=time_constant, static=static, **kwargs)
    
    def set_timeconstant(self, time_constant):
        """ 
        Set the time constant of the node."""
        self.time_constant = time_constant

    def set_parameter(self, param, value):
        """ 
        Set a parameter of the node."""
        setattr(self, param, value)
        self.node_parameters[param] = value

class InputNode(Node):
    """ 
    A class representing an input node in a neural network."""
    def __init__(self, label, model, gain=1, time_constant=1, static=False, **kwargs):
        """
        Initialize the input node.

        Args:
            label (str): The name of the node.
        """
        super().__init__(label, model, gain, time_constant, static,**kwargs)
        self.inputs = []
        
    
    def set_input(self, inp):
        """ 
        Set the inputs of the input node."""
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
    def __init__(self, graph, input_nodes, node_parameters=None, edge_parameters=None, static_nodes=None, inputs=None, time_points=None):
        """
        Initialize the model.

        Args:
            graph (networkx.DiGraph): The graph representing the neural network.
            weights (dict): A dictionary where the keys are the edges and the values are the weights.
            input_nodes (list): A list of nodes that receive external input.
            gains (dict): A dictionary where the keys are the nodes and the values are the list of gain terms for each neuron.
            time_constants (dict): A dictionary where the keys are the nodes and the values are the list of time constants for each neuron.
        """
        super().__init__()
        self.nodes = {}
        self.input_nodes = {}
        if static_nodes is None:
            static_nodes = []

        for node, data in graph.nodes(data=True):
            node_args = {param: node_params.get(node) for param, node_params in node_parameters.items()}
            if not node in input_nodes:
                self.nodes[node] = Node(node, self, static=node in static_nodes, **node_args, **data) #gain=gains[node], time_constant=time_constants[node], 
            else:
                self.input_nodes[node] = InputNode(node, self, static=node in static_nodes, **node_args, **data)
                #self.input_nodes[node] = InputNode(node, self, gain=gains[node], time_constant=time_constants[node], static=node in static_nodes, **data)
        
        self.nodes.update(self.input_nodes)
        self.dynamic_nodes = [self.nodes[node] for node in self.nodes if not self.nodes[node].static]
        self.static_nodes = [self.nodes[node] for node in self.nodes if self.nodes[node].static]

        for edge_0, edge_1, data in graph.edges(data=True):
            edge_args = {param: edge_params.get((edge_0, edge_1)) for param, edge_params in edge_parameters.items()}
            self.add_edge(self.nodes[edge_0], self.nodes[edge_1], **edge_args, **data)
            #self.add_edge(self.nodes[edge_0], self.nodes[edge_1], weight=weights[(edge_0, edge_1)] if weights is not None else data['weight'], **data)
        self.time_points = time_points
        self.node_parameters = {par: {self.nodes[node]: node_parameters[par][node] for node in node_parameters[par]} for par in node_parameters}
        # print({par: {(self.nodes[edge[0]], self.nodes[edge[1]], 0): edge_parameters[par][edge] for edge in edge_parameters[par]} for par in edge_parameters})
        self.edge_parameters = {par: {(self.nodes[edge[0]], self.nodes[edge[1]], 0): edge_parameters[par][edge] for edge in edge_parameters[par]} for par in edge_parameters}
        self.inputs = inputs
        if inputs is not None:
            self.set_inputs(inputs)

    def set_inputs(self, inputs):
        """Set the inputs to the input nodes."""
        assert all(isinstance(inp, Input) for inp in inputs)
        for inp in inputs:
            for node in inp.input_nodes:
                self.nodes[node].set_input(inp)
        self.inputs = inputs

    def remove_inputs(self):
        """Remove the inputs from the input nodes."""
        for node in self.input_nodes:
            self.nodes[node].inputs = []
        self.inputs = None
    
    def set_node_parameters(self, node_parameters):
        """Set the parameters of the nodes.""" 
        for par in self.node_parameters:
            for node in self.node_parameters[par]:
                if not node in self.nodes.values():
                    raise ValueError(f"Node {node} not found in the model")
                if node in node_parameters[par]:
                    node.set_parameter(par, node_parameters[par][node])
        self.update_node_parameters()

    def set_edge_parameters(self, edge_parameters):
        """Set the parameters of the edges."""
        for par in self.edge_parameters:
            for edge in self.edge_parameters[par]:
                if not edge in self.edges:
                    raise ValueError(f"Edge {edge} not found in the model")
                if edge in edge_parameters[par]:
                    nx.set_edge_attributes(self, {edge: {par: edge_parameters[par][edge]}})
        self.update_edge_parameters()
    
    def update_node_parameters(self):
        for n,node in self.nodes.items():
            for par in self.node_parameters:
                self.node_parameters[par][node] = node.node_parameters[par]

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
    def __init__(self, graph, input_nodes, weights=None, gains=None, time_constants=None, static_nodes=None, time_points=None, inputs=None) -> None:
        """
        Initialize the rate model.

        Args:
            graph (networkx.DiGraph): The graph representing the neural network.
            weights (dict): A dictionary where the keys are the edges and the values are the weights.
            input_nodes (list): A list of nodes that receive external input.
            gains (dict): A dictionary where the keys are the nodes and the values are the list of gain terms for each neuron.
            time_constants (dict): A dictionary where the keys are the nodes and the values are the list of time constants for each neuron.
            static_nodes (list): A list of nodes that are static.
            time_points (list): A list of time points.
        """
        # super().__init__(graph, input_nodes, weights, gains, time_constants, static_nodes, time_points)
        super().__init__(graph, input_nodes, node_parameters={'gain':gains, 'time_constant':time_constants}, edge_parameters={'weight':weights},  static_nodes=static_nodes, time_points=time_points, inputs=inputs)
        self.weights = {edge: self[edge[0]][edge[1]]['weight'] for edge in self.edges} if weights is None else weights
        self.gains = {node:self.graph.nodes[node]['gain'] for node in self.nodes} if gains is None else gains
        self.time_constants = {node:self.graph.nodes[node]['time_constant'] for node in self.nodes} if time_constants is None else time_constants
        
    def rate_equations(self, t):
        """
        Compute the derivatives of the rates with respect to time.

        Args:
            t (float): The current time.

        Returns:
            list: The derivatives of the rates with respect to time.
        """
        derivatives = np.zeros(len(self.dynamic_nodes))
        for i, node in enumerate(self.dynamic_nodes):
            derivative = -node.rate/node.time_constant
            if node.label in self.input_nodes:
                derivative += node.process_inputs(t)
            else:
                for in_node,_,data in self.in_edges(node, data=True):
                    derivative += node.gain * in_node.rate * data['weight']
            derivatives[i] = derivative
        return derivatives

    def simulate(self, time_points=None, initial_rates=None):
        """
        Simulate the rates over time.

        Args:
            time_points (list): The time points at which to simulate the rates.
            inputs (list): The inputs to the input nodes at each time point.

        Returns:
            list: The simulated rates at each time point.
        """
        if time_points is not None:
            self.time_points = time_points
        if self.time_points is None:
            raise ValueError("Time points must be set")
        if initial_rates is None:
            initial_rates = np.zeros(len(self.dynamic_nodes))
        simulated_rates = np.zeros((len(self.time_points), len(self.dynamic_nodes)))
        static_rates = np.zeros((len(self.time_points), len(self.static_nodes)))

        rates = {node: [] for node in self.static_nodes + self.dynamic_nodes}
        # ## Set inputs
        # assert all(isinstance(inp, Input) for inp in inputs)
        # self.set_inputs(inputs)

        ## Initial conditions
        t = 0
        static_rates[t]= [node.process_inputs(self.time_points[t]) for node in self.static_nodes]
        simulated_rates[t] = initial_rates

        for i, node in enumerate(self.dynamic_nodes):
            node.rate = simulated_rates[t,i]
        for i, node in enumerate(self.static_nodes):
            node.rate = static_rates[t,i]

        for t in range(len(self.time_points)-1):
            derivatives = self.rate_equations(self.time_points[t])
            simulated_rates[t+1] = simulated_rates[t] + derivatives * (self.time_points[t+1] - self.time_points[t])
            static_rates[t+1,:] = [node.process_inputs(self.time_points[t+1]) for node in self.static_nodes]

            for i, node in enumerate(self.dynamic_nodes):
                node.rate = simulated_rates[t+1,i]
            for i, node in enumerate(self.static_nodes):
                node.rate = static_rates[t+1,i]
        
        for i, node in enumerate(self.static_nodes):
            rates[node] = static_rates[:,i]
        for i, node in enumerate(self.dynamic_nodes):
            rates[node] = simulated_rates[:,i]

        # rates = np.concatenate((static_rates, simulated_rates), axis=1)
        return rates
    

    def reinitialize(self):
        pass