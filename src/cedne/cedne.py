"""
This module contains the main classes and functions for the CeDNe library.

Functions:
- generate_random_alphanumeric
- loadWorm

Classes:
- Worm
- Behavior
- NervousSystem
- Neuron
- NeuronGroup
- Trial
- StimResponse
- Connection
- Mapping
"""
import string
import random
import pickle
import json
import logging
import copy
import sys

import numpy as np
import networkx as nx
#from warnings import warn
import scipy.stats as ss
from scipy import signal

RECURSION_LIMIT = 10000
sys.setrecursionlimit(RECURSION_LIMIT)

# Superparameters
RANDOM_SEED = 42
F_SAMPLE = 5 # Hz

# Setting up log
logging.basicConfig(level=logging.INFO, filename='CeDNe.log'\
    , filemode='w', format='%(name)s - %(levelname)s - %(message)s')

# Restricting unpickling
ALLOWED_MODULES = [
    "cedne",
    "networkx"
]

class RestrictedUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        # Allow all functions and classes from the allowed modules and their submodules
        if any(module.startswith(allowed_module) for allowed_module in ALLOWED_MODULES):
            return getattr(__import__(module, fromlist=[name]), name)
        if module == "builtins" and name in ["set", "frozenset"]:
            return getattr(__import__(module), name)
        raise pickle.UnpicklingError(f"global '{module}.{name}' is forbidden")

def load_pickle(file):
    ''' Loading restricted pickles.'''
    return RestrictedUnpickler(file).load()

class Organism:
    ''' This is a full organism class'''
    def __init__(self, species = '', name='', stage='', sex='', genotype=''):
        ''' Initializes an Organism class'''
        self.species = species
        self.name = name
        self.stage = stage
        self.sex = sex
        self.genotype = genotype
        self.networks = {}

    def save(self, file_path, file_format='pickle'):
        """
        Saves the Organism object to a pickle file at the specified file path.

        Args:
            file_path (str): The path to the pickle file.
        """
        if file_format == 'pickle':
            if not file_path.endswith('.pkl'):
                file_path += '.pkl'
            with open(file_path, 'wb') as pickle_file:
                pickle.dump(self, pickle_file, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            raise NotImplementedError("Only pickle format is supported.")

class Worm(Organism):
    ''' This is an explicit Worm class, a container for network(s).'''
    def __init__(self, name='', stage='Day-1 Adult', sex='Hermaphrodite', genotype='N2') -> None:
        """
        Initializes a Worm object.

        Parameters:
            name (str): The name of the worm. If empty,
                a random alphanumeric string will be generated.
            stage (str): The stage of the worm. Default is 'Day-1 Adult'.
              Other options can be L1, L2, L3, L4, Day-2 Adult, etc.
            sex (str): The sex of the worm. Default is 'Hermaphrodite'.
                Other options can be 'Male', 'Feminized male', etc.
            genotype (str): The genotype of the worm. Default is 'N2'.
                Other options can be mutant names or other wild types, etc.

        Returns:
            None
        """
        if not name:
            name = 'Worm-' + generate_random_string()
        super().__init__(species='Caenorhabditis elegans', name=name, stage=stage, sex=sex, genotype=genotype)

class Fly(Organism):
    ''' This is an explicit Fly class, a container for network(s).'''
    def __init__(self, name='', stage='Day-7 Adult', sex='Female', genotype='w1118 x Canton-S G1') -> None:
        """
        Initializes a Fly object.

        Parameters:
            name (str): The name of the Fly. If empty,
                a random alphanumeric string will be generated.
            stage (str): The stage of the fly. Default is 'Day-7 Adult'.
              Other options can be E, L1, L2, P1, Day-2 Adult, etc.
            sex (str): The sex of the fly. Default is 'Female'.
                Other options can be 'Male', 'Feminized Male', etc.
            genotype (str): The genotype of the worm. Default is 'w1118 x Canton-S G1'.
                Other options can be mutant names or other wild-types.

        Returns:
            None
        """
        if not name:
            name = 'Fly-' + generate_random_string()
        super().__init__(species='Drosophila melanogaster', name=name, stage=stage, sex=sex, genotype=genotype)


def load_worm(file_path):
    """
    Load a Worm object from a pickle file.

    Args:
        file_path (str): The path to the pickle file.

    Returns:
        Worm: The loaded Worm object.
    """
    try:
        with open(file_path, 'rb') as pickle_file:
            # return pickle.load(pickle_file)
            return load_pickle(pickle_file)
    except:
        raise Exception(f"Failed to load {file_path}.")

class Behavior:
    ''' This is a behavior class for the organism'''
    def __init__(self, worm: Worm = None, network: str = "Neutral") -> None:
        """
        Initializes a Behavior object.

        Args:
            worm (Worm, optional): The worm object associated with the behavior. Defaults to None.
            network (str, optional): The network for the behavior. Defaults to "Neutral".
        """
        self.worm = worm or Worm()
        if self.worm.networks.get(network) is None:
            self.worm.networks[network] = self

class NervousSystem(nx.MultiDiGraph):
    '''
    This is the Nervous System class. This inherits from networkx.MultiDiGraph
      and is the main high level class for the nervous system. '''
    def __init__(self, worm: Worm = None, network: str = "Neutral") -> None:
        """
        Initializes the NervousSystem object with the given worm and network.

        Args:
            worm (Worm, optional): The worm object associated with the nervous system.
            Defaults to None.
            network (str, optional): The network for the nervous system. Can be different
            conditions or network types.
            Defaults to "Neutral".
        """
        super().__init__()

        self.worm = worm or Worm()
        self.name = network
        self.worm.networks[network] = self
        self.groups = {}

        self.neurons = NeuronGroup(self, group_name='all_neurons')  \
            # dictionary of all neurons in the nervous system
        self.connections = ConnectionGroup(self, group_name='all_connections') \
              # dictionary of all connections in the nervous system

        self._filtered_nodes = set()
        self._filtered_edges = set()

    @property
    def num_groups(self):
        """
        Returns the current number of Neuron Groups for the Nervous System.
        """
        return len(self.groups)

    def build_nervous_system(self, neuron_data, chem_synapses, elec_synapses, positions, chem_only=False, gapjn_only=False):
        """
        Builds the nervous system by loading pickle files containing neuron data, chemical synapses,
        electrical synapses, and positions.

        Args:
            neuron_data (str): 
                The path to the pickle file containing neuron data.
            chem_synapses (str): 
                The path to the pickle file containing chemical synapses.
            elec_synapses (str): 
                The path to the pickle file containing electrical synapses.
            positions (str): 
                The path to the pickle file containing positions.

        Returns:
            None

        Raises:
            FileNotFoundError: If any of the pickle files do not exist.

        Description:
            This function loads the pickle files containing neuron data, chemical synapses,
            electrical synapses, and positions. It then extracts the necessary information
            from the pickle files and uses it to create neurons, set up chemical connections,
            and set up gap junctions.

        """
        with open(neuron_data, 'rb') as neuron_file, \
             open(chem_synapses, 'rb') as chem_file, \
             open(elec_synapses, 'rb') as elec_file, \
             open(positions, 'rb') as positions_file:

            neuron_info = pickle.load(neuron_file)
            chem_adjacency = pickle.load(chem_file)
            elec_adjacency = pickle.load(elec_file)
            locations = pickle.load(positions_file)

            labels, neuron_types, categories, modalities = neuron_info.iloc[:,0].to_list(), \
                                                    neuron_info.iloc[:,1].to_list(), \
                                                    neuron_info.iloc[:,2].to_list(), \
                                                    neuron_info.iloc[:,3].to_list()

            self.create_neurons(labels, type=neuron_types, category=categories, modality=modalities, position=locations)
            assert not all([gapjn_only, chem_only]), "Select at most one of gapjn_only or chem_only attributes to be True."
            if not gapjn_only:
                self.setup_chemical_connections(chem_adjacency)
            if not chem_only:
                self.setup_gap_junctions(elec_adjacency)
    def build_network(self, neuron_data, adj, label):
        """
        Make a network with the neurons
        Args:
            neurons: 
                The file containing neuron information
            adj: 
                The adjacency matrix
            label: 
                The label for the network
        """
        with open(neuron_data, 'rb') as neuron_file:
            node_dict = pickle.load(neuron_file)
            node_labels, l1_list, l2_list, l3_list = node_dict.iloc[:,0].to_list(),\
                  node_dict.iloc[:,1].to_list(), \
                    node_dict.iloc[:,2].to_list(), \
                        node_dict.iloc[:,3].to_list()
            self.create_neurons(node_labels, type=l1_list, category=l2_list, modality=l3_list)
        self.setup_connections(adj, label)

    def create_neurons(self, labels, **kwargs):
        """
        Creates a set of Neuron objects based on the given labels,
          types, categories, modalities, and positions.

        Args:
            labels (list): 
                A list of labels for the neurons.
            neuron_types (list, optional): 
                A list of types for the neurons. Defaults to None.
            categories (list, optional): 
                A list of categories for the neurons. Defaults to None.
            modalities (list, optional): 
                A list of modalities for the neurons. Defaults to None.
            positions (dict, optional): 
                A dictionary mapping labels to positions. Defaults to None.

        Returns:
            None

        This function iterates over the labels, types, categories, modalities, and positions
        using the zip function. For each combination, it checks if the label is present in the
        positions dictionary. If it is, it creates a Neuron object with the given label, type,
        category, modality, and position. Otherwise, it creates a Neuron object with the given
        label, type, category, and modality. The Neuron object is then added to the neurons
        dictionary with the label as the key.
        """
        network_args = {}
        for key, value in kwargs.items():
            if isinstance(value, dict):
                if not all([k in labels for k in value.keys()]):
                    raise ValueError(f"{key}: Dictionary keys must be one of neuron labels")
                network_args[key] = {lab:value[lab] if lab in value else None for lab in labels}
            elif isinstance(value, int) or isinstance(value, str) or isinstance(value, float) or isinstance(value, bool):
                network_args[key] = {lab:value for lab in labels}
            elif hasattr(value, '__len__'):
                if not len(value)==len(labels):
                    raise ValueError(f"{key} must be same length as neuron labels")
                network_args[key] = {lab:val for (lab,val) in zip(labels, value)}  
            else:
                raise NotImplementedError(f"Attribute setting not implemented for datatype {type(value)}.")
        
        
        for label in labels:
            neuron_args = {}
            for key, value in network_args.items():
                neuron_args[key] = value[label]
            #print(label, self, neuron_args)
            neuron = Neuron(label, self, **neuron_args)

        # neuron_types=None, categories=None, modalities=None, positions=None

        # neuron_types = [None] * len(labels) if neuron_types is None else neuron_types
        # categories = [None] * len(labels) if categories is None else categories
        # modalities = [None] * len(labels) if modalities is None else modalities
        # positions = {} if positions is None else positions

        # for label, neuron_type, category, modality in \
        #     zip(labels, neuron_types, categories, modalities):
        #     if label in positions:
        #         neuron = Neuron(label, self, neuron_type, category, modality, positions[label])
        #     else:
        #         neuron = Neuron(label, self, neuron_type, category, modality)
            self.neurons[label] = neuron
    
    def create_neurons_from(self, network, data=False):
        """ 
        Creates a set of Neuron objects based on the given network.
        Args:
            network (Network): 
                A Network object.
            data (bool, optional): 
                A flag indicating whether to include data in the Neuron objects. Defaults to False.
        """
        ## Check if network object is a NervousSystem object
        if not isinstance(network, NervousSystem):
            raise TypeError("The network object must be a NervousSystem object")
        if not data:
            for node in network.nodes:
                self.neurons.update({node.name:Neuron(node.name, self)})
        else:
            for node,data in network.nodes(data=True):
                self.neurons.update({node.name:Neuron(node.name, self, **data)})

    def create_connections_from(self, network, data=False):
        """
        Creates a set of Connection objects based on the given network.
        Args:
            network (Network): 
                A Network object.
            data (bool, optional): 
                A flag indicating whether to include data in the Connection objects.\
                    Defaults to False.
        """
        ## Check if network object is a NervousSystem object
        if not isinstance(network, NervousSystem):
            raise TypeError("The network object must be a NervousSystem object")
        
        for u,v,k,data in network.edges(keys=True, data=True):
            n1 = self.neurons[u.name]
            n2 = self.neurons[v.name]
            if not data:
                self.connections.update({(n1,n2,k):Connection(n1, n2, k)})
            else:
                self.connections.update({(n1,n2,k):Connection(n1, n2, k, **data)})

    def update_neurons(self):
        """
        Update the dictionary of neurons
        """
        self.neurons.clear()
        for node in self.nodes:
            self.neurons.update({node.name:node})

    def update_connections(self):
        """
        Update the dictionary of connections. Need more precaution here.
        """
        #print({connection_id: self.connections[connection_id] for connection_id in self.connections})
        pop_conns = [] ## Smells good ;)!
        for connection_id in self.connections:
            if not connection_id in self.edges:
                pop_conns.append(connection_id)
        for pop_conn in pop_conns:
            if pop_conn in self.connections:
                self.connections.pop(pop_conn)
        for n in self.neurons:
            self.neurons[n].update_connections()

    def setup_connections(self, adjacency, edge_type, input_type = 'adjacency', **kwargs):
        """
        Set up connections between neurons based on the adjacency matrix and edge type.
        """
        if input_type == 'adjacency':
            for source_id, neighbors in adjacency.items():
                for target_id, properties in neighbors.items():
                    if 'weight' in properties:
                        if properties['weight'] == 0:
                            continue
                        else:
                            source_neuron = self.neurons[source_id]
                            target_neuron = self.neurons[target_id]
                            edge_weight = properties['weight']
                    else:
                        source_neuron = self.neurons[source_id]
                        target_neuron = self.neurons[target_id]
                        edge_weight = 1
                    
                    # edge_id = self.add_edge(
                    #         source_neuron, target_neuron,
                    #         weight=edge_weight, color='k', edge_type=edge_type
                    #     )
                    connection = Connection(
                        #source_neuron, target_neuron, edge_id, edge_type, weight=edge_weight
                        source_neuron, target_neuron, edge_type=edge_type, weight=edge_weight
                    )
                    self.connections[(source_neuron, target_neuron, connection.uid)] = connection
        

        elif input_type == 'edge':
            source_neuron = self.neurons[adjacency['pre']]
            target_neuron = self.neurons[adjacency['post']]
            edge_weight = adjacency['weight']

            # edge_id = self.add_edge(
            #                 source_neuron, target_neuron,
            #                 weight=edge_weight, color='k', edge_type=edge_type
            # )

            # connection = Connection(source_neuron, target_neuron, edge_id, edge_type,\
            #                          weight=edge_weight, **kwargs)
            connection = Connection(source_neuron, target_neuron, edge_type=edge_type,\
                                     weight=edge_weight, **kwargs)
            self.connections[(source_neuron, target_neuron, connection.uid)] = connection
        
        else:
            raise NotImplementedError("Not implemented for this input type. Try 'adjacency' or 'edge'.")

    def setup_chemical_connections(self, chemical_adjacency, **kwargs):
        """
        Set up chemical connections in the network based on the given adjacency dictionary.

        Parameters:
            chemical_adjacency (dict): A dictionary representing the adjacency of chemical synapses.
                The keys are source neurons and the values are dictionaries where the keys are
                target neurons and the values are dictionaries containing the connection data.

        Returns:
            None

        This function iterates over the `chemical_adjacency` dictionary and adds chemical synapse
        edges between source neurons and target neurons if the connection weight is greater than 0.
        It uses the `add_edge` method to add the edge to the network and creates a `Connection`
        object to store the connection details. The created connection is added to the `connections`
        dictionary using a tuple of the source neuron, target neuron, and edge key as the key.
        """
        edge_type='chemical-synapse'
        for source_neuron, target_neurons in chemical_adjacency.items():
            for target_neuron, connection_data in target_neurons.items():
                if connection_data['weight'] > 0:
                    # edge_key = self.add_edge(
                    #     self.neurons[source_neuron],
                    #     self.neurons[target_neuron],
                    #     weight=connection_data['weight'],
                    #     color='orange',
                    #     edgeType=edge_type
                    # )
                    connection = ChemicalSynapse(
                        self.neurons[source_neuron],
                        self.neurons[target_neuron],
                        #edge_key,
                        edge_type=edge_type,
                        weight=connection_data['weight'],
                        color='orange',
                        **kwargs
                    )
                    self.connections[(self.neurons[source_neuron], \
                                      self.neurons[target_neuron], connection.uid)] = connection
    #self.add_edges_from(e) # Add edge attributes here.

    def setup_gap_junctions(self, gap_junction_adjacency):
        """
        Set up gap junctions in the network based on the given adjacency dictionary.

        Parameters:
            gap_junction_adjacency (dict): A dictionary representing the adjacency of gap junctions.
                The keys are source neurons and the values are dictionaries where the keys are
                target neurons and the values are dictionaries containing the connection data.

        Returns:
            None

        This function iterates over the `gap_junction_adjacency` dictionary and adds gap junction
        edges between source neurons and target neurons if the connection weight is greater than 0.
        It uses the `add_edge` method to add the edge to the network and creates a `Connection`
        object to store the connection details. The created connection is added to the `connections`
        dictionary using a tuple of the source neuron, target neuron, and edge key as the key.

        Note:
            - The `add_edge` method is assumed to be defined in the class.
            - The `Connection` class is assumed to be defined in the class.
            - The `neurons` dictionary is assumed to be defined in the class.
        """
        edge_type = 'gap-junction'
        for source_neuron, target_neurons in gap_junction_adjacency.items():
            for target_neuron, connection_data in target_neurons.items():
                if connection_data['weight'] > 0:
                    # edge_key = self.add_edge(
                    #     self.neurons[source_neuron],
                    #     self.neurons[target_neuron],
                    #     weight=connection_data['weight'],
                    #     color='gray',
                    #     edge_type=edge_type
                    # )
                    connection = GapJunction(
                        self.neurons[source_neuron],
                        self.neurons[target_neuron],
                        #edge_key,
                        edge_type=edge_type,
                        color='gray',
                        weight=connection_data['weight']
                    )
                    self.connections[(
                        self.neurons[source_neuron],
                        self.neurons[target_neuron],
                        connection.uid
                    )] = connection
    def load_neuron_data(self, file, file_format='summary-xlsx'):
        ''' Standard formats to load data into the network'''
        #pass

    def load_connection_data(self, file, file_format='summary-xlsx'):
        ''' Standard formats to load data into the network'''
        #pass

    def subnetwork(self, neuron_names=None, name=None, connections=None, as_view=False):
        """
        Generates a subgraph of the network based on the given list of neuron names.

        Args:
            neuron_names (List[str]): List of neuron names to include in the subgraph.
            connections (List[tuple]): List of connections to include in the subgraph.

        Returns:
            NervousSystem: A deep copy of the subgraph generated from the neuron_names
            or connections. The subgraph contains a dictionary of neurons with their
            names as keys.
        """

        
        if not as_view:
            graph_copy = self.copy(copy_type='deep_custom', name=name)
            assert not (neuron_names and connections),\
            "Specify either neuron_names or connections, not both."

            if neuron_names is not None:
                subgraph_nodes = [graph_copy.neurons[name] for name in neuron_names]
            elif connections is not None:
                new_connections = [(graph_copy.neurons[conn[0].name], graph_copy.neurons[conn[1].name], conn[2])\
                                 for conn in connections]
                new_connections = [graph_copy.connections[key]._id for key in new_connections]
            else:
                subgraph = self

            if neuron_names is not None:    
                subgraph = graph_copy.subgraph(subgraph_nodes)
                subgraph.connections = {key: value for key, value in graph_copy.connections.items()\
                                         if key[0] in subgraph_nodes and key[1] in subgraph_nodes}
            elif connections is not None:
                subgraph = graph_copy.edge_subgraph(new_connections)
                subgraph.connections = {key: value for key, value in graph_copy.connections.items()\
                                         if key in new_connections}
            
            subgraph.update_neurons()
            subgraph.update_connections()

        else:
            if neuron_names is not None:
                filter_neurons = [self.neurons[name] for name in neuron_names]
                subgraph = self.subgraph_view(filter_neurons=filter_neurons)
            elif connections is not None:
                filter_neurons = list(set([neu for c in connections for neu in [c[0], c[1]]]))
                subgraph = self.subgraph_view(filter_connections=connections, filter_neurons=filter_neurons)
            else:
                subgraph = self
        return subgraph #subgraph.copy(as_view)
    
    def join_networks(self, networks, mode='consensus'):
        ''' Goes through the list of networks and joins them to the current graph.'''
        assert all([isinstance(network, NervousSystem) for network in networks])
        assert len(set([network.name for network in networks])) == len(networks)
        joined_networks = (network.name for network in networks)
        all_neurons = {self.name: self.neurons}
        all_connections = {self.name: self.connections}
        
        combined_network = NervousSystem(network=f"{'-'.join(joined_networks)}")
        for network in networks:
            all_neurons[network.name] = network.neurons
            all_connections[network.name] = network.connections
        
        if mode == 'consensus':
            # All neurons that are common between networks are chosen. All edges that are common are picked. 
            # The weight of the edge is the average of weights of all networks.
            neuron_set = [set(neuron for neuron in network.neurons) for _netname, network in all_neurons.items()]
            connection_set = [set((edge[0].name, edge[1].name, edge[2]) for edge in network.connections.keys()) for _netname, network in all_connections.items()]
            #print(set(neuron.name for neuron in network.neurons) for _netname, network in all_neurons.items() )
            joined_neurs = set.intersection(*neuron_set)
            joined_conns = set.intersection(*connection_set)
            combined_network.create_neurons(joined_neurs)
            
            for edge in joined_conns:
                source_neuron, target_neuron, edge_type = edge
                weights = [connections.connections[(connections.network.neurons[edge[0]], connections.network.neurons[edge[1]], edge[2])].weight for _netname, connections in all_connections.items()]
                
                #edge_weight = np.mean([all_connections[netname].connections[ (all_connections[netname].neurons[source_neuron], all_connections[netname].neurons[target_neuron], edge_type) ].weight for netname in all_connections.keys()])
                edge_weight = np.mean(weights)
                # edge_id = combined_network.add_edge(
                #                 combined_network.neurons[source_neuron], combined_network.neurons[target_neuron],
                #                 weight=edge_weight, color='k', edge_type=edge_type
                            # )
                # connection = Connection(
                #             combined_network.neurons[source_neuron], combined_network.neurons[target_neuron], edge_id, edge_type, weight=edge_weight
                #         )
                connection = Connection(
                            combined_network.neurons[source_neuron], combined_network.neurons[target_neuron], edge_type=edge_type, weight=edge_weight)
                connection.set_property('joined_networks', joined_networks)
                combined_network.connections[(combined_network.neurons[source_neuron], combined_network.neurons[target_neuron], connection.uid)] = connection
        return combined_network

    def fold_network(self, fold_by, name=None, data='collect', exceptions=None, self_loops=True):
        """
        Fold the network based on a filter.

        <TODO>
        
        !!! The fold_by can also take Neuron Group objects as input.

        </TODO>

        Args:
            fold_by (tuple): A tuple of length 2 specifying the neurons to fold.
                The first element is the neurons to fold, and the second element
                is the neurons that are exempt from folding.
                The tuple can contain any neuron name as a string.
            data (str, optional): The data to use for folding. Defaults to 'collect'.
                Available options are:
                    - 'collect': Collect the data together from all neurons in the fold_by
                        tuple, but keep them separate.
                    - 'union' : Union the data from all neurons in the fold_by tuple.
                    - 'intersect': Intersect the data from all neurons in the fold_by tuple.

        Returns:
            None

        Raises:
            AssertionError: If the length of fold_by is not 2.

        Notes:
            This function folds the network by contracting the specified neurons.
            The neurons specified in exceptions will not be folded.
        """
        assert isinstance(fold_by, dict), "Enter a dictionary with neuron class\
            names as keys and the neurons to fold as values. If there is only one\
                neuron in the list of values, the neuron will be renamed to the key."
        
        graph_copy = self.copy(copy_type='deep_custom', name=name)
        if exceptions is None:
            exceptions = []
        for merged_nodename, nodes_to_fold in fold_by.items():
            if len(nodes_to_fold) >1:
                merged_node = nodes_to_fold[0]
                for j in range(1,len(nodes_to_fold)):
                    npair = (merged_node, nodes_to_fold[j])
                    if not npair[0] in exceptions and not npair[1] in exceptions:
                        #self.contract_neurons(npair, merged_nodename, data=data)
                        graph_copy.contract_neurons(npair, merged_nodename, data=data, self_loops=self_loops)
                        merged_node = merged_nodename
            else:
                graph_copy.neurons[nodes_to_fold[0]].name = merged_nodename
        
        graph_copy.update_neurons()
        graph_copy.reassign_connections()

        if data == 'collect':
            return graph_copy
        if data == 'clean':
            parsed_conns = {}
            for e,conn in graph_copy.connections.items():
                if (e[0],e[1], conn.edge_type) not in parsed_conns:
                    parsed_conns[(e[0],e[1], conn.edge_type)] = []
                parsed_conns[(e[0],e[1], conn.edge_type)].append(conn)
            contracted_graph = graph_copy.contract_connections(parsed_conns)
            return contracted_graph


            # if data == 'collect':
            #     return self
            # elif data == 'union':
            #     pass
            # elif data == 'intersect':
            #     pass
            # else:
            #     raise ValueError("data condition must be 'collect', 'union' or 'intersect'.")

    # def reassign_nodes(self):
    #     self.update_neurons()
            
    def reassign_connections(self):
        """ 
        Reassign connections after folding based on the folding _ids and correcting connection names.
        """
        self._connections = {}
        for e in self.edges(data=True, keys=True):
            if '_id' in e[3]:
                self._connections.update({(e[0], e[1], e[2]): self.connections[e[3]['_id']]})
                self._connections[(e[0], e[1], e[2])].pre = e[0]
                self._connections[(e[0], e[1], e[2])].post = e[1]
                self._connections[(e[0], e[1], e[2])]._id = (e[0], e[1], e[2])
                
                del e[3]['_id']
            else:
                self._connections.update({(e[0], e[1], e[2]): self.connections[((e[0], e[1], e[2]))]})
        self.connections = self._connections
        self.update_connections()
        # for e in self.in_edges(self.neurons[contracted_name], keys=True, data=True):
        #     self.connections.update({(e[0], e[1], e[2]): self.connections[e[3]['_id']]})
        # for e in self.out_edges(self.neurons[contracted_name], keys=True, data=True):
        #     self.connections.update({(e[0], e[1], e[2]): self.connections[e[3]['_id']]})
        
    def contract_neurons(self, pair, contracted_name, data='collect', copy_graph=False, self_loops=True):
        """
        Contract two neurons together. Currently, data from other nodes is stored in the contraction
        attribute on each contracted node. Attributes are currently carried on from the source neuron
        to the contraction.

        Args:
            pair (tuple): 
                Pair of neuron names to contract.
            copy_graph (bool): 
                If True, returns a new graph with the contraction.
                Otherwise, modifies the current graph.

        Returns:
            NervousSystem: 
                A deep copy of the subgraph generated from the neuron_names.
                The subgraph contains a dictionary of neurons with their names as keys.
                Only returned if copy_graph is True.
        """
        source_neuron, target_neuron = pair
        if copy_graph:
            new_graph = self.copy()
            new_graph = new_graph.contract_neurons((source_neuron, target_neuron, contracted_name)\
                                                   , copy_graph=False)
            return new_graph

        for _cid, conn in self.neurons[source_neuron].get_connections().items():
            conn.set_property('_id', conn._id)
        for _cid, conn in self.neurons[target_neuron].get_connections().items():
            conn.set_property('_id', conn._id)
        nx.contracted_nodes(self, self.neurons[source_neuron], self.neurons[target_neuron],\
                                copy=copy_graph, self_loops=self_loops)
        self.neurons[source_neuron].name = contracted_name
        self.update_neurons()
    
    def contract_connections(self, contraction_dict):
        """
        Contracts the connections into a single connection and modifies the graph accordingly.
        """
        #empty_graph_copy = nx.create_empty_copy(self, with_data=True)
        empty_graph_copy = NervousSystem(self.worm, self.name + "_copy")
        empty_graph_copy.create_neurons_from(self, data=True)
        _connections = {}
        for contraction, conns in contraction_dict.items():
            contraction_data = {}
            weight = 0
            for conn in conns:
                weight+=conn.weight
                contraction_data[conn._id] = conn
                
            #uid = empty_graph_copy.add_edge(contraction[0], contraction[1], weight=weight, edge_type=contraction[2])
            #new_conn = Connection(contraction[0], contraction[1], uid=uid, edge_type=contraction[2], weight=weight)
            n1 = empty_graph_copy.neurons[contraction[0].name]
            n2 = empty_graph_copy.neurons[contraction[1].name]
            new_conn = Connection(n1, n2, edge_type=contraction[2], weight=weight)
            new_conn.set_property('contraction_data', copy.copy(contraction_data))
            #_connections[(contraction[0], contraction[1], new_conn.uid)] = new_conn
            _connections[(n1,n2, new_conn.uid)] = new_conn
        empty_graph_copy.connections = _connections

        empty_graph_copy.update_connections()
        empty_graph_copy.update_neurons()
        
        return empty_graph_copy

    def copy_data_from(self, nervous_system):
        """
        Copies data from another nervous system to this one.

        Args:
            nervous_system (NervousSystem): 
                The nervous system to copy data from.
        Returns:
            None
        """

    def neurons_have(self, key):
        ''' Returns neuron attributes'''
        return nx.get_node_attributes(self, key)

    def connections_have(self, key):
        ''' Gets connection attributes'''
        return nx.get_edge_attributes(self, key)

    def connections_between(self, neuron1, neuron2, directed=True):
        ''' Returns connections between neurons in neuron list.'''
        if directed:
            return neuron1.get_connections(neuron2, direction='out')
        else:
            return neuron1.get_connections(neuron2)

    def __filter_node__(self, node):
        """
        Checks if a specific node is filtered within the network.

        Parameters:
            node (Any): The node to check for filtering.

        Returns:
            bool: True if the node is filtered, False otherwise.
        """
        return node in self._filtered_nodes

    def __filter_edge__(self, neuron_1,neuron_2,key):
        """
        Checks if a specific edge is filtered within the network.

        Parameters:
            n1: 
                The starting node of the edge.
            n2: 
                The ending node of the edge.
            key: 
                The key identifying the edge.

        Returns:
            Boolean: True if the edge is in the filtered edges, False otherwise.
        """
        return (neuron_1,neuron_2,key) in self._filtered_edges

    def return_network_where(self, neurons_have=None, connections_have=None, condition='AND'):
        """
        Returns a subgraph view of the current network based on the specified conditions.

        Parameters:
            neurons_have (dict): 
                A dictionary of neuron attributes and their corresponding values.
                The subgraph will only include neurons that have all the specified attributes
                and values. Defaults to an empty dictionary.
            connections_have (dict):
                A dictionary of connection attributes and their corresponding
                values. The subgraph will only include connections that have all the specified
                attributes and values. Defaults to an empty dictionary.
            condition (str): 
                The condition to apply when filtering neurons and connections.
                Can be 'AND' or 'OR'. Default is 'AND'.

        Returns:
            networkx.classes.Graph: A subgraph view of the current network that satisfies
            the specified conditions.
        """
        ## First filter the neurons
        if neurons_have is None:
            neurons_have = {}
        if connections_have is None:
            connections_have = {}

        total_node_list = []
        filtered_node_list = None
        filtered_edge_list = None

        if len(neurons_have):
            for (key, value) in neurons_have.items():
                each_filter = []
                for node, val in self.neurons_have(key).items():
                    if val==value:
                        each_filter.append(node)
                total_node_list.append(each_filter)
            if condition=='AND':
                filtered_node_list = set([node for _n,node in self.neurons.items()\
                                       if all(node in sublist for sublist in total_node_list)])

            elif condition=='OR':
                filtered_node_list = set([node for _n,node in self.neurons.items()\
                                       if any(node in sublist for sublist in total_node_list)])
            else:
                raise ValueError("condition must be 'AND' or 'OR'")
       
        ## Then filter the connections
        total_edge_list = []
        if len(connections_have):
            for (key, value) in connections_have.items():
                each_filter = []
                for edge, val in self.connections_have(key).items():
                    if val==value:
                        each_filter.append(edge)
                total_edge_list.append(each_filter)
            #print(totalList)
            if condition=='AND':
                filtered_edge_list = set([edge for _e,edge in self.connections.items()\
                                       if all(_e in sublist for sublist in total_edge_list)])

            elif condition=='OR':
                filtered_edge_list = set([edge for _e,edge in self.connections.items()\
                                       if any(_e in sublist for sublist in total_edge_list)])
            else:
                raise ValueError("condition must be 'AND' or 'OR'")

        return self.subgraph_view(filter_neurons=filtered_node_list, filter_connections=filtered_edge_list)


    def copy(self, name=None, copy_type='deep'):
        """
        Returns a deep copy of the Nervous System object.

        Parameters:
            as_view (bool): 
                If True, the copy will be a view of the original graph.
                Default is False.

        Returns:
            object: 
                a deep copy of the Nervous System object.
        """
        if copy_type=='shallow':
            return super().copy(as_view=False)
        elif copy_type=='deep':
            return copy.deepcopy(self)
        elif copy_type == 'deep_custom':
            deep_copy = NervousSystem(self.worm, network=name or self.name + "_copy")
            deep_copy.create_neurons_from(self, data=True)
            deep_copy.create_connections_from(self, data=True)
            return deep_copy
        else:
            raise ValueError("copy_type must be 'deep', 'shallow'")

    def subgraph_view(self, filter_neurons=None, filter_connections=None):
        ''' Creates a read only view of a subgraph'''
        
        if not filter_neurons:
            self._filtered_edges = filter_connections
            return nx.subgraph_view(self, filter_edge=self.__filter_edge__)
        if not filter_connections:
            self._filtered_nodes = filter_neurons
            return nx.subgraph_view(self,filter_node=self.__filter_node__)
        self._filtered_nodes = filter_neurons
        self._filtered_edges = filter_connections
        return nx.subgraph_view(self,filter_node=self.__filter_node__, filter_edge=self.__filter_edge__)

    def search_motifs(self, motif):
        """
        Search for a motif in the network structure.
        """
        matcher = nx.algorithms.isomorphism.DiGraphMatcher(self, motif)
        motif_graphs = []
        for subgraph in matcher.subgraph_isomorphisms_iter():
            subgraph_inverse = {motif_node: node for node, motif_node in subgraph.items()}
            members = {edge:(subgraph_inverse[edge[0]], subgraph_inverse[edge[1]]) for edge in motif.edges}
            motif_graphs.append(members)
        return motif_graphs

    def export_graph(self, path, fmt= 'dot'):
        """
        Exports the graph to the specified path.

        Parameters:
            path (str): 
                The path to save the exported graph.

        Returns:
            None
        """
        if fmt == 'dot':
            nx.drawing.nx_pydot.write_dot(self, path)
        elif fmt == 'graphviz':
            nx.drawing.nx_agraph.write_dot(self, path)
        elif fmt == 'nx':
            nx.write_graphml(self, path)
        elif fmt == 'json':
            with open(path, 'w', encoding='utf-8') as f:
                jn = nx.cytoscape_data(self, path)
                json.dump(jn, f, ensure_ascii=False, indent=4)
        elif fmt == 'gml':
            nx.write_gml(self, path)
        elif fmt == 'graphml':
            nx.write_graphml(self, path)
        else:
            raise ValueError("format must be 'dot', 'graphviz', 'nx', 'json', 'gml', or 'graphml'")

    def remove_unconnected_neurons(self):
        """
        Removes neurons that are not connected to any other neurons.

        Returns:
            None
        """
        self.remove_nodes_from(list(nx.isolates(self)))
        self.update_neurons()

    def make_neuron_group(self, members, group_name=None):
        """
        Creates a neuron group with the specified members.

        Parameters:
            members (List[str]): 
                The list of members in the neuron group.
            groupname (str): 
                The name of the neuron group. Defaults to None.
            group_id (int, optional): 
                The ID of the neuron group. Defaults to 0.

        Returns:
            NeuronGroup: The created neuron group.
        """
        return NeuronGroup(self, members, group_name)

    def delete_neuron_group(self, groupname):
        """
        Deletes a neuron group with the specified name.

        Parameters:
            groupname (str): The name of the neuron group to be deleted.

        Returns:
            None
        """
        del self.groups[groupname]

    def make_connection_group(self, members, group_name=None):
        """
        Creates a connection group with the specified members.

        Parameters:
            members (List[str]): 
                The list of members in the connection group.
            groupname (str): 
                The name of the connection group. Defaults to None.
            group_id (int, optional): 
                The ID of the connection group. Defaults to 0.

        Returns:
            ConnectionGroup: The created connection group.
        """
        return ConnectionGroup(self, members, group_name)

    def delete_connection_group(self, groupname):
        """
        Deletes a connection group with the specified name.

        Parameters:
            groupname (str): The name of the connection group to be deleted.

        Returns:
            None
        """
        del self.groups[groupname]

    def __delete__(self, neuron):
        """
        Deletes the object from the network.
        """
        self.remove_node(neuron)
        self.update_neurons()

class NeuronGroup:
    ''' This contains a group of neurons in the network'''
    def __init__(self, network, members=None, group_name=None) -> None:
        """
        Initializes a new instance of the NeuronGroup class.

        Parameters:
            groupname (str): 
                The name of the neuron group.
            members (List[str]): 
                The list of members in the neuron group.
            group_id (int, optional): 
                The ID of the neuron group. Defaults to 0.

        Returns:
            None
        """
        if group_name is None:
            self.group_name = 'Group-'+ generate_random_string(8)
        else:
            self.group_name = group_name

        if members is None:
            members = []
        else:
            assert all([isinstance(m, Neuron)for m in members]), "Neuron group members must be\
                 of type Neuron"
        self.members = members
        self.neurons = {m.name: m for m in members}
        self.network = network
        assert self.group_name not in self.network.groups, f"Group name {self.group_name}\
            already exists in the network"
        self.network.groups.update({self.group_name: self})

    def __iter__(self):
        """
        Returns an iterator over the members of the group.
        """
        return iter(self.neurons)

    def items(self):
        """
        Returns an iterator over the members of the group.
        """
        for key, value in self.neurons.items():
            yield key, value
    def keys(self):
        """
        Returns an iterator over the members of the group.
        """
        return list(self.neurons.keys())
    def values(self):
        """
        Returns an iterator over the members of the group.
        """
        return list(self.neurons.values())

    def __len__(self):
        """
        Returns the number of members in the group.
        """
        return len(self.neurons)

    def __contains__(self, neuron):
        """
        Returns True if the neuron with the specified name is in the group, False otherwise.
        """
        return neuron in self.neurons

    def __getitem__(self, neuron_name):
        """
        Returns the neuron with the specified name in the group.
        """
        return self.neurons[neuron_name]

    def __setitem__(self, neuron_name, neuron):
        """
        Sets the neuron with the specified name in the group.
        """
        assert isinstance(neuron, Neuron), "Neuron group members must be of type Neuron"
        self.neurons[neuron_name] = neuron

    def clear(self):
        """
        Removes all neurons from the group.
        """
        self.neurons = {}
        self.members = []

    def update(self, member_dict):
        """
        Updates the list of members in the group.
        """
        assert all([isinstance(neuron, Neuron) for nname,neuron in member_dict.items()]),\
          "Neuron group members must be of type Neuron"
        self.neurons.update(member_dict)
        self.members = list(self.neurons.values())

    def pop(self, neuron_name):
        """
        Deletes the neuron with the specified name from the group.
        """
        self.neurons.pop(neuron_name)

    def set_property(self, property_name, property_value):
        """
        Sets a new property attribute for all neurons in the group.
        """
        for neuron in self.members:
            neuron.set_property(property_name, property_value)

    def get_property(self, property_name):
        """
        Returns the value of the specified property for all neurons in the group.
        """
        return [neuron.get_property(property_name) for neuron in self.members]

    def get_connections(self):
        """
        Returns a list of all connections in the group.
        """
        return [neuron.get_connections() for neuron in self.members]

class ConnectionGroup:
    ''' This is a group of connections in the network'''
    def __init__(self, network, members=None, group_name=None) -> None:
        """
        Initializes a new instance of the ConnectionGroup class.

        Parameters:
            groupname (str): 
                The name of the connection group.
            members (List[str]): 
                The list of members in the connection group.
            group_id (int, optional): 
                The ID of the neuron group. Defaults to 0.

        Returns:
            None
        """
        if group_name is None:
            self.group_name = 'Group-'+ generate_random_string(8)
        else:
            self.group_name = group_name

        if members is None:
            members = []
        else:
            assert all([isinstance(m, Connection)for m in members]),\
                  "Connection group members must be of type Connection"
        self.members = members
        self.connections = {m._id:m for m in members}
        self.network = network
        self.neurons = set([neuron for m in members for neuron in (m.pre, m.post)])
        assert self.group_name not in self.network.groups, \
        f"Group name {self.group_name} already exists in the network"
        self.network.groups.update({self.group_name: self})

    def __iter__(self):
        """
        Returns an iterator over the members of the group.
        """
        return iter(self.connections)

    def clear(self):
        """
        Removes all connections from the group.
        """
        self.connections = {}
        self.members = []

    def items(self):
        """ 
        Returns the itemized connection dictionary
        """
        return self.connections.items()

    def keys(self):
        """ Returns the IDs for the Connection Group"""
        return list(self.connections.keys())

    def values(self):
        """ Returns the IDs for the Connection Group"""
        return list(self.connections.values())
    
    def __len__(self):
        """
        Returns the number of members in the group.
        """
        return len(self.connections)

    def __contains__(self, member):
        """
        Returns True if the connection with the specified name is in the group, False otherwise.
        """
        assert isinstance(member, Connection) or isinstance(member, Neuron),\
            "Membership checking is limited to Connections and Neurons"
        return member in self.connections or member in self.neurons

    def __getitem__(self, connection_id):
        """
        Returns the connection with the specified name in the group.
        """
        return self.connections[connection_id]

    def __setitem__(self, connection_id, connection):
        """
        Sets the connection with the specified name in the group.
        """
        assert isinstance(connection, Connection), "Connection must be of type Connection"
        self.connections[connection_id] = connection

    def update(self, member_dict):
        """
        Updates the list of members in the group.
        """
        assert all([isinstance(connection, Connection) for ename, connection in \
                    member_dict.items()]), "Connection group members must be\
             of type Connection"
        self.connections.update(member_dict)
        self.members = list(self.connections.values())

    def pop(self, connection_id):
        """
        Deletes the connection with the specified name from the group.
        """
        self.connections.pop(connection_id)

    def set_property(self, property_name, property_value):
        """
        Sets a new property attribute for all connections in the group.
        """
        for connection in self.members:
            connection.set_property(property_name, property_value)

    def get_property(self, property_name):
        """
        Gets the property attribute for all connections in the group.
        """
        return [connection.get_property(property_name) for connection in self.members]

class Path(ConnectionGroup):
    ''' This is a sequence of Connections in the network.'''
    def __init__(self, network, members=None, group_name=None):
        if group_name is None:
            self.group_name = 'Path-'+ generate_random_string(8)
        else:
            self.group_name = group_name

        if members is None:
            members = []
        else:
            assert all([isinstance(m, Connection)for m in members]),\
                  "Path members must be of type Connection"
            assert all([members[j].post == members[j+1].pre for j in range(1,len(members)-1)]),\
                  "Path members must be continous connections from source to target"
        self.source = members[0].pre
        self.target = members[-1].post
        super().__init__(network, members, group_name)

    def update(self, member_dict):
        """
        Updates the list of members in the group.
        """
        raise NotImplementedError(f'Cannot update connections in {self.__class__.__name__}')

    def pop(self, connection_id):
        """
        Deletes the connection with the specified name from the group.
        """
        raise NotImplementedError(f'Cannot remove connections from {self.__class__.__name__}')

class Cell:
    ''' 
    Models a biological cell.
    '''
    def __init__(self, name, network, **kwargs):
        """
        Initializes a new instance of the Cell class.

        Args:
            name (str): 
                The name of the neuron.
            network (NeuronalNetwork): 
                The neuronal network to which the neuron belongs.
            type (str, optional): 
                The type of the neuron. Defaults to ''.
            category (str, optional): 
                The category of the neuron. Defaults to ''.
            modality (str, optional): 
                The modality of the neuron. Defaults to ''.
            position (dict, optional): 
                The position of the neuron. Defaults to None.
            presynapses (list, optional): 
                The list of presynaptic components. Defaults to None.
            postsynapses (dict, optional): 
                The dictionary of postsynaptic components. Defaults to None.
        """
        self.name = name
        self.group_id = 0
        self._data = {}
        self.network = network

        # self.type = kwargs.pop('cell_type', '')
        # self.category= kwargs.pop('category', '')
        # self.modality= kwargs.pop('modality','')
        # self.position= kwargs.pop('position', {'AP': 0, 'LR': 0, 'DV': 0})

        # self.surface_area = kwargs.pop('surface_area', 1)
        # self.volume = kwargs.pop('volume', 1)

        for key, value in kwargs.items():
            setattr(self, key, value)
        
        self.in_connections = {}
        self.out_connections = {}
        
        self.network.add_node(self, **kwargs)#type=self.type, category=self.category, modality=self.modality)
    
class Neuron(Cell):
    ''' Models a biological neuron'''
    def __init__(self, name, network, **kwargs): #neuron_type='', category='', modality='',\ position=None, presynapse=None, postsynapse=None,
        """
        Initializes a new instance of the Neuron class.

        Args:
            name (str): 
                The name of the neuron.
            network (NeuronalNetwork): 
                The neuronal network to which the neuron belongs.
            type (str, optional): 
                The type of the neuron. Defaults to ''.
            category (str, optional): 
                The category of the neuron. Defaults to ''.
            modality (str, optional): 
                The modality of the neuron. Defaults to ''.
            position (dict, optional): 
                The position of the neuron. Defaults to None.
            presynapses (list, optional): 
                The list of presynaptic components. Defaults to None.
            postsynapses (dict, optional): 
                The dictionary of postsynaptic components. Defaults to None.
        """
        super().__init__(name, network, **kwargs)#cell_type=neuron_type, category=category, modality=modality,\ position=position)
        # self.name = name
        # self.group_id = 0
        # self._data = {}
        # self.network = network

        # self.type = neuron_type
        # self.category = category
        # self.modality = modality
        # # self.position = position or {'AP': 0, 'LR': 0, 'DV': 0}

        # self.in_connections = {}
        # self.out_connections = {}
        
        # self.network.add_node(self, type=self.type, category=self.category, modality=self.modality)
        self.trial = {}
        self.features = {0: 'Ca_max', 1: 'Ca_area', 2: 'Ca_avg',
                         3: 'Ca_time_to_peak', 4: 'Ca_area_to_peak',
                         5: 'Ca_min', 6: 'Ca_onset', 7: 'positive_area', 8: 'positive_time'}
        
        
        # self.presynapse = presynapse or []
        # self.postsynapse = postsynapse or {}
        #self.cable_length = kwargs.pop('cable_length', 1)
        

    # def set_presynapse(self, presynapse):
    #     """
    #     Set the presynapse of the neuron.

    #     Parameters:
    #         presynapse (list): The presynaptic connections of the neuron.

    #     Returns:
    #         None
    #     """
    #     assert isinstance(presynapse, list), "preSynapse must be a list"
    #     self.presynapse = presynapse

    # def set_postsynapse(self, postsynapse):
    #     """
    #     Set the postsynapse of the neuron.

    #     Parameters:
    #         postsynapse (dict): The postsynaptic connections of the neuron.
    #                            Key: Receptor name, Value: List of ligand names.

    #     Returns:
    #         None
    #     """
    #     # postsynapse should be a dictionary where the key is the receptor name and
    #     # the value is a list of ligand names
    #     assert isinstance(postsynapse, dict), ("postSynapse must be a dictionary, "
    #                                            "where the key is the receptor name "
    #                                            "and the value is a list of ligand names")
    #     self.postsynapse = postsynapse  # {Receptor: ['Ligand_0', 'Ligand_1', ...]}

    def add_trial(self, trial_num=0):
        """
        Adds a new trial to the `trial` dictionary of the current object with the given `trial_num`.
        If `trial_num` is not provided, it defaults to 0.

        Returns:
            Trial: The newly added trial object.
        """
        self.trial[trial_num] = Trial(self, trial_num)
        return self.trial[trial_num]

    def get_connections(self, paired_neuron=None, direction='both'):
        """
        Returns all connections that the neuron is involved in.

        :return: A list of connections where the neuron is present.
        :rtype: list
        """
        if paired_neuron is None:
            if direction == 'both':
                return self.in_connections | self.out_connections
                #return [edge for edge in self.network.edges if self in edge]
            if direction == 'in':
                return self.in_connections
            if direction == 'out':
                return self.out_connections
            raise ValueError('Direction must be either "both", "in", or "out"')

        if paired_neuron is not None:
            if direction == 'both':
                return self.outgoing(paired_neuron) | self.incoming(paired_neuron)
            if direction == 'in':
                return self.incoming(paired_neuron)
            if direction == 'out':
                return self.outgoing(paired_neuron)
            raise ValueError('Direction must be either "both", "in", or "out"')

    def get_connected_neurons(self, direction='both', weight_filter = 1):
        """ Returns all connected neurons for this neuron. """
        if direction == 'both':
            conns = self.in_connections | self.out_connections
        elif direction == 'in':
            conns = self.in_connections
        elif direction == 'out':
            conns =  self.out_connections
        else:
            raise ValueError('Direction must be either "both", "in", or "out"')
        all_conns = []
        for c, conn in conns.items():
            if conn.weight>weight_filter:
                all_conns+= [c[0]]
                all_conns+= [c[1]]
        all_conns = set(all_conns)
        return all_conns

    def update_connections(self):
        """
        Updates the `in_connections` and `out_connections` dictionaries of the current object.
        """
        self.in_connections = {_id: self.network.connections[_id] for _id in self.network.in_edges(self, keys=True)}
        self.out_connections = {_id: self.network.connections[_id] for _id in self.network.out_edges(self, keys=True)}

    def outgoing(self, paired_neuron=None):
        """
        Returns a list of all outgoing connections from the current object.

        :return: A list of connections from the current object to other objects.
        :rtype: list
        """
        if paired_neuron is None:
            return self.out_connections
        if isinstance(paired_neuron, Neuron):
            return {edge:conn for edge,conn in self.out_connections.items() if edge[0] == self and edge[1] == paired_neuron}
        raise TypeError('paired_neuron must be a Neuron object')

    def incoming(self, paired_neuron=None):
        """
        Returns a list of all incoming connections to the current object.
        """
        if paired_neuron is None:
            return self.in_connections
        if isinstance(paired_neuron, Neuron):
            return {edge:conn for edge,conn in self.in_connections.items() if edge[1] == self and edge[0] == paired_neuron}
        raise TypeError('paired_neuron must be a Neuron object')

    def set_property(self, property_name, property_value):
        """
        Sets a new property attribute for the class.

        Args:
            property_name (str): The name of the property.
            property_value: The value of the property.
        """
        setattr(self, property_name, property_value)
        nx.set_node_attributes(self.network, {self: {property_name: property_value}})

    def get_property(self, key):
        ''' Gets an attribute for the class'''
        return getattr(self, key)

    def connects_to(self, other):
        ''' Checks if this neuron connects to another neuron '''
        for o in self.out_connections:
            if o[1] == other:
                return True
        for i in self.in_connections:
            if i[0] == other:
                return True
        return False

    def paths_to(self, target, path_length=1):
        ''' 
        Returns all paths as a list of connections from this neuron to the target neuron
        '''
        path_list = [self.network.groups[group] for group in self.network.groups if group.startswith(f'Path_{self.name}_{target.name}_length_{path_length}')]
        paths = nx.all_simple_edge_paths(self.network, self, target, cutoff=path_length)
        connection_paths = [[self.network.connections[edge] for edge in path] for path in paths]
        if len(path_list) == len(connection_paths):
            return path_list
        else:
            return [Path(self.network, path, f'Path_{self.name}_{target.name}_length_{path_length}_{j}') for j,path in enumerate(connection_paths)] 

    def __str__(self):
        ## For use in debugging and testing
        return self.name

    # def __repr__(self):
    #     ## For use in debugging and testing
    #     return self.name

class Connection:
    ''' This class represents a connection between two cells. '''
    def __init__(self, pre, post, uid=None, edge_type='chemical-synapse', **kwargs):
        """
        Initializes a new instance of the Connection class.

        Args:
            pre (Neuron): 
                The neuron sending the connection.
            post (Neuron): 
                The neuron receiving the connection.
            uid (int, optional): 
                The unique identifier for the connection.
            edge_type (str, optional): 
                The type of the connection.
            weight (float, optional): 
                The weight of the connection.
        """
        self.pre = pre
        self.post = post
        self.network = post.network
        self.weight = kwargs.pop('weight',1)
        # self.color = kwargs.pop('color', 'k')
        if pre.network != post.network:
            raise AssertionError("The Nervous Systems of the pre and post neurons must be the same.")
        
        if not uid:
            self.uid = self.network.add_edge(pre, post, weight=self.weight, edge_type=edge_type, **kwargs)
        else:
            self.uid = uid
            self.network.add_edge(pre, post, key=uid, weight=self.weight, edge_type=edge_type, **kwargs)

        self._id = (pre, post, self.uid)
        self.edge_type = edge_type
        

        self.pre.out_connections[self._id] = self
        self.post.in_connections[self._id] = self

        for key, value in kwargs.items():
            self.set_property(key, value)

    @property
    def by_name(self):
        """
        Returns the connecting neuron names (Pre,Post)
        """
        return (self.pre.name, self.post.name)
    def update_weight(self, weight, delta=False):
        ''' Sets the connection weight '''
        if not delta:
            self.weight = weight
        else:
            self.weight += weight
        nx.set_edge_attributes(self.network, {self._id:{'weight':self.weight}})

    def set_property(self, key, val):
        ''' Sets an attribute for the class'''
        setattr(self, key, val)
        nx.set_edge_attributes(self.network, {self._id:{key:val}})

    def get_property(self, key):
        ''' Gets an attribute for the class'''
        return getattr(self, key)

class ChemicalSynapse(Connection):
    ''' This is a convenience class that represents connections of type chemical synapses.'''
    def __init__(self, pre, post, uid=0, edge_type='chemical-synapse', weight=1, **kwargs):
        super().__init__(pre, post, uid=uid, edge_type=edge_type, weight=weight, **kwargs)
        self.position= kwargs.pop('position', {'AP': 0, 'LR': 0, 'DV': 0})

class GapJunction(Connection):
    ''' This is a convenience class that represents connections of type gap junctions.'''
    def __init__(self, pre, post, uid=1, edge_type='gap-junction', weight=1, **kwargs):
        super().__init__(pre, post, uid=uid, edge_type=edge_type, weight=weight, **kwargs)
        self.position= kwargs.pop('position', {'AP': 0, 'LR': 0, 'DV': 0})

class BulkConnection(Connection):
    ''' This is a convenience class that represents connections of type neuropeptide-receptors.'''
    def __init__(self, pre, post, uid, edge_type, weight=1, **kwargs):
        super().__init__(pre, post, uid=uid, edge_type=edge_type, weight=weight, **kwargs)

class Trial:
    """ This is the trial class for different trials on the same wo Write a utir, neuron, etc"""
    def __init__(self, parent, trialnum):
        """
        Initializes the Trial object with the given parent and trial number.

        Parameters:
            parent (datatype): 
                Description of the parameter.
            trialNum (datatype): 
                Description of the parameter.

        Returns:
            None
        """
        self.parent = parent
        self.i = trialnum

    @property
    def recording(self):
        """
        Get the recording data for the Trial object.

        Returns:
            datatype: The recording data.
        """
        return self._data

    @recording.setter
    def recording(self, _data, discard=0):
        """
        Set the recording data for the Trial object.

        Parameters:
            signal (array-like): The timecourse signal to be recorded.

        Raises:
            ValueError: If the length of the signal is not 451 or 601.

        Returns:
            None
        """
        if not discard:
            self.discard = []
            self._data = _data.astype(np.float64)
        elif discard>0:
            self.discard = discard*F_SAMPLE #Initial points to be discarded due to bleaching, etc.
            self._data = _data[discard*F_SAMPLE:].astype(np.float64)
        else:
            raise ValueError("Discard cannot be negative")

class StimResponse:
    """
    This is the stimulus and response class, for different trials on the same worm, neuron, etc
    """
    def __init__(self, trial, stimulus, response, baseline) -> None:
        """
        Initializes a StimResponse object.

        Parameters:
            trial (Trial): 
                The trial object associated with the stimulus and response.
            stimulus (array-like): 
                The stimulus signal.
            response (array-like): 
                The response signal.
            baseline (int): 
                The number of baseline samples to consider for the response.

        Returns:
            None
        """
        self.stim = stimulus
        self.response = response
        self.feature = {}
        self.neuron = trial.parent
        self.f_sample = F_SAMPLE # frames per sec
        self.sampling_time = 1./self.f_sample
        self.baseline = self.response[:baseline]

        for feature_index in range(len(self.neuron.features)):
            self.feature.update({feature_index: self.extract_feature(feature_index)})

    def extract_feature(self, feature_index):
        """
        Extracts a specific feature from the stimulus response pair.

        Parameters:
            feature_index (int): The index of the feature to extract. Possible values are:
                0: 
                    Maximum value of the response.
                1: 
                    Area under the curve of the response.
                2: 
                    Mean value of the response.
                3: 
                    Time to peak of the response.
                4: 
                    Area under the curve to peak of the response.
                5: 
                    Minimum value of the response.
                6: 
                    Onset time of the response.
                7: 
                    Positive area of the response.
                8: 
                    Absolute area under the curve of the response.

        Returns:
            The extracted feature value. The return type depends on the feature index.
        """

        feature_mapping = {
            0: self._find_maximum,
            1: self._area_under_the_curve,
            2: self._find_mean,
            3: self._find_time_to_peak,
            4: self._area_under_the_curve_to_peak,
            5: self._find_minimum,
            6: self._find_onset_time,
            7: lambda: (self._find_positive_area()[0], self._find_positive_area()[1]),
            8: self._absolute_area_under_the_curve,
        }

        return feature_mapping[feature_index]()

    # Features
    def _find_maximum(self):
        '''Finds the maximum of the vector in a given interest'''
        return np.max(self.response)

    def _find_minimum(self):
        '''Finds the maximum of the vector in a given interest'''
        return np.min(self.response)

    def _find_time_to_peak(self):
        '''Finds the time to maximum of the vector in a given interest'''
        max_index = np.argmax(self.response)
        time_to_peak = (max_index)*self.sampling_time
        return time_to_peak

    def _find_mean(self):
        '''Finds the mean of the vector in a given interest'''
        return np.average(self.response)

    def _area_under_the_curve(self, bin_size=5):
        '''Finds the area under the curve of the vector in the given window.
           This will subtract negative area from the total area.'''
        undersampling = self.response[::bin_size]
        auc = np.trapz(undersampling, dx=self.sampling_time*bin_size)  # in V.s
        return auc

    def _absolute_area_under_the_curve(self, bin_size=5):
        '''Finds the area under the curve of the vector in the given window.
           This will subtract negative area from the total area.'''
        undersampling = np.abs(self.response[::bin_size])
        auc = np.trapz(undersampling, dx=self.sampling_time*bin_size)  # in V.s
        return auc

    def _area_under_the_curve_to_peak(self, bin_size=10):
        '''Finds the area under the curve of the vector in the given window'''
        undersampling = self.response[::bin_size]
        max_index = np.argmax(undersampling)
        window_to_peak = undersampling[:max_index+1]
        auctp = np.trapz(window_to_peak, dx=self.sampling_time*bin_size)  # in V.s
        return auctp

    def _find_onset_time(self, step=2, slide = 1, init_pval_tolerance=0.5):
        ''' Find the onset of the curve using a 2 sample KS test
	maxOnset, step and slide are in ms'''

        window_size = int(step*self.f_sample)
        step_size = int(slide*self.f_sample)
        index_right = window_size
        for index_left in range(0, len(self.response)+1-window_size, step_size):
            _stat, pval = ss.ks_2samp(self.baseline, self.response[index_left:index_right])
            index_right += step_size
            if pval < init_pval_tolerance:
                # print index_left, pVal, stat#, self.self.response_raw[index_left:index_right]
                break
        return float(index_left)/self.f_sample

    def _find_positive_area(self, bin_size=10):
        '''Finds the area under the curve of the vector in the given window'''
        undersampling = self.response[::bin_size]
        undersampling = undersampling[np.where(undersampling>0.)]
        auc_pos = np.trapz(undersampling,  dx=self.sampling_time*bin_size)  # in V.s
        positive_time = self.sampling_time*len(undersampling)
        return auc_pos, positive_time

    #def _flagNoise(self, pValTolerance=0.01):
    #    ''' This function asseses if the distributions of the baseline
    #    and interest are different or not '''
    #    m, pVal = ss.ks_2samp(self.baselineWindow, self.self.response)
    #    if pVal < pValTolerance:
    #        return 0
    #    else:
    #        print "No response measured in trial {}".format(self.index)
    #        return 1  # Flagged as noisy

    def _normalize_to_baseline(self, baseline_window):
        '''normalizes the vector to an average baseline'''
        baseline = np.average(baseline_window)
        response_new = self.response - baseline  # Subtracting baseline from whole array
        return response_new, baseline

    def _time_series_filter(self, ts_filter='', cutoff=2000., order=4, trace=None):
        ''' Filter the time series vector '''
        if trace is None:
            trace = self.response
        if ts_filter == 'bessel': # f_sample/2 is Niquist, cutoff is the low-pass cutoff.
            cutoff_to_niquist_ratio = 2*cutoff/(self.f_sample)
            _b, _a = signal.bessel(order, cutoff_to_niquist_ratio, analog=False)
            trace =  signal.filtfilt(_b, _a, trace)
        return trace

    def _smoothen(self, smoothening_time):
        '''normalizes the vector to an average baseline'''
        smoothening_window = smoothening_time*self.f_sample
        window = np.ones(int(smoothening_window)) / float(smoothening_window)
        self.response = np.convolve(self.response, window, 'same')  # Convolving with a rectangle
        return self.response

class Map:
    ''' This class represents a map between two entities. It can be used to map between 
    sets of nodes, edges or graphs. For example, it can be used to map neurotransmitters or 
    neuropeptides to their corresponding receptors, protein-protein interactions, or 
    abstract motifs to graphs etc.
    '''
    def __init__(self, mapping_dict, **kwargs):
        """
        Initialize a directed graph representing a map.

        Args:
            mapping_dict (dict): A dictionary where the keys are the nodes in the first graph and the values are the nodes in the second graph.
            kwargs (dict, optional): Additional keyword arguments.
        """
        self.mapping_dict = mapping_dict
        self.mapping = None

class NodeMap(Map):
    """
    This class represents a map between two nodes.
    """
    def __init__(self, mapping_dict, **kwargs):
        """
        Initialize a directed graph representing a map between two sets of nodes.

        Args:
            mapping_dict (dict): A dictionary where the keys are labels or nodes, and values are nodes in a graph.
            kwargs (dict, optional): Additional keyword arguments.
        """
        super().__init__(mapping_dict, **kwargs)
        self.mapping = (len(self.mapping_dict.keys()), sum(len(val) for val in self.mapping_dict.values()))

class EdgeMap(Map):
    """
    This class represents a map between two sets of edges."""
    def __init__(self, mapping_dict, **kwargs):
        """
        Initialize a directed graph representing a map between two sets of edges.

        Args:
            mapping_dict (dict): A dictionary where the keys are the edges in the first graph and the values are the edges in the second graph.
            kwargs (dict, optional): Additional keyword arguments.
        """
        super().__init__(mapping_dict, **kwargs)
        self.mapping = (len(self.mapping_dict.keys()), sum(len(val) for val in self.mapping_dict.values()))

class GraphMap(Map):
    """ 
    This class represents a map between two graphs."""
    def __init__(self, mapping_dict, graph_1, graph_2, map_type = 'node', **kwargs):
        """
        Initialize a directed graph representing a map between two graphs.

        Args:
            mapping_dict (dict): A dictionary where the keys are the nodes or edges in the first graph and the values are the nodes or edges in the second graph.
            graph_1 (networkx.DiGraph): The graph representing the first graph.
            graph_2 (networkx.DiGraph): The graph representing the second graph.
            map_type (str, optional): The type of mapping, either 'node' or 'edge'. Defaults to 'node'.
            kwargs (dict, optional): Additional keyword arguments.

        Raises:
            ValueError: If any of the nodes in the mapping_dict are not found in either graph_1 or graph_2.
        """
        super().__init__(mapping_dict, **kwargs)
        self.graph_1 = graph_1
        self.graph_2 = graph_2
        for item_1, item_2 in self.mapping_dict.items():
            if  map_type == 'node':
                if (isinstance(item_1, str) or isinstance(item_1, Neuron)) and (isinstance(item_2, str) or isinstance(item_2, Neuron)):
                    if item_1 not in self.graph_1.nodes():
                        raise ValueError(f"Node {item_1} not found in graph_1")
                    if item_2 not in self.graph_2.nodes():
                        raise ValueError(f"Node {item_2} not found in graph_2")
                else:
                    raise ValueError("Nodes must be of type str or Neuron")
            elif map_type == 'edge':
                if (isinstance(item_1, tuple) or isinstance (item_1, Connection)) and (isinstance(item_2, tuple) or isinstance (item_2, Connection)):
                    if item_1 not in self.graph_1.edges():
                        raise ValueError(f"Edge {item_1} not found in graph_1")
                    if item_2 not in self.graph_2.edges():
                        raise ValueError(f"Edge {item_2} not found in graph_2")
                else:
                    raise ValueError("Edges must be of type tuple or Connection")
        self.mapping = (len(self.mapping_dict.keys()), len(self.mapping_dict.values()))
        if map_type == 'node':
            for node in mapping_dict.keys():
                graph_1.nodes[node]['map'] = mapping_dict[node]
        if map_type == 'edge':
            for edge in mapping_dict.keys():
                graph_1.edges[edge]['map'] = mapping_dict[edge]
                for j,node in enumerate(edge):
                    graph_1.nodes[node]['map'] = mapping_dict[edge][j]

## Functions
def generate_random_string(length: int = 8) -> str:
    """
    Generates a random string of given length.

    Args:
        length (int): The length of the string to generate.

    Returns:
        str: A random string of the specified length.
    """
    characters = string.ascii_letters + string.digits
    return ''.join(random.choice(characters) for _ in range(length))

def join_networks(networks, copy_graph=False):
    """
    A function to join multiple networks into a single composed network.
    
    Parameters:
        networks (list): 
            List of networks to be joined.
        copy_graph (bool, optional): 
            Whether to create a copy of the graph. Defaults to False.
    
    Returns:
        network_composed: The composed network after joining all input networks.
    """
    network_composed = nx.compose_all(networks)
    fold_dict = {}
    for node in network_composed.nodes:
        if not node.name in fold_dict:
            fold_dict[node.name] = [node]
        else:
            fold_dict[node.name].append(node)

    for nodename, nodes_to_fold in fold_dict.items():
        node1 = fold_dict[nodename][0]
        for node2 in nodes_to_fold[1:]:
            network_composed = nx.contracted_nodes(network_composed, node1, node2,\
                                 copy=copy_graph)

    network_composed.update_neurons()
    network_composed.update_connections()
    return network_composed

# Transformations
def _linear_transform(value, minvalue, maxvalue):
    return (value - minvalue)/(maxvalue - minvalue)