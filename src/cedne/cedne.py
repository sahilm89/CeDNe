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
import copy
import numpy as np
import networkx as nx
#from warnings import warn
import scipy.stats as ss
from scipy import signal

# Superparameters
RANDOM_SEED = 42
F_SAMPLE = 5 # Hz
class Worm:
    ''' This is a full organism class'''
    def __init__(self, name='', stage='Day-1 Adult', sex='Hermaphrodite', genotype='N2') -> None:
        """
        Initializes a Worm object.

        Parameters:
            name (str): The name of the worm. If empty,
                a random alphanumeric string will be generated.
            stage (str): The stage of the worm. Default is 'Day-1 Adult'.
              Other options can be L1, L2, L3, L4, Day-2 Adult, etc.
            sex (str): The sex of the worm. Default is 'Hermaphrodite'.
                Other options can be 'Male'.
            genotype (str): The genotype of the worm. Default is 'N2'.
                Other options can be mutant names.

        Returns:
            None
        """
        if not name:
            name = 'Worm-' + generate_random_string()
        self.name = name
        self.stage = stage
        self.sex = sex
        self.genotype = genotype
        self.networks = {}

    def save(self, file_path, file_format='pickle'):
        """
        Saves the Worm object to a pickle file at the specified file path.

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
            return pickle.load(pickle_file)
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

    def build_nervous_system(self, neuron_data, chem_synapses, elec_synapses, positions):
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

            self.create_neurons(labels, neuron_types, categories, modalities, locations)
            self.setup_chemical_connections(chem_adjacency)
            self.setup_gap_junctions(elec_adjacency)
    def build_network(self, neurons, adj, label):
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
        with open(neurons, 'rb') as neuron_file:
            node_dict = pickle.load(neuron_file)
            node_labels, l1_list, l2_list, l3_list = node_dict.iloc[:,0].to_list(),\
                  node_dict.iloc[:,1].to_list(), \
                    node_dict.iloc[:,2].to_list(), \
                        node_dict.iloc[:,3].to_list()
            self.create_neurons(node_labels, l1_list, l2_list, l3_list)
        self.setup_connections(adj, label)

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
        print({connection_id: self.connections[connection_id] for connection_id in self.connections})
        for connection_id in self.connections:
            if not connection_id in self.edges:
                self.connections.pop(connection_id)

    def setup_connections(self, adjacency_matrix, edge_type):
        """
        Set up connections between neurons based on the adjacency matrix and edge type.
        """
        for source_id, neighbors in adjacency_matrix.items():
            for target_id, properties in neighbors.items():
                if properties['weight'] > 0:
                    source_neuron = self.neurons[source_id]
                    target_neuron = self.neurons[target_id]
                    edge_weight = properties['weight']
                    edge_id = self.add_edge(
                        source_neuron, target_neuron,
                        weight=edge_weight, color='k', edge_type=edge_type
                    )
                    connection = Connection(
                        source_neuron, target_neuron, edge_id, edge_type, weight=edge_weight
                    )
                    self.connections[(source_neuron, target_neuron, edge_id)] = connection

    def create_neurons(self, labels, \
                       neuron_types=None, categories=None, modalities=None, positions=None):
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
        neuron_types = [None] * len(labels) if neuron_types is None else neuron_types
        categories = [None] * len(labels) if categories is None else categories
        modalities = [None] * len(labels) if modalities is None else modalities
        positions = {} if positions is None else positions

        for label, neuron_type, category, modality in \
            zip(labels, neuron_types, categories, modalities):
            if label in positions:
                neuron = Neuron(label, self, neuron_type, category, modality, positions[label])
            else:
                neuron = Neuron(label, self, neuron_type, category, modality)
            self.neurons[label] = neuron

    def setup_chemical_connections(self, chemical_adjacency):
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

        Note:
            - The `add_edge` method is assumed to be defined in the class.
            - The `Connection` class is assumed to be defined in the class.
            - The `neurons` dictionary is assumed to be defined in the class.
        """
        edge_type='chemical-synapse'
        for source_neuron, target_neurons in chemical_adjacency.items():
            for target_neuron, connection_data in target_neurons.items():
                if connection_data['weight'] > 0:
                    edge_key = self.add_edge(
                        self.neurons[source_neuron],
                        self.neurons[target_neuron],
                        weight=connection_data['weight'],
                        color='orange',
                        edgeType=edge_type
                    )
                    connection = Connection(
                        self.neurons[source_neuron],
                        self.neurons[target_neuron],
                        edge_key,
                        edge_type,
                        weight=connection_data['weight']
                    )
                    self.connections[(self.neurons[source_neuron], \
                                      self.neurons[target_neuron], edge_key)] = connection
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
                    edge_key = self.add_edge(
                        self.neurons[source_neuron],
                        self.neurons[target_neuron],
                        weight=connection_data['weight'],
                        color='gray',
                        edge_type=edge_type
                    )
                    connection = Connection(
                        self.neurons[source_neuron],
                        self.neurons[target_neuron],
                        edge_key,
                        edge_type,
                        weight=connection_data['weight']
                    )
                    self.connections[(
                        self.neurons[source_neuron],
                        self.neurons[target_neuron],
                        edge_key
                    )] = connection
    def load_neuron_data(self, file, file_format='summary-xlsx'):
        ''' Standard formats to load data into the network'''
        #pass

    def load_connection_data(self, file, file_format='summary-xlsx'):
        ''' Standard formats to load data into the network'''
        #pass

    def subnetwork(self, neuron_names=None, connections=None, as_view=False):
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

        assert not (neuron_names and connections),\
            "Specify either neuron_names or connections, not both."
        if not as_view:
            graph_copy = self.copy(copy_type='deep')
        else:
            graph_copy = self.copy(copy_type='view')

        if neuron_names is not None:
            subgraph_nodes = [graph_copy.neurons[name] for name in neuron_names]
            subgraph = graph_copy.subgraph(subgraph_nodes)
            subgraph.connections = {key: value for key, value in graph_copy.connections.items() if key[0] in subgraph_nodes and key[1] in subgraph_nodes}
        elif connections is not None:
            subgraph = graph_copy.edge_subgraph(connections)
        else:
            subgraph = self
        subgraph.update_neurons()
        subgraph.update_connections()
        return subgraph #subgraph.copy(as_view)

    def fold_network(self, fold_by, exceptions=None):
        """
        Fold the network based on a filter.

        Args:
            fold_by (tuple): A tuple of length 2 specifying the neurons to fold.
                The first element is the neurons to fold, and the second element
                is the neurons that are exempt from folding.
                The tuple can contain any neuron name as a string.

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
        if exceptions is None:
            exceptions = []
        for merged_nodename, nodes_to_fold in fold_by.items():
            if len(nodes_to_fold) >1:
                merged_node = nodes_to_fold[0]
                for j in range(1,len(nodes_to_fold)):
                    npair = (merged_node, nodes_to_fold[j])
                    if not npair[0] in exceptions and not npair[1] in exceptions:
                        self.contract_neurons(npair, merged_nodename)
                        merged_node = merged_nodename
            else:
                self.neurons[nodes_to_fold[0]].name = merged_nodename
                self.update_neurons()

    def contract_neurons(self, pair, contracted_name, copy_graph=False):
        """
        Contract two neurons together.

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
        else:
            nx.contracted_nodes(self, self.neurons[source_neuron], self.neurons[target_neuron],\
                                 copy=copy_graph)
            self.neurons[source_neuron].name = contracted_name
            self.update_neurons()
            return self


    def neurons_have(self, key):
        ''' Returns neuron attributes'''
        return nx.get_node_attributes(self, key)

    def connections_have(self, key):
        ''' Gets connection attributes'''
        return nx.get_edge_attributes(self, key)

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
        if len(neurons_have):
            for (key, value) in neurons_have.items():
                each_filter = []
                for node, val in self.neurons_have(key).items():
                    if val==value:
                        each_filter.append(node)
                total_node_list.append(each_filter)
            if condition=='AND':
                filtered_node_list = [node for _n,node in self.neurons.items()\
                                       if all(node in sublist for sublist in total_node_list)]

            elif condition=='OR':
                filtered_node_list = [node for _n,node in self.neurons.items()\
                                       if any(node in sublist for sublist in total_node_list)]
            else:
                raise ValueError("condition must be 'AND' or 'OR'")
        else:
            filtered_node_list = self.neurons.values()

        self._filtered_nodes = set(filtered_node_list)

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
                filtered_edge_list = [edge for _e,edge in self.connections.items()\
                                       if all(_e in sublist for sublist in total_edge_list)]

            elif condition=='OR':
                filtered_edge_list = [edge for _e,edge in self.connections.items()\
                                       if any(_e in sublist for sublist in total_edge_list)]
            else:
                raise ValueError("condition must be 'AND' or 'OR'")
        else:
            filtered_edge_list = self.connections.keys()

        self._filtered_edges = set(filtered_edge_list)

        return nx.subgraph_view(self, filter_node=self.__filter_node__,\
                                 filter_edge=self.__filter_edge__)

    def copy(self, copy_type='deep'):
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
        if copy_type=='view':
            return super().copy(as_view=True)
            #nx.graphviews.generic_graph_view(self, create_using=NervousSystem)
        elif copy_type=='shallow':
            return super().copy(as_view=False)
        elif copy_type=='deep':
            return copy.deepcopy(self)
        else:
            raise ValueError("copy_type must be 'deep', 'shallow', or 'view'")

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
        return self.neurons.keys()
    def values(self):
        """
        Returns an iterator over the members of the group.
        """
        return self.neurons.values()

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
        self.members = self.neurons.values()

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
            assert all([isinstance(m, Connection)for m in members]),\
                  "Connection group members must be of type Connection"
        self.members = members
        self.connections = {m._id:m for m in members}
        self.network = network
        assert self.group_name not in self.network.groups, \
        f"Group name {self.group_name} already exists in the network"
        self.network.groups.update({self.group_name: self})

    def __iter__(self):
        """
        Returns an iterator over the members of the group.
        """
        return iter(self.connections)

    def items(self):
        """
        Returns an iterator over the members of the group.
        """
        for key, value in self.connections.items():
            yield key, value

    def clear(self):
        """
        Removes all connections from the group.
        """
        self.connections = {}
        self.members = []
    def keys(self):
        """
        Returns an iterator over the members of the group.
        """
        return self.connections.keys()

    def values(self):
        """
        Returns an iterator over the members of the group.
        """
        return self.connections.values()

    def __len__(self):
        """
        Returns the number of members in the group.
        """
        return len(self.connections)

    def __contains__(self, connection):
        """
        Returns True if the connection with the specified name is in the group, False otherwise.
        """
        return connection in self.connections

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
        self.members = self.connections.values()

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

class Neuron:
    ''' Models a biological neuron'''
    def __init__(self, name, network, neuron_type='', category='', modality='',\
         position=None, presynapse=None, postsynapse=None):
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
        self.name = name
        self.group_id = 0
        self._data = {}
        self.trial = {}
        self.features = {0: 'Ca_max', 1: 'Ca_area', 2: 'Ca_avg',
                         3: 'Ca_time_to_peak', 4: 'Ca_area_to_peak',
                         5: 'Ca_min', 6: 'Ca_onset', 7: 'positive_area', 8: 'positive_time'}
        self.type = neuron_type
        self.category = category
        self.modality = modality
        self.position = position or {'AP': 0, 'LR': 0, 'DV': 0}
        self.presynapse = presynapse or []
        self.postsynapse = postsynapse or {}
        self.in_connections = {}
        self.out_connections = {}
        self.network = network
        self.network.add_node(self, type=self.type, category=self.category, modality=self.modality)

    def set_presynapse(self, presynapse):
        """
        Set the presynapse of the neuron.

        Parameters:
            presynapse (list): The presynaptic connections of the neuron.

        Returns:
            None
        """
        assert isinstance(presynapse, list), "preSynapse must be a list"
        self.presynapse = presynapse

    def set_postsynapse(self, postsynapse):
        """
        Set the postsynapse of the neuron.

        Parameters:
            postsynapse (dict): The postsynaptic connections of the neuron.
                               Key: Receptor name, Value: List of ligand names.

        Returns:
            None
        """
        # postsynapse should be a dictionary where the key is the receptor name and
        # the value is a list of ligand names
        assert isinstance(postsynapse, dict), ("postSynapse must be a dictionary, "
                                               "where the key is the receptor name "
                                               "and the value is a list of ligand names")
        self.postsynapse = postsynapse  # {Receptor: ['Ligand_0', 'Ligand_1', ...]}

    def add_trial(self, trial_num=0):
        """
        Adds a new trial to the `trial` dictionary of the current object with the given `trial_num`.
        If `trial_num` is not provided, it defaults to 0.

        Returns:
            Trial: The newly added trial object.
        """
        self.trial[trial_num] = Trial(self, trial_num)
        return self.trial[trial_num]

    def get_connections(self):
        """
        Returns all connections that the neuron is involved in.

        :return: A list of connections where the neuron is present.
        :rtype: list
        """
        return [edge for edge in self.network.edges if self in edge]

    def outgoing(self):
        """
        Returns a list of all outgoing connections from the current object.

        :return: A list of connections from the current object to other objects.
        :rtype: list
        """
        return self.out_connections

    def incoming(self):
        """
        Returns a list of all incoming connections to the current object.
        """
        return self.in_connections

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
        return other in self.connections

class Connection:
    ''' This class represents a connection between two neurons. '''
    def __init__(self, pre_neuron, post_neuron, uid=0, edge_type='chemical-synapse', weight=1):
        """
        Initializes a new instance of the Connection class.

        Args:
            pre_neuron (Neuron): 
                The neuron sending the connection.
            post_neuron (Neuron): 
                The neuron receiving the connection.
            uid (int, optional): 
                The unique identifier for the connection.
            edge_type (str, optional): 
                The type of the connection.
            weight (float, optional): 
                The weight of the connection.

        Raises:
            AssertionError: If the neural networks of the pre and post neurons are not the same.
        """
        self.pre_neuron = pre_neuron
        self.post_neuron = post_neuron
        self.network = post_neuron.network
        self.uid = uid
        self._id = (pre_neuron, post_neuron, self.uid)
        self.edge_type = edge_type
        self.weight = weight

        self.pre_neuron.out_connections[self._id] = self
        self.post_neuron.in_connections[self._id] = self

    @property
    def by_name(self):
        """
        Returns the connecting neuron names (Pre,Post)
        """
        return (self.pre_neuron.name, self.post_neuron.name)
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
        nx.set_edge_attributes(self.network.graph, {self._id:{key:val}})

    def get_property(self, key):
        ''' Gets an attribute for the class'''
        return getattr(self, key)

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

class Mapping(nx.DiGraph):
    ''' This class represents a directed graph that can be mapped to any
    property of worm/nervous system or neurons that have graph-based properties.
    This can be used to map neurotransmitters or neuropeptides to their corresponding
    receptors, protein-protein interactions, etc.
    '''
    def __init__(self):
        """
        Initialize a directed graph representing a map.
        """
        super().__init__()

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
