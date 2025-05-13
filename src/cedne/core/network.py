"""
Graph-based nervous system representation for CeDNe.

This module defines the `NervousSystem` class, which models a complete 
neural network using a subclass of `networkx.MultiDiGraph`. It serves as 
the central container for neurons (`Neuron`), connections (`Connection`), 
and associated metadata, and provides high-level methods for construction, 
analysis, and manipulation of neural circuits.

Main components:

- `NervousSystem`: Inherits from `networkx.MultiDiGraph` and integrates neuron
  and connection management with experimental and structural logic.

Key functionality includes:
- Creating neurons and connections from raw data or other networks
- Managing and updating network state (including filters, subgraphs, folding)
- Supporting motif search, groupings, and topological export
- Generating subgraphs based on attribute filters or structural criteria
- Contracting neurons and connections to simplify network topology
- Interfacing with experimental metadata (`Worm`, `Trial`, etc.)

This module is central to most workflows in CeDNe, serving as the graph-theoretic 
and biological representation of the nervous system.
"""

__author__ = "Sahil Moza"
__date__ = "2025-04-06"
__license__ = "MIT"


import copy
import pickle
import json
import numpy as np
import networkx as nx
from .connection import Connection, \
    ChemicalSynapse, GapJunction, ConnectionGroup
from .neuron import Neuron, NeuronGroup
from .animal import Worm

class NervousSystem(nx.MultiDiGraph):
    '''
    This is the Nervous System class. This inherits from networkx.MultiDiGraph
      and is the main high level class for the nervous system. '''
    def __init__(self, worm: Worm = None, network: str = "Neutral", **kwargs) -> None:
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

        for key, value in kwargs.items():
            self.set_property(key, value)

    @property
    def num_groups(self):
        """
        Returns the current number of Neuron Groups for the Nervous System.
        """
        return len(self.groups)
    
    def set_property(self, key, value):
        """
        Set a property of the nervous system.

        Args:
            key (str): The name of the property.
            value: The value of the property.

        Returns:
            None
        """
        setattr(self, key, value)

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
            Neuron(label, self, **neuron_args)

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
                Neuron(node.name, self)
        else:
            for node,data in network.nodes(data=True):
                Neuron(node.name, self, **data)

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
        Synchronizes the neurons dictionary with the network's nodes.
        This should only be needed if the network's nodes are modified directly.
        """
        self.neurons.clear()
        for node in self.nodes:
            if node.name not in self.neurons:
                self.neurons[node.name] = node

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
    
    def update_network(self):
        """
        Update the network by setting the network attribute of all connections to self.
        """
        for node in self.nodes:
            node.network = self
        for _, c in self.connections.items():
            c.network = self


    def setup_connections(self, adjacency, connection_type, input_type = 'adjacency', **kwargs):
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
                    #         weight=edge_weight, color='k', connection_type=connection_type
                    #     )
                    connection = Connection(
                        #source_neuron, target_neuron, edge_id, connection_type, weight=edge_weight
                        source_neuron, target_neuron, connection_type=connection_type, weight=edge_weight
                    )
                    self.connections[(source_neuron, target_neuron, connection.uid)] = connection
        

        elif input_type == 'edge':
            source_neuron = self.neurons[adjacency['pre']]
            target_neuron = self.neurons[adjacency['post']]
            edge_weight = adjacency['weight']

            # edge_id = self.add_edge(
            #                 source_neuron, target_neuron,
            #                 weight=edge_weight, color='k', connection_type=connection_type
            # )

            # connection = Connection(source_neuron, target_neuron, edge_id, connection_type,\
            #                          weight=edge_weight, **kwargs)
            connection = Connection(source_neuron, target_neuron, connection_type=connection_type,\
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
        connection_type='chemical-synapse'
        for source_neuron, target_neurons in chemical_adjacency.items():
            for target_neuron, connection_data in target_neurons.items():
                if connection_data['weight'] > 0:
                    # edge_key = self.add_edge(
                    #     self.neurons[source_neuron],
                    #     self.neurons[target_neuron],
                    #     weight=connection_data['weight'],
                    #     color='orange',
                    #     edgeType=connection_type
                    # )
                    connection = ChemicalSynapse(
                        self.neurons[source_neuron],
                        self.neurons[target_neuron],
                        #edge_key,
                        connection_type=connection_type,
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
        connection_type = 'gap-junction'
        for source_neuron, target_neurons in gap_junction_adjacency.items():
            for target_neuron, connection_data in target_neurons.items():
                if connection_data['weight'] > 0:
                    # edge_key = self.add_edge(
                    #     self.neurons[source_neuron],
                    #     self.neurons[target_neuron],
                    #     weight=connection_data['weight'],
                    #     color='gray',
                    #     connection_type=connection_type
                    # )
                    connection = GapJunction(
                        self.neurons[source_neuron],
                        self.neurons[target_neuron],
                        #edge_key,
                        connection_type=connection_type,
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
            
            subgraph.update_network()
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
                source_neuron, target_neuron, connection_type = edge
                weights = [connections.connections[(connections.network.neurons[edge[0]], connections.network.neurons[edge[1]], edge[2])].weight for _netname, connections in all_connections.items()]
                
                #edge_weight = np.mean([all_connections[netname].connections[ (all_connections[netname].neurons[source_neuron], all_connections[netname].neurons[target_neuron], connection_type) ].weight for netname in all_connections.keys()])
                edge_weight = np.mean(weights)
                # edge_id = combined_network.add_edge(
                #                 combined_network.neurons[source_neuron], combined_network.neurons[target_neuron],
                #                 weight=edge_weight, color='k', connection_type=connection_type
                            # )
                # connection = Connection(
                #             combined_network.neurons[source_neuron], combined_network.neurons[target_neuron], edge_id, connection_type, weight=edge_weight
                #         )
                connection = Connection(
                            combined_network.neurons[source_neuron], combined_network.neurons[target_neuron], connection_type=connection_type, weight=edge_weight)
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
        
        graph_copy.update_network()
        graph_copy.update_neurons()
        graph_copy.reassign_connections()
        

        if data == 'collect':
            return graph_copy
        if data == 'clean':
            parsed_conns = {}
            for e,conn in graph_copy.connections.items():
                if (e[0],e[1], conn.connection_type) not in parsed_conns:
                    parsed_conns[(e[0],e[1], conn.connection_type)] = []
                parsed_conns[(e[0],e[1], conn.connection_type)].append(conn)
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
                
            #uid = empty_graph_copy.add_edge(contraction[0], contraction[1], weight=weight, connection_type=contraction[2])
            #new_conn = Connection(contraction[0], contraction[1], uid=uid, connection_type=contraction[2], weight=weight)
            n1 = empty_graph_copy.neurons[contraction[0].name]
            n2 = empty_graph_copy.neurons[contraction[1].name]
            new_conn = Connection(n1, n2, connection_type=contraction[2], weight=weight)
            new_conn.set_property('contraction_data', copy.copy(contraction_data))
            #_connections[(contraction[0], contraction[1], new_conn.uid)] = new_conn
            _connections[(n1,n2, new_conn.uid)] = new_conn
        empty_graph_copy.connections = _connections

        empty_graph_copy.update_network()
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