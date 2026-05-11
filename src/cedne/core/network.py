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
from collections.abc import Sequence
from .connection import Connection, \
    ChemicalSynapse, GapJunction, ConnectionGroup
from .history import record
from .neuron import Neuron, NeuronGroup, MERGED_TYPE, MERGE_TRACK_ATTRS
from .animal import Worm
from .source import Citable

class NervousSystem(nx.MultiDiGraph, Citable):
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
        nx.MultiDiGraph.__init__(self)
        Citable.__init__(self)  # provides self.citations = {}

        self.worm = worm or Worm()
        self.name = network
        self.worm.networks[network] = self
        self.groups = {}

        self.neurons = NeuronGroup(self, group_name='all_neurons')  \
            # dictionary of all neurons in the nervous system
        self.connections = ConnectionGroup(self, group_name='all_connections') \
              # dictionary of all connections in the nervous system
        
        self.visualization_metadata = {}

        self._filtered_nodes = set()
        self._filtered_edges = set()

        for key, value in kwargs.items():
            self.set_property(key, value)

    def _parent_citables(self):
        """Walk citation resolution up to the owning Animal/Worm."""
        return (self.worm,) if self.worm is not None else ()

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
        Make a network with the neurons.
        
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

    def remove_neurons(self, neurons):
        """ Remove neurons from the network."""
        for neuron in neurons:
            if not neuron in self.neurons:
                raise TypeError(f" {neuron} is not a valid neuron name.")
            else:
                self.remove_node(neuron)
        self.update_neurons()

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

    def create_connections(self, connection_dict):
        ''' Creates a set of connections from a dictinary of connections with pre-post pairs as keys and
        data as values.'''
        for (pre,post), data in connection_dict.items():
            if not pre in self.neurons or not post in self.neurons:
                raise TypeError("Input dictionary must use neuron names for connection IDs")
            n1 = self.neurons[pre]
            n2 = self.neurons[post]
            if len(data):
                conn = Connection(n1,n2, **data)
            else:
                conn = Connection(n1,n2)
            self.connections.update({(n1,n2,conn.uid): conn})

    def remove_connections(self, connections):
        """ Remove connections from the network """
        for connection in connections:
            if isinstance(connection, Connection):
                self.remove_edge(connection.pre, connection.post)
            elif isinstance(connection, tuple):
                if isinstance(connection[0], Neuron) and isinstance(connection[1], Neuron):
                    n1 = connection[0]
                    n2 = connection[1]
                    self.remove_edge(n1, n2)
                elif isinstance(connection[0], str) and isinstance(connection[1], str): 
                    if connection[0] in self.neurons and connection[1] in self.neurons:
                        n1 = self.neurons[connection[0]]
                        n2 = self.neurons[connection[1]]
                        self.remove_edge(n1, n2)
                    else:
                        raise NameError(f"{connection[0]} and {connection[1]} not in the network")
                else:
                    raise TypeError("Connections must be either a list of tuples of neurons or neuron names, or a list of Connections.")
            else:
                raise TypeError("Connections must be either a list of tuples of neurons or neuron names, or a list of Connections.")
        
        self.update_connections()
    
    def remove_all_connections(self):
        """ Remove all connections from the network """
        self.remove_connections(self.connections)

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
        
        for u,v,k,edge_data in network.edges(keys=True, data=True):
            n1 = self.neurons[u.name]
            n2 = self.neurons[v.name]
            if not data:
                self.connections.update({(n1,n2,k):Connection(n1, n2, k)})
            else:
                self.connections.update({(n1,n2,k):Connection(n1, n2, k, **edge_data)})

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

    @record("subnetwork")
    def subnetwork(self, neuron_names=None, name=None, connections=None, as_view=False, data=True):
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
            if data==True:
                graph_copy = self.copy(copy_type='deep_with_data', name=name)
            else:
                graph_copy = self.copy(copy_type='deep_without_data', name=name)

            assert not (neuron_names and connections),\
            "Specify either neuron_names or connections, not both."

            if neuron_names is not None:
                missing = [n for n in neuron_names if n not in graph_copy.neurons]
                if missing:
                    raise KeyError(f"Neurons not found in network: {missing}")

                selected_names = set(neuron_names)
                nodes_to_remove = [
                    node for node in list(graph_copy.nodes)
                    if node.name not in selected_names
                ]
                graph_copy.remove_nodes_from(nodes_to_remove)
                subgraph = graph_copy

                # subgraph_nodes = [graph_copy.neurons[name] for name in neuron_names]
                # subgraph = graph_copy.subgraph(subgraph_nodes)
                # subgraph.connections = {key: value for key, value in graph_copy.connections.items()\
                #                          if key[0] in subgraph_nodes and key[1] in subgraph_nodes}
            elif connections is not None:
                new_connections = [(graph_copy.neurons[conn[0].name], graph_copy.neurons[conn[1].name], conn[2])\
                                 for conn in connections]
                new_connections = [graph_copy.connections[key]._id for key in new_connections]
                subnet = graph_copy.edge_subgraph(new_connections) # That will put it through custom copy again.
                subgraph = NervousSystem(self.worm, network=name or self.name + "_subnetwork")
                subgraph.create_neurons_from(subnet, data=data)
                subgraph.create_connections_from(subnet, data=data)
                # subgraph.connections = {key: value for key, value in graph_copy.connections.items()\
                    #  if key in new_connections}
            else:
                subgraph = graph_copy
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

    @record("fold_network")
    def fold_network(self, fold_by, name=None, data='collect', exceptions=None,
                     self_loops=True, legacy=False):
        """
        Fold the network based on a partition.

        Args:
            fold_by (dict[str, list[str]]):
                Mapping from each merged-neuron name to the list of original
                neuron names that should collapse into it. Singleton lists
                are treated as renames.
            name (str, optional):
                Name for the resulting NervousSystem.
            data (str, optional): One of:
                - 'collect': Preserve every original edge as a separate
                  parallel in the folded view (default).
                - 'clean': Sum weights of parallel edges per
                  ``(folded_pre, folded_post, connection_type)`` and union
                  list-valued edge metadata (ligands, neurotransmitters,
                  putative receptors, receptor dicts).
            exceptions (list[str], optional):
                Neuron names that should NOT be folded into their class —
                they pass through to the folded view under their original
                names.
            self_loops (bool, optional):
                If False, intra-class edges (i.e. edges whose endpoints
                both belong to the same merged class) are dropped from
                the result. Defaults to True.
            legacy (bool, optional):
                Selects between two implementations that should be
                behaviourally equivalent. Default ``False`` uses the
                batch path which builds the folded view directly from
                the partition map in O(V + E). Set ``True`` to use the
                pre-batch implementation that does N-1 pair-wise
                ``contract_neurons`` calls per class (O(class_size ×
                growing_supernode_degree) per class — unusable at
                scale). Preserved during the rollout so callers can
                bisect parity issues, and scheduled for removal once
                the fast path has soaked.

        Returns:
            NervousSystem: The folded graph. Each merged neuron carries
            ``constituents``, ``is_merged``, ``constituent_subgraph`` and
            the merge-policy-resolved ``MERGE_TRACK_ATTRS``. Both code
            paths produce structurally equivalent NervousSystem objects;
            see ``_fold_network_batch`` for the precise preservation
            contract.
        """
        assert isinstance(fold_by, dict), "Enter a dictionary with neuron class\
            names as keys and the neurons to fold as values. If there is only one\
                neuron in the list of values, the neuron will be renamed to the key."

        if exceptions is None:
            exceptions = []

        # Validate every fold's destination name BEFORE touching the
        # graph. Renaming onto an unrelated existing neuron silently
        # produces duplicate names that crash subsequent copies; the
        # underlying ``contract_neurons`` has the same guard but
        # surfacing the error at the fold level gives a clearer message
        # and avoids partially applying a multi-fold dict.
        for merged_nodename, nodes_to_fold in fold_by.items():
            if (
                merged_nodename in self.neurons
                and merged_nodename not in nodes_to_fold
            ):
                raise ValueError(
                    f"Cannot fold into '{merged_nodename}': a neuron with "
                    f"that name already exists in the network and is not "
                    f"one of the neurons being folded "
                    f"({list(nodes_to_fold)!r})."
                )

        # The partition must be disjoint: a neuron can belong to at
        # most one merged class. Without this check the batch path
        # silently overwrites ``rename_map`` (last-write-wins) and the
        # legacy path also misbehaves — both produce silent data loss
        # / duplication rather than a clean error. Excepted members
        # are ignored: they pass through to the folded view regardless
        # of how many classes nominally list them.
        seen_assignments: dict[str, str] = {}
        for merged_nodename, nodes_to_fold in fold_by.items():
            for n in nodes_to_fold:
                if n in exceptions:
                    continue
                prior = seen_assignments.get(n)
                if prior is not None and prior != merged_nodename:
                    raise ValueError(
                        f"Cannot fold: neuron '{n}' is listed under "
                        f"multiple merged classes ('{prior}' and "
                        f"'{merged_nodename}'). Each neuron may appear "
                        f"in at most one class."
                    )
                seen_assignments[n] = merged_nodename

        if legacy:
            return self._fold_network_legacy(
                fold_by, name, data, exceptions, self_loops,
            )
        return self._fold_network_batch(
            fold_by, name, data, exceptions, self_loops,
        )

    def _fold_network_legacy(self, fold_by, name, data, exceptions, self_loops):
        """Pre-batch fold: pair-wise ``contract_neurons`` for every class
        member after the first, then optional ``contract_connections``.

        Preserved for parity comparison. O(class_size × supernode_degree)
        per class — unusable at scale (a 43k-member class on a 17M-edge
        graph wedges for many hours). The batch implementation
        (``_fold_network_batch``) is the default.
        """
        # Capture per-fold constituent subgraphs from the *pre-fold*
        # graph BEFORE any contraction mutates the working copy. Each
        # captured subnetwork holds the originally-selected set + all
        # internal edges + per-element attributes, so drill-down can
        # reconstruct the un-folded state without walking graph history.
        #
        # Why we capture at fold granularity (not the internal
        # contract_neurons pair-wise steps): the fold is the user-level
        # primitive. Hierarchy depth and provenance should count folds.
        # A single fold of N neurons is one hierarchy level, irrespective
        # of how many pair-wise contractions implement it internally.
        #
        # ``_silent=True`` keeps the per-fold subnetwork copies out of
        # the animal log — only the fold_network event is user-relevant.
        constituent_subgraphs = {}
        for merged_nodename, nodes_to_fold in fold_by.items():
            if len(nodes_to_fold) <= 1:
                continue
            present = [
                n for n in nodes_to_fold
                if n not in exceptions and n in self.neurons
            ]
            if len(present) <= 1:
                continue
            try:
                constituent_subgraphs[merged_nodename] = self.subnetwork(
                    neuron_names=present,
                    name=f"Constituents of {merged_nodename}",
                    _silent=True,
                )
            except Exception:
                # Defensive: a subnetwork failure must not block the fold.
                pass

        graph_copy = self.copy(copy_type='deep_with_data', name=name)
        # Internal pair-wise contractions are an implementation detail
        # of the fold — pass ``_silent=True`` so the animal log only
        # carries the user-level fold_network event, not N-1 noisy
        # contract_neurons events per fold.
        for merged_nodename, nodes_to_fold in fold_by.items():
            if len(nodes_to_fold) >1:
                merged_node = nodes_to_fold[0]
                for j in range(1,len(nodes_to_fold)):
                    npair = (merged_node, nodes_to_fold[j])
                    if not npair[0] in exceptions and not npair[1] in exceptions:
                        graph_copy.contract_neurons(npair, merged_nodename,
                                                   data=data, self_loops=self_loops,
                                                   _silent=True)
                        merged_node = merged_nodename
            else:
                graph_copy.neurons[nodes_to_fold[0]].name = merged_nodename

        graph_copy.update_network()
        graph_copy.update_neurons()
        graph_copy.reassign_connections()

        def _attach_constituent_subgraphs(result_graph):
            for merged_name, subg in constituent_subgraphs.items():
                if merged_name not in result_graph.neurons:
                    continue
                merged_neuron = result_graph.neurons[merged_name]
                merged_neuron.constituent_subgraph = subg
                result_graph.nodes[merged_neuron]['constituent_subgraph'] = subg

        if data == 'collect':
            _attach_constituent_subgraphs(graph_copy)
            return graph_copy
        if data == 'clean':
            parsed_conns = {}
            for e,conn in graph_copy.connections.items():
                if (e[0],e[1], conn.connection_type) not in parsed_conns:
                    parsed_conns[(e[0],e[1], conn.connection_type)] = []
                parsed_conns[(e[0],e[1], conn.connection_type)].append(conn)
            contracted_graph = graph_copy.contract_connections(parsed_conns, _silent=True)
            _attach_constituent_subgraphs(contracted_graph)
            return contracted_graph

    def _fold_network_batch(self, fold_by, name, data, exceptions, self_loops):
        """Batch fold: construct the folded NervousSystem directly from
        the partition map in O(V + E + Σ class_internal_edges).

        Preservation contract:
        - The folded NervousSystem is a fresh object with fresh Neuron
          and Connection instances. Mutable attribute *values* on the
          new neurons/connections may share references with the parent's
          counterparts (same as ``copy(copy_type='deep_with_data')`` and
          the legacy fold path). Use ``copy(copy_type='deep')`` if you
          need full attribute isolation.
        - Merged-neuron provenance matches ``contract_neurons`` exactly:
          per-member snapshots are captured via the same ``_snapshot``
          logic; merge policy on ``MERGE_TRACK_ATTRS`` (all-same →
          preserved value; mixed → ``MERGED_TYPE`` sentinel) is applied
          once across the snapshot set; constituents from transitively
          merged inputs are flattened into the new constituents dict
          (matching the ``contract_neurons`` line "Fold any constituents
          the target itself had from prior merges").
        - The merged neuron's ``constituent_subgraph`` is the pre-fold
          subnetwork captured by ``self.subnetwork(neuron_names=present,
          _silent=True)`` and is mirrored to the nx node dict so
          subsequent ``deep_with_data`` copies propagate it.
        - Edges:
          * ``data='collect'``: every parent edge becomes one folded
            edge with the original key and ``**edge_data`` preserved.
          * ``data='clean'``: parallels per
            ``(folded_pre, folded_post, connection_type)`` are summed
            (weights) and union'd (ligands / neurotransmitters /
            putative_neurotrasmitter_receptors / receptors), and the
            resulting Connection carries ``contraction_data`` mapping
            each pre-fold edge's ``_id`` → its pre-fold Connection
            object. Note: the legacy ``data='clean'`` path computes
            ``contraction_data`` keyed by the post-pair-wise edge IDs;
            keying by pre-fold IDs here is more directly useful for
            consumers (e.g. the FlyWire notebook reads
            ``conn.contraction_data[k].neurotransmitter`` — the
            attribute survives on either set of Connection objects).
        - ``self_loops=False`` drops intra-class edges from the folded
          view, matching ``nx.contracted_nodes(self_loops=False)``.
        - ``exceptions`` members pass through to the folded view under
          their original names (matching the legacy path: the pair-wise
          contraction loop skips pairs containing an excepted neuron).
        """
        # ---------------------------------------------------------------
        # Step 1: pre-fold constituent_subgraph capture in ONE O(V + E)
        # pass over the parent graph.
        #
        # The earlier draft of this step called ``self.subnetwork(...)``
        # per class, which iterates the entire parent edge list per call
        # (the view-based subnetwork filters by node-membership on every
        # edge access). That made the capture phase O(classes × E_parent)
        # — pathological at FlyWire's 510 classes × 17M edges scale.
        #
        # The fix here: bucket internal-to-class edges by a single sweep
        # of ``self.edges(...)``, then build each constituent NervousSystem
        # directly from its bucket. Per-class cost is now
        # O(class_size + class_internal_edges); total O(V + E).
        from collections import defaultdict

        class_members: dict[str, list[str]] = {}
        member_to_class: dict[str, str] = {}
        for merged_nodename, nodes_to_fold in fold_by.items():
            if len(nodes_to_fold) <= 1:
                continue
            present = [
                n for n in nodes_to_fold
                if n not in exceptions and n in self.neurons
            ]
            if len(present) <= 1:
                continue
            class_members[merged_nodename] = present
            for m in present:
                member_to_class[m] = merged_nodename

        # Single sweep over parent edges. Keep tuples of (u_name, v_name,
        # key, edge_data) so the build loop below can construct fresh
        # Connections without holding refs to the parent's Connection
        # objects (matches what self.subnetwork(...) would have done).
        by_class_edges: dict[str, list] = defaultdict(list)
        if class_members:
            for u, v, k, edge_data in self.edges(keys=True, data=True):
                cu = member_to_class.get(u.name)
                if cu is None:
                    continue
                if member_to_class.get(v.name) != cu:
                    continue
                by_class_edges[cu].append((u.name, v.name, k, edge_data))

        constituent_subgraphs: dict[str, 'NervousSystem'] = {}
        for cname, members in class_members.items():
            try:
                sub = NervousSystem(
                    self.worm,
                    network=f"Constituents of {cname}",
                )
                # Mirror what ``deep_with_data`` does: each new Neuron
                # gets the parent's nx node attribute dict as **kwargs.
                # Same preservation contract as the legacy
                # ``subnetwork(...)`` path, just scoped to the class.
                for member_name in members:
                    parent_n = self.neurons[member_name]
                    nx_node_data = dict(self.nodes[parent_n])
                    Neuron(member_name, sub, **nx_node_data)
                for un, vn, k, edge_data in by_class_edges.get(cname, ()):
                    src = sub.neurons[un]
                    dst = sub.neurons[vn]
                    # Preserve the parent edge's key + every attribute it
                    # carries. ``Connection(...)`` uses the kwargs to both
                    # ``add_edge`` (nx side) and to ``set_property`` (Python
                    # side) — identical to ``create_connections_from``.
                    ed = dict(edge_data)
                    ct = ed.pop('connection_type', 'chemical-synapse')
                    wt = ed.pop('weight', 1)
                    new_conn = Connection(src, dst, k, connection_type=ct,
                                          weight=wt, **ed)
                    sub.connections[(src, dst, new_conn.uid)] = new_conn
                sub.visualization_metadata = copy.deepcopy(self.visualization_metadata)
                constituent_subgraphs[cname] = sub
            except Exception:
                # Defensive: a single bad class must not block the fold.
                pass

        # ---------------------------------------------------------------
        # Step 2: build the partition map: original_name -> folded_name.
        # Excepted members pass through unchanged.
        # ---------------------------------------------------------------
        rename_map = {}
        for merged_nodename, nodes_to_fold in fold_by.items():
            present = [n for n in nodes_to_fold
                       if n not in exceptions and n in self.neurons]
            if len(present) > 1:
                for m in present:
                    rename_map[m] = merged_nodename
            elif len(present) == 1:
                # Singleton: rename the single member to the class key.
                # Matches the legacy ``else`` branch (line 660-661 in the
                # pre-refactor source) which renamed unconditionally.
                rename_map[present[0]] = merged_nodename

        # ---------------------------------------------------------------
        # Step 3: snapshot pre-fold provenance per merged class. Mirrors
        # ``contract_neurons``'s ``_snapshot`` helper and merge-policy
        # loop. Single-member classes don't enter this step (their
        # "merged" neuron is just the renamed source — its existing
        # constituents/etc. survive via the nx-node-data copy in step 5).
        # ---------------------------------------------------------------
        def _snapshot(neuron, n_name):
            snap = {'name': n_name}
            for attr in MERGE_TRACK_ATTRS:
                snap[attr] = getattr(neuron, attr, '')
            return snap

        merged_class_snapshots = {}  # cname -> {orig_name: snapshot}
        merged_class_resolved_attrs = {}  # cname -> {merge-policy attrs}
        for merged_nodename in constituent_subgraphs:
            present = [n for n in fold_by[merged_nodename]
                       if n not in exceptions and n in self.neurons]
            snapshots = {}
            for m in present:
                mn = self.neurons[m]
                # Flatten constituents from any transitively-merged source.
                if getattr(mn, 'constituents', None):
                    for child_name, child_meta in mn.constituents.items():
                        snapshots.setdefault(child_name, dict(child_meta))
                snapshots.setdefault(m, _snapshot(mn, m))
            merged_class_snapshots[merged_nodename] = snapshots

            resolved = {}
            for attr in MERGE_TRACK_ATTRS:
                values = {meta.get(attr) for meta in snapshots.values()}
                values.discard(None)
                values.discard('')
                if len(values) == 1:
                    resolved[attr] = next(iter(values))
                elif len(values) > 1:
                    resolved[attr] = MERGED_TYPE
                # else: leave attr unset
            merged_class_resolved_attrs[merged_nodename] = resolved

        # ---------------------------------------------------------------
        # Step 4: construct the folded NervousSystem and its neurons.
        # ---------------------------------------------------------------
        folded = NervousSystem(self.worm, network=name or (self.name + "_folded"))
        folded.visualization_metadata = copy.deepcopy(self.visualization_metadata)

        created = set()
        # 4a. Multi-member merged neurons.
        for cname, snapshots in merged_class_snapshots.items():
            attrs = dict(merged_class_resolved_attrs[cname])
            # ``dict(snapshots)`` creates a fresh top-level constituents
            # dict (snapshot meta values are still shared, matching the
            # behaviour of contract_neurons line 936). The freshness of
            # the OUTER dict matters: prevents the
            # constituents-leak-across-folds bug captured by
            # ``test_hierarchical_fold_keeps_inner_constituents_intact``.
            attrs['constituents'] = dict(snapshots)
            Neuron(cname, folded, **attrs)
            created.add(cname)

        # 4b. Singletons and pass-through neurons. Pull attributes from
        # the parent's nx node attribute dict (same source-of-truth as
        # ``create_neurons_from(self, data=True)``).
        for orig_name, orig_neuron in self.neurons.items():
            target = rename_map.get(orig_name, orig_name)
            if target in created:
                continue
            nx_data = dict(self.nodes[orig_neuron])
            Neuron(target, folded, **nx_data)
            created.add(target)

        # ---------------------------------------------------------------
        # Step 5: aggregate edges into the folded view.
        # ---------------------------------------------------------------
        if data == 'collect':
            for u, v, k, edge_data in self.edges(keys=True, data=True):
                fu = rename_map.get(u.name, u.name)
                fv = rename_map.get(v.name, v.name)
                if fu == fv and not self_loops:
                    continue
                src = folded.neurons[fu]
                dst = folded.neurons[fv]
                conn = Connection(src, dst, k, **edge_data)
                folded.connections[(src, dst, conn.uid)] = conn
        elif data == 'clean':
            # Aggregate parallels per (folded_pre, folded_post, type).
            # Mirrors the union-metadata logic in ``contract_connections``.
            from collections import defaultdict
            bucket = defaultdict(lambda: {
                'weight': 0.0,
                'contraction_data': {},  # orig_id -> orig Connection
                'ligands': [], 'lig_seen': set(),
                'nts': [], 'nt_seen': set(),
                'putative': [], 'put_seen': set(),
                'receptors': {},
            })
            for u, v, k, edge_data in self.edges(keys=True, data=True):
                fu = rename_map.get(u.name, u.name)
                fv = rename_map.get(v.name, v.name)
                if fu == fv and not self_loops:
                    continue
                ct = edge_data.get('connection_type', 'chemical-synapse')
                b = bucket[(fu, fv, ct)]
                b['weight'] += float(edge_data.get('weight', 1) or 0)
                # The connections dict on a NervousSystem is a custom
                # ``ConnectionGroup`` and has no ``.get`` method — fall
                # back to membership-check + bracket lookup.
                if (u, v, k) in self.connections:
                    orig_conn = self.connections[(u, v, k)]
                    b['contraction_data'][orig_conn._id] = orig_conn
                # Union list-valued edge metadata (order-preserving dedupe).
                for lig in (edge_data.get('ligands') or []):
                    key = lig if isinstance(lig, str) else repr(lig)
                    if key not in b['lig_seen']:
                        b['lig_seen'].add(key)
                        b['ligands'].append(lig)
                for nt in (edge_data.get('neurotransmitters') or []):
                    key = nt if isinstance(nt, str) else repr(nt)
                    if key not in b['nt_seen']:
                        b['nt_seen'].add(key)
                        b['nts'].append(nt)
                for pair in (edge_data.get('putative_neurotrasmitter_receptors') or []):
                    key = tuple(pair) if isinstance(pair, (list, tuple)) else pair
                    if key not in b['put_seen']:
                        b['put_seen'].add(key)
                        b['putative'].append(pair)
                receptors = edge_data.get('receptors')
                if isinstance(receptors, dict):
                    for rk, rv in receptors.items():
                        b['receptors'].setdefault(rk, rv)

            for (fu, fv, ct), b in bucket.items():
                src = folded.neurons[fu]
                dst = folded.neurons[fv]
                new_conn = Connection(src, dst, connection_type=ct, weight=b['weight'])
                new_conn.set_property('contraction_data', dict(b['contraction_data']))
                if b['ligands']:
                    new_conn.set_property('ligands', list(b['ligands']))
                if b['nts']:
                    new_conn.set_property('neurotransmitters', list(b['nts']))
                if b['putative']:
                    new_conn.set_property('putative_neurotrasmitter_receptors', list(b['putative']))
                if b['receptors']:
                    new_conn.set_property('receptors', dict(b['receptors']))
                folded.connections[(src, dst, new_conn.uid)] = new_conn
        else:
            raise ValueError(f"Unknown data mode for fold_network: {data!r}")

        # ---------------------------------------------------------------
        # Step 6: attach constituent subgraphs and mirror merge-track
        # attrs to the nx node dict (same shape contract_neurons +
        # _attach_constituent_subgraphs produce on the legacy path).
        # ---------------------------------------------------------------
        for cname, subg in constituent_subgraphs.items():
            if cname not in folded.neurons:
                continue
            merged_neuron = folded.neurons[cname]
            merged_neuron.constituent_subgraph = subg
            folded.nodes[merged_neuron]['constituent_subgraph'] = subg
            # Mirror constituents + MERGE_TRACK_ATTRS to nx node dict so
            # subsequent ``deep_with_data`` copies propagate them
            # (matches the mirror block at the bottom of contract_neurons).
            folded.nodes[merged_neuron]['constituents'] = merged_neuron.constituents
            for attr in MERGE_TRACK_ATTRS:
                if hasattr(merged_neuron, attr):
                    folded.nodes[merged_neuron][attr] = getattr(merged_neuron, attr)

        return folded


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
    def adjacency(self, order=None, weighted=False, connection_type=None):
        """
        Output the adjacency matrix for the network ordered by neurons.

        Args:
            order:
                Optional ordered list of neuron names or neuron objects.
                When omitted, neurons are ordered deterministically by name.
            weighted (bool):
                If ``True``, matrix entries are summed edge weights. Otherwise
                returns a binary adjacency matrix.
            connection_type:
                Optional string or list of strings selecting which connection
                types to include. Supports convenience aliases such as
                ``"chemical"``, ``"gap-junction"``, and ``"bulk"`` in
                addition to exact connection type names.
        """
        if order is None:
            ordered_neurons = sorted(self.neurons.values(), key=lambda neuron: neuron.name)
        else:
            ordered_neurons = []
            for item in order:
                if isinstance(item, Neuron):
                    ordered_neurons.append(item)
                elif isinstance(item, str):
                    if item not in self.neurons:
                        raise KeyError(f"Neuron '{item}' not found in network")
                    ordered_neurons.append(self.neurons[item])
                else:
                    raise TypeError("order entries must be neuron names or Neuron objects")

        if connection_type is None:
            selected_types = None
            include_bulk = False
        else:
            selectors = [connection_type] if isinstance(connection_type, str) else list(connection_type)
            selected_types = set()
            include_bulk = False
            for selector in selectors:
                if not isinstance(selector, str):
                    raise TypeError("connection_type entries must be strings")
                key = selector.strip().lower()
                if key in {"chemical", "chemical-synapse", "chemical_synapse"}:
                    selected_types.add("chemical-synapse")
                elif key in {"gap", "gap-junction", "gap_junction"}:
                    selected_types.add("gap-junction")
                elif key == "bulk":
                    include_bulk = True
                else:
                    selected_types.add(selector)

        index_map = {neuron: idx for idx, neuron in enumerate(ordered_neurons)}
        adjacency = np.zeros((len(ordered_neurons), len(ordered_neurons)), dtype=float)

        for pre, post, edge_data in self.edges(data=True):
            if pre not in index_map or post not in index_map:
                continue

            edge_type = edge_data.get("connection_type")
            if selected_types is not None:
                is_bulk = edge_type not in {"chemical-synapse", "gap-junction"}
                if edge_type not in selected_types and not (include_bulk and is_bulk):
                    continue

            adjacency[index_map[pre], index_map[post]] += edge_data.get("weight", 1.0) if weighted else 1.0

        if not weighted:
            adjacency = (adjacency > 0).astype(int)

        return adjacency
    
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
        
    @record("contract_neurons")
    def contract_neurons(self, pair, contracted_name, data='collect', copy_graph=False, self_loops=True):
        """
        Contract ``target`` into ``source``, redirecting target's edges
        to source and renaming source to ``contracted_name``.

        Args:
            pair (tuple[str, str]):
                ``(source_name, target_name)``. Source survives (renamed
                to ``contracted_name``); target is removed and its edges
                are redirected onto source.
            contracted_name (str):
                New name for the surviving neuron.
            data (str, optional):
                No-op at the core layer — accepted for backwards
                compatibility with callers that use it as a hint for
                post-merge cleanup. The cedne_web backend uses
                ``data='clean'`` to indicate that
                ``contract_connections`` should be called after a chain
                of contractions to collapse parallel edges.
            copy_graph (bool, optional):
                If True, work on a deep copy and return it; the original
                is untouched. If False (default), mutate in place and
                return None.
            self_loops (bool, optional):
                Forwarded to ``nx.contracted_nodes``. If False, drops
                any self-loop produced by edges between the merged pair.

        Returns:
            NervousSystem | None:
                The new graph if ``copy_graph=True``; ``None`` otherwise
                (mutates in place).

        Notes:
            Tracks merge provenance on the surviving neuron via three
            new Neuron API surfaces:

            * ``Neuron.constituents``: dict mapping each pre-merge name
              to a snapshot ``{'name', 'type'}``. Transitive — merging
              A+B then merging the result with C yields constituents
              ``{A, B, C}``.
            * ``Neuron.is_merged``: bool, True iff ``constituents`` is
              non-empty.
            * ``Neuron.constituent_types``: sorted list of distinct
              constituent types, derived from ``constituents`` so the
              two cannot drift out of sync.

            Type-merge policy:

            * If all constituents share a single type → that type is
              preserved on the surviving neuron.
            * If types differ → ``Neuron.type`` is set to the sentinel
              ``MERGED_TYPE`` (``'merged'``) so analyses that switch on
              type can branch on the merged case explicitly rather than
              silently inheriting one constituent's type.
        """
        source_name, target_name = pair

        if copy_graph:
            new_graph = self.copy()
            # FIX: previously this passed a 3-tuple as `pair`, which the
            # destructure rejects with ValueError. The 2-tuple goes in
            # `pair`; `contracted_name` is its own argument.
            new_graph.contract_neurons(
                (source_name, target_name),
                contracted_name,
                data=data, copy_graph=False, self_loops=self_loops,
            )
            return new_graph

        src = self.neurons[source_name]
        tgt = self.neurons[target_name]

        # Refuse to rename onto an unrelated existing neuron. Without this
        # check the rename `src.name = contracted_name` silently produces
        # two neurons with the same name in the network (one being the
        # surviving src, the other being whatever was already there),
        # corrupting subsequent lookups and triggering "already exists"
        # crashes deep inside `copy()` / `create_neurons_from`.
        if (
            contracted_name in self.neurons
            and contracted_name != source_name
            and contracted_name != target_name
        ):
            raise ValueError(
                f"Cannot contract into '{contracted_name}': a neuron with "
                f"that name already exists in this network and isn't one of "
                f"the pair being merged ({source_name!r}, {target_name!r})."
            )

        # ----- Merge-provenance bookkeeping (new in Issue 10A) -------------
        # We track constituents on the surviving neuron explicitly, BEFORE
        # nx.contracted_nodes mutates the graph, because:
        #   (a) transitive merges (A+B → A_m, then A_m+C → A_m) accumulate
        #       cleanly through `setdefault`,
        #   (b) we don't depend on networkx's internal `contraction` attr,
        #       which has changed semantics across versions.
        # Snapshot helper: capture every MERGE_TRACK_ATTRS value on a
        # neuron at merge time. Stored under the original name so the
        # merge policy below can reason about all of them uniformly.
        def _snapshot(neuron, name):
            snap = {'name': name}
            for attr in MERGE_TRACK_ATTRS:
                snap[attr] = getattr(neuron, attr, '')
            return snap

        if not getattr(src, 'constituents', None):
            # First merge for src — record its pre-merge identity.
            # src.name is still the original here (rename happens below).
            src.constituents = {src.name: _snapshot(src, src.name)}
        else:
            # Re-merge: src is already a merged neuron. The dict it
            # carries may be shared with previously-captured constituent
            # subgraphs (``create_neurons_from`` propagates node attrs
            # by reference, not deep copy, so the same dict is reachable
            # through any nested ``constituent_subgraph`` taken before
            # this contraction). The setdefault calls below mutate that
            # dict in place; without taking ownership of a fresh copy
            # first, the captured snapshot's view of its constituents
            # would silently grow to include this fold's other inputs
            # too — corrupting drill-down provenance.
            src.constituents = dict(src.constituents)

        # Fold any constituents the target itself had from prior merges so
        # nested merges produce a flat list of original neurons.
        if getattr(tgt, 'constituents', None):
            for c_name, c_meta in tgt.constituents.items():
                src.constituents.setdefault(c_name, c_meta)

        # Add the target as a constituent (keyed by its current name,
        # which is its original name unless the target itself was already
        # a merged neuron — in which case the fold above already covered
        # the originals and target_name is the merged-name placeholder
        # we still record for traceability).
        src.constituents.setdefault(target_name, _snapshot(tgt, target_name))

        # Apply the merge policy across every tracked enum-like attribute:
        # all-same → keep that value; mixed → set to the MERGED_TYPE
        # sentinel; no usable values → leave the attribute alone. type,
        # category, and modality all need this — silent inheritance from
        # the source neuron was the bug Issue 10 fixed for type, and the
        # same fix needs to apply to category/modality (which are
        # equally enum-like and equally susceptible to silent overwrite).
        for attr in MERGE_TRACK_ATTRS:
            values = {meta.get(attr) for meta in src.constituents.values()}
            values.discard(None)
            values.discard('')
            if len(values) == 1:
                setattr(src, attr, next(iter(values)))
            elif len(values) > 1:
                setattr(src, attr, MERGED_TYPE)

        # Mirror `constituents` and any updated tracked attributes to
        # the networkx node attribute dict so paths that reconstruct
        # neurons via `create_neurons_from(data=True)` (used by
        # contract_connections, copy(copy_type='deep_with_data'), etc.)
        # propagate the merge state instead of dropping it.
        self.nodes[src]['constituents'] = src.constituents
        for attr in MERGE_TRACK_ATTRS:
            if hasattr(src, attr):
                self.nodes[src][attr] = getattr(src, attr)

        # NOTE on provenance: pair-wise contraction is an implementation
        # detail of folding. We deliberately do NOT attach a
        # ``constituent_subgraph`` here — that record belongs at the
        # user-level operation (``fold_network``), which captures the
        # full flat subgraph of the originally-selected set on the
        # resulting merged neuron. Counting pair-wise steps would
        # over-count hierarchy depth (a single fold of N neurons would
        # look like N-1 nested levels rather than 1).

        # ----- Existing edge-contraction flow ------------------------------
        for _cid, conn in src.get_connections().items():
            conn.set_property('_id', conn._id)
        for _cid, conn in tgt.get_connections().items():
            conn.set_property('_id', conn._id)
        nx.contracted_nodes(self, src, tgt, copy=False, self_loops=self_loops)
        src.name = contracted_name
        self.update_neurons()
    
    @record("contract_connections")
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
            # Union semantic edge metadata across the constituent connections
            # so folding / merging does not silently drop it. Order-preserving
            # dedupe for list-valued properties; dict merge for receptors.
            merged_ligands = []
            merged_nts = []
            merged_putative = []
            merged_receptors = {}
            lig_seen, nt_seen, put_seen = set(), set(), set()
            for conn in conns:
                weight += conn.weight
                contraction_data[conn._id] = conn
                for lig in getattr(conn, 'ligands', []) or []:
                    key = lig if isinstance(lig, str) else repr(lig)
                    if key not in lig_seen:
                        lig_seen.add(key)
                        merged_ligands.append(lig)
                for nt in getattr(conn, 'neurotransmitters', []) or []:
                    key = nt if isinstance(nt, str) else repr(nt)
                    if key not in nt_seen:
                        nt_seen.add(key)
                        merged_nts.append(nt)
                for pair in getattr(conn, 'putative_neurotrasmitter_receptors', []) or []:
                    key = tuple(pair) if isinstance(pair, (list, tuple)) else pair
                    if key not in put_seen:
                        put_seen.add(key)
                        merged_putative.append(pair)
                receptors = getattr(conn, 'receptors', None)
                if isinstance(receptors, dict):
                    for rk, rv in receptors.items():
                        if rk not in merged_receptors:
                            merged_receptors[rk] = rv

            n1 = empty_graph_copy.neurons[contraction[0].name]
            n2 = empty_graph_copy.neurons[contraction[1].name]
            new_conn = Connection(n1, n2, connection_type=contraction[2], weight=weight)
            new_conn.set_property('contraction_data', copy.copy(contraction_data))
            if merged_ligands:
                new_conn.set_property('ligands', merged_ligands)
            if merged_nts:
                new_conn.set_property('neurotransmitters', merged_nts)
            if merged_putative:
                new_conn.set_property('putative_neurotrasmitter_receptors', merged_putative)
            if merged_receptors:
                new_conn.set_property('receptors', merged_receptors)
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
        elif copy_type == 'deep_with_data':
            deep_copy = NervousSystem(self.worm, network=name or self.name + "_copy")
            deep_copy.create_neurons_from(self, data=True)
            deep_copy.create_connections_from(self, data=True)
            deep_copy.visualization_metadata = copy.deepcopy(self.visualization_metadata)
            return deep_copy
        elif copy_type == 'deep_without_data':
            deep_copy = NervousSystem(self.worm, network=name or self.name + "_copy")
            deep_copy.create_neurons_from(self, data=False)
            deep_copy.create_connections_from(self, data=False)
            deep_copy.visualization_metadata = copy.deepcopy(self.visualization_metadata)
            return deep_copy
        else:
            raise ValueError("copy_type must be 'deep', 'shallow'")
    
    def copy_neurons(self, name=None, data=False):
        """ Copies the neurons from the network and creates a new network with them"""
        new_network = NervousSystem(self.worm, network=name or self.name + "_copy")
        new_network.create_neurons_from(self, data=data)
        return new_network

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
    
    def shortest_path(self, source, target, weight=None, method='dijkstra'):
        """
        Finds a single shortest path between two neurons in the network.
        """
        return nx.shortest_path(self, source, target, weight=weight, method=method)

    def shortest_paths(self, source, target, weight=None, method='dijkstra'):
        """
        Finds all shortest paths between two neurons in the network.
        """
        return nx.all_shortest_paths(self, source, target, weight=weight, method=method)

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

    def to_dict(self) -> dict:
        """Serialize the full network to a plain Python dictionary.

        Composes ``Neuron.to_dict()`` and ``Connection.to_dict()`` for each
        node and edge, then attaches group membership, group summaries,
        network-level metadata, and visualization hints.

        Returns:
            dict: A JSON-compatible dictionary with keys:
                - ``name``: Network name.
                - ``organism``: Organism name (from worm), if available.
                - ``nodes``: List of neuron dicts.
                - ``links``: List of connection dicts.
                - ``groups``: List of group summary dicts.
                - ``visualization_metadata``: Arbitrary viz hints.
        """
        # -- Group membership lookups --
        node_groups = {}
        link_groups = {}
        for gname, group in self.groups.items():
            if hasattr(group, 'neurons'):  # NeuronGroup
                for nname in group.neurons:
                    node_groups.setdefault(nname, []).append(gname)
            if hasattr(group, 'connections'):  # ConnectionGroup
                for conn_id in group.connections:
                    key = (
                        conn_id[0].name if hasattr(conn_id[0], 'name') else conn_id[0],
                        conn_id[1].name if hasattr(conn_id[1], 'name') else conn_id[1],
                        conn_id[2],
                    )
                    link_groups.setdefault(key, []).append(gname)

        # -- Nodes --
        nodes = []
        for nname, neuron in self.neurons.items():
            nd = neuron.to_dict()
            if nname in node_groups:
                nd['groups'] = node_groups[nname]
            nodes.append(nd)

        # -- Links --
        links = []
        for (u, v, k), conn in self.connections.items():
            ld = conn.to_dict()
            lk = (u.name, v.name, k)
            if lk in link_groups:
                ld['groups'] = link_groups[lk]
            links.append(ld)

        # -- Groups summary --
        groups_list = []
        from .neuron import NeuronGroup
        from .connection import ConnectionGroup
        for gname, g in self.groups.items():
            if isinstance(g, NeuronGroup):
                groups_list.append({
                    "name": gname, "type": "neuron",
                    "count": len(g),
                    "members": list(g.neurons),
                })
            elif isinstance(g, ConnectionGroup):
                groups_list.append({
                    "name": gname, "type": "connection",
                    "count": len(g),
                    "members": [
                        f"{c.pre.name}->{c.post.name}" for c in g.members
                    ],
                })

        # -- Assemble result with network-level attributes --
        result = {
            "name": self.name,
            "nodes": nodes,
            "links": links,
            "groups": groups_list,
            "visualization_metadata": getattr(
                self, 'visualization_metadata', {}
            ),
        }

        # Include organism metadata if present
        if hasattr(self, 'worm') and self.worm:
            result["organism"] = getattr(self.worm, 'name', None)

        return result

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
            members (Iterable[Union[Neuron, str]]):
                The members of the group. Each entry may be a ``Neuron``
                instance or a neuron name (string); strings are resolved
                against ``self.neurons``.
            group_name (str, optional):
                The name of the neuron group. Auto-generated if omitted.

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
            members (Iterable[Connection]):
                The members of the group. Each entry must be a ``Connection``
                instance — connection identity is the ``(pre, post, uid)``
                triple, not a string name.
            group_name (str, optional):
                The name of the connection group. Auto-generated if omitted.

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
