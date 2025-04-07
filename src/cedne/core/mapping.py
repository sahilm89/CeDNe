"""
Mapping relationships across nervous systems in CeDNe.

This module defines data structures for comparing and aligning neural networks, 
such as between trained and naive worms, males and hermaphrodites, or different datasets.

Key class:
- `Mapping`: Represents a correspondence between neurons or subgraphs 
  across two `NervousSystem` objects.
"""

from .neuron import Neuron
from .connection import Connection
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
        for key, value in kwargs.items():
            self.set_property(key, value)
    
    def set_property(self, key, value):
        """
        Set a property of the map.
        Args:
            key (str): The name of the property.
            value: The value of the property.
        """
        setattr(self, key, value)

    def get_property(self, key):
        """
        Get a property of the map.
        Args:
            key (str): The name of the property.
        Returns:
            The value of the property.
        """
        return getattr(self, key)
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
        self.mapping_cardinality = (len(self.mapping_dict.keys()), len(self.mapping_dict.values()))
        if map_type == 'node':
            for node in mapping_dict.keys():
                graph_1.nodes[node]['map'] = mapping_dict[node]
        if map_type == 'edge':
            for edge in mapping_dict.keys():
                graph_1.edges[edge]['map'] = mapping_dict[edge]
                for j,node in enumerate(edge):
                    graph_1.nodes[node]['map'] = mapping_dict[edge][j]