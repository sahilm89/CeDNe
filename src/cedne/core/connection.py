"""
Connection primitives for CeDNe.

This module defines the core data structures for representing connections between cells
in the nervous system. It includes:

- `Connection`: Base class for all types of connections.
- `ChemicalSynapse`: Specialized connection for chemical synapses.
- `GapJunction`: Specialized connection for gap junctions.
- `BulkConnection`: Specialized connection for bulk connections.
- `ConnectionGroup`: Container for managing sets of connections.
- `Path`: A sequence of connections between cells.

Each connection type can maintain its own properties and weights while sharing common
functionality through the base Connection class.
"""

__author__ = "Sahil Moza"
__date__ = "2025-04-06"
__license__ = "MIT"

import networkx as nx
from collections import Counter
import numpy as np
from .io import generate_random_string
from typing import Optional, List, Dict, Any, Union, TYPE_CHECKING
from numbers import Number

if TYPE_CHECKING:
    from .neuron import Neuron
    from .network import NervousSystem

class Connection:
    ''' This class represents a connection between two cells. '''
    def __init__(self, pre: 'Neuron', post: 'Neuron', uid=None, connection_type='chemical-synapse', **kwargs):
        """
        Initializes a new instance of the Connection class.

        Args:
            pre (Neuron): 
                The neuron sending the connection.
            post (Neuron): 
                The neuron receiving the connection.
            uid (int, optional): 
                The unique identifier for the connection.
            connection_type (str, optional): 
                The type of the connection.
            weight (float, optional): 
                The weight of the connection. Must be a numeric value.
        Raises:
            ValueError: If weight is not a numeric value.
            AssertionError: If pre and post neurons are from different networks.
        """
        self.pre = pre
        self.post = post
        self.network = post.network
        
        # Validate weight
        weight = kwargs.pop('weight', 1)
        if not (isinstance(weight, Number) or isinstance(weight, np.number)):
            raise ValueError("Weight must be a numeric value, but is of type {}".format(type(weight)))
        self.weight = weight
        
        if pre.network != post.network:
            raise AssertionError("The Nervous Systems of the pre and post neurons must be the same.")
        
        if not uid:
            self.uid = self.network.add_edge(pre, post, weight=self.weight, connection_type=connection_type, **kwargs)
        else:
            self.uid = uid
            self.network.add_edge(pre, post, key=uid, weight=self.weight, connection_type=connection_type, **kwargs)

        self._id = (pre, post, self.uid)
        self.connection_type = connection_type
        

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
    def __init__(self, pre, post, uid=0, connection_type='chemical-synapse', weight=1, **kwargs):
        super().__init__(pre, post, uid=uid, connection_type=connection_type, weight=weight, **kwargs)
        self.position= kwargs.pop('position', {'AP': 0, 'LR': 0, 'DV': 0})

class GapJunction(Connection):
    ''' This is a convenience class that represents connections of type gap junctions.'''
    def __init__(self, pre, post, uid=1, connection_type='gap-junction', weight=1, **kwargs):
        super().__init__(pre, post, uid=uid, connection_type=connection_type, weight=weight, **kwargs)
        self.position= kwargs.pop('position', {'AP': 0, 'LR': 0, 'DV': 0})

class BulkConnection(Connection):
    ''' This is a convenience class that represents connections of type neuropeptide-receptors.'''
    def __init__(self, pre, post, uid, connection_type, weight=1, **kwargs):
        super().__init__(pre, post, uid=uid, connection_type=connection_type, weight=weight, **kwargs)

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
        if TYPE_CHECKING:
            from .neuron import Neuron
            if isinstance(member, Neuron):
                return member in self.neurons
        return member in self.connections

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

    def update_weights(self, weight, delta=False):
        """
        Updates weights for all connections in the group.
        
        Args:
            weight (float): The new weight value or weight delta.
            delta (bool, optional): If True, weight is added to current weights.
                                  If False, weight replaces current weights.
        
        Raises:
            ValueError: If weight is not a numeric value.
        """
        if not (isinstance(weight, Number) or isinstance(weight, np.number)):
            raise ValueError("Weight must be a numeric value, but is of type {}".format(type(weight)))
            
        for connection in self.members:
            connection.update_weight(weight, delta)

    def update_weights_by_function(self, weight_function):
        """
        Updates weights for all connections using a custom function.
        
        Args:
            weight_function (callable): A function that takes a Connection object
                                      and returns a new weight value.
        
        Raises:
            ValueError: If weight_function is not callable or returns non-numeric values.
        """
        if not callable(weight_function):
            raise ValueError("weight_function must be callable")
            
        for connection in self.members:
            new_weight = weight_function(connection)
            if not (isinstance(new_weight, Number) or isinstance(new_weight, np.number)):
                raise ValueError("weight_function must return numeric values")
            connection.update_weight(new_weight)

    def filter_by_type(self, connection_type):
        """
        Returns a new ConnectionGroup containing only connections of the specified type.
        
        Args:
            connection_type (str): The type of connections to filter for.
            
        Returns:
            ConnectionGroup: A new group containing only the filtered connections.
        """
        filtered_members = [m for m in self.members if m.connection_type == connection_type]
        return ConnectionGroup(self.network, filtered_members, f"{self.group_name}-{connection_type}")

    def filter_by_property(self, property_name, property_value):
        """
        Returns a new ConnectionGroup containing only connections with the specified property value.
        
        Args:
            property_name (str): The name of the property to filter by.
            property_value: The value to match.
            
        Returns:
            ConnectionGroup: A new group containing only the filtered connections.
        """
        filtered_members = [m for m in self.members if hasattr(m, property_name) and 
                          getattr(m, property_name) == property_value]
        return ConnectionGroup(self.network, filtered_members, 
                             f"{self.group_name}-{property_name}-{property_value}")

    def filter_by_function(self, filter_function):
        """
        Returns a new ConnectionGroup containing only connections that pass the filter function.
        
        Args:
            filter_function (callable): A function that takes a Connection object
                                      and returns True if the connection should be included.
            
        Returns:
            ConnectionGroup: A new group containing only the filtered connections.
        """
        if not callable(filter_function):
            raise ValueError("filter_function must be callable")
            
        filtered_members = [m for m in self.members if filter_function(m)]
        return ConnectionGroup(self.network, filtered_members, f"{self.group_name}-filtered")

    def get_statistics(self):
        """
        Returns statistics about the connections in the group.
        
        Returns:
            dict: A dictionary containing statistics about the connections.
        """
        stats = {
            'count': len(self.members),
            'weight_mean': sum(m.weight for m in self.members) / len(self.members) if self.members else 0,
            'weight_min': min(m.weight for m in self.members) if self.members else 0,
            'weight_max': max(m.weight for m in self.members) if self.members else 0,
            'types': Counter([m.connection_type for m in self.members])
        }
        return stats

class Path(ConnectionGroup):
    ''' This is a sequence of Connections in the network.'''
    def __init__(self, network, members=None, group_name=None):
        if group_name is None:
            group_name = 'Path-'+ generate_random_string(8)
        
        if members is None:
            members = []
            self.source = None
            self.target = None
        else:
            assert all([isinstance(m, Connection)for m in members]),\
                  "Path members must be of type Connection"
            if len(members) > 1:
                # Check continuity only if there's more than one connection
                for i in range(len(members)-1):
                    if members[i].post != members[i+1].pre:
                        raise AssertionError("Path members must be continuous connections from source to target")
            if members:
                self.source = members[0].pre
                self.target = members[-1].post
            else:
                self.source = None
                self.target = None
                
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

    def get_length(self):
        """
        Returns the number of connections in the path.
        
        Returns:
            int: The number of connections in the path.
        """
        return len(self.members)

    def get_total_weight(self):
        """
        Returns the sum of weights of all connections in the path.
        
        Returns:
            float: The sum of connection weights.
        """
        return sum(conn.weight for conn in self.members)

    def get_average_weight(self):
        """
        Returns the average weight of connections in the path.
        
        Returns:
            float: The average connection weight.
        """
        if not self.members:
            return 0.0
        return self.get_total_weight() / len(self.members)

    def get_min_weight(self):
        """
        Returns the minimum weight among all connections in the path.
        
        Returns:
            float: The minimum connection weight.
        """
        if not self.members:
            return 0.0
        return min(conn.weight for conn in self.members)

    def get_max_weight(self):
        """
        Returns the maximum weight among all connections in the path.
        
        Returns:
            float: The maximum connection weight.
        """
        if not self.members:
            return 0.0
        return max(conn.weight for conn in self.members)

    def reverse(self):
        """
        Creates a new path with connections in reverse order.
        
        Returns:
            Path: A new path with reversed connections.
        """
        if not self.members:
            return Path(self.network)
            
        reversed_connections = []
        for conn in reversed(self.members):
            # Create a new connection with reversed pre/post neurons
            reversed_conn = Connection(conn.post, conn.pre, 
                                     connection_type=conn.connection_type,
                                     weight=conn.weight)
            reversed_connections.append(reversed_conn)
            
        return Path(self.network, reversed_connections, f"{self.group_name}-reversed")

    def concatenate(self, other_path):
        """
        Creates a new path by concatenating this path with another path.
        
        Args:
            other_path (Path): The path to concatenate with.
            
        Returns:
            Path: A new path containing all connections from both paths.
            
        Raises:
            AssertionError: If the paths cannot be concatenated (target of first path
                          must match source of second path).
        """
        if not self.members:
            return Path(self.network, other_path.members.copy())
        if not other_path.members:
            return Path(self.network, self.members.copy())
            
        if self.target != other_path.source:
            raise AssertionError("Cannot concatenate paths: target of first path must match source of second path")
            
        combined_connections = self.members.copy() + other_path.members.copy()
        return Path(self.network, combined_connections, f"{self.group_name}-{other_path.group_name}")

    def is_valid(self):
        """
        Checks if the path is valid (all connections are continuous).
        
        Returns:
            bool: True if the path is valid, False otherwise.
        """
        if not self.members:
            return True
            
        try:
            for i in range(len(self.members)-1):
                if self.members[i].post != self.members[i+1].pre:
                    return False
            return True
        except Exception:
            return False

    def get_neurons(self):
        """
        Returns a list of neurons in the path in order.
        
        Returns:
            list: List of neurons from source to target.
        """
        if not self.members:
            return []
            
        neurons = [self.source]
        for conn in self.members:
            neurons.append(conn.post)
        return neurons

    def get_connection_types(self):
        """
        Returns a list of connection types in the path in order.
        
        Returns:
            list: List of connection types.
        """
        return [conn.connection_type for conn in self.members]