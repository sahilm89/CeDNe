"""
Connection primitives for CeDNe neural networks.

This module defines the data structures representing edges in the 
neural network graph, including chemical synapses, gap junctions, 
and bulk interactions such as neuropeptide-receptor signaling.

Includes:

- `Connection`: Base class representing a directed, weighted connection 
  between two `Neuron` objects. Supports custom attributes and weight updates.

- `ChemicalSynapse`: A convenience subclass of `Connection` for chemical synapses, 
  optionally storing anatomical position or other features.

- `GapJunction`: A subclass representing symmetric electrical connections between neurons.

- `BulkConnection`: Represents non-point-to-point signaling modes like neuropeptide-receptor 
  interactions or hormonal modulation.

- `ConnectionGroup`: A container for managing and operating on sets of `Connection` objects.

- `Path`: A specialized `ConnectionGroup` representing an ordered, continuous sequence of 
  connections (e.g., a feedforward chain) from a source to a target neuron.
"""

__author__ = "Sahil Moza"
__date__ = "2025-04-06"
__license__ = "MIT"

import networkx as nx
from .io import generate_random_string




class Connection:
    ''' This class represents a connection between two cells. '''
    def __init__(self, pre, post, uid=None, connection_type='chemical-synapse', **kwargs):
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