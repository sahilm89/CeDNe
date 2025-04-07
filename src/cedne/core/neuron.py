"""
Neuron and cell-level primitives for CeDNe.

This module defines the core data structures for representing neurons 
and their organization within a nervous system. It includes:

- `Cell`: A base class for any biological cell modeled in the nervous system.
- `Neuron`: A subclass of `Cell` specialized for neural structures, supporting
  connectivity, trial-specific recordings, and calcium feature extraction.
- `NeuronGroup`: A container for managing sets of neurons with shared structure,
  metadata, or functional properties.

Neurons are stored within a `NervousSystem` graph, and may maintain their own 
set of incoming and outgoing `Connection` objects. Each neuron can host multiple
`Trial` objects, representing experimental recordings under different conditions.
"""

__author__ = "Sahil Moza"
__date__ = "2025-04-06"
__license__ = "MIT"

import networkx as nx
import copy
from .io import generate_random_string
from .recordings import Trial
from .connection import Path

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

    def get_connections(self, paired_neuron=None, direction='both', connection_type='all'):
        """
        Returns all connections that the neuron is involved in.

        :return: A list of connections where the neuron is present.
        :rtype: list
        """
        if connection_type == 'all':
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
        else:
            if paired_neuron is None:
                if direction == 'both':
                    return {key:value for key, value in self.in_connections.items() if value.connection_type == connection_type} | {key:value for key, value in self.out_connections.items() if value.connection_type == connection_type}
                    #return [edge for edge in self.network.edges if self in edge]
                if direction == 'in':
                    return {key:value for key, value in self.in_connections.items() if value.connection_type == connection_type}
                if direction == 'out':
                    return {key:value for key, value in self.out_connections.items() if value.connection_type == connection_type}
                raise ValueError('Direction must be either "both", "in", or "out"')

            if paired_neuron is not None:
                if direction == 'both':
                    return {key:value for key, value in self.outgoing(paired_neuron) if value.connection_type == connection_type} | {key:value for key, value in self.incoming(paired_neuron) if value.connection_type == connection_type}
                if direction == 'in':
                    return {key:value for key, value in self.incoming(paired_neuron) if value.connection_type == connection_type}
                if direction == 'out':
                    return {key:value for key, value in self.outgoing(paired_neuron) if value.connection_type == connection_type}
                raise ValueError('Direction must be either "both", "in", or "out"')


    def get_connected_neurons(self, direction='both', weight_filter = 1, connection_type='all'):
        """ Returns all connected neurons for this neuron. """
        if connection_type == 'all':
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
        else:
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
                if conn.weight>weight_filter and conn.connection_type == connection_type:
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
    
    def all_paths(self, path_length=1, direction='both'):
        '''
        Returns all paths as a list of connections from this neuron to all other neurons in the network
        '''
        if direction == 'out':
            out_paths = [nx.all_simple_edge_paths(self.network, self, self.network.neurons[n], cutoff=path_length) for n in self.network.neurons]
            connection_paths = [[[self.network.connections[edge] for edge in path] for path in paths] for paths in out_paths]
            return [Path(self.network, path, f'Path_{self.name}_out_length_{path_length}_{j}_{k}')  for k, paths in enumerate(connection_paths) for j,path in enumerate(paths)]
        elif direction == 'in':
            in_paths = [nx.all_simple_edge_paths(self.network, self.network.neurons[n], self, cutoff=path_length) for n in self.network.neurons]
            connection_paths = [[[self.network.connections[edge] for edge in path] for path in paths] for paths in in_paths]
            return [Path(self.network, path, f'Path_{self.name}_in_length_{path_length}_{j}_{k}') for k, paths in enumerate(connection_paths) for j,path in enumerate(paths)]
        elif direction=='both':
            in_paths = [nx.all_simple_edge_paths(self.network, self.network.neurons[n], self, cutoff=path_length) for n in self.network.neurons]
            out_paths = [nx.all_simple_edge_paths(self.network, self, self.network.neurons[n], cutoff=path_length) for n in self.network.neurons]
            connection_paths_out = [[[self.network.connections[edge] for edge in path] for path in paths] for paths in out_paths] 
            connection_paths_in = [[[self.network.connections[edge] for edge in path] for path in paths] for paths in in_paths]
            return [Path(self.network, path, f'Path_{self.name}_out_length_{path_length}_{j}_{k}')  for k, paths in enumerate(connection_paths_out) for j,path in enumerate(paths)] + [Path(self.network, path, f'Path_{self.name}_in_length_{path_length}_{j}_{k}') for k, paths in enumerate(connection_paths_in) for j,path in enumerate(paths)]
    
    def __str__(self):
        ## For use in debugging and testing
        return self.name

    # def __repr__(self):
    #     ## For use in debugging and testing
    #     return self.name


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