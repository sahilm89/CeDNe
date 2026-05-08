"""
Neuron and cell-level primitives for CeDNe.

This module defines the core data structures for representing cells in the nervous system.
It includes:

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

import math
import networkx as nx
import copy
from .io import generate_random_string
from typing import Optional, List, Dict, Any, Union, TYPE_CHECKING
from dataclasses import dataclass, field
import numpy as np
from .config import F_SAMPLE
from .recordings import Trial
from .connection import Path
from .source import Citable, serialize_citations

if TYPE_CHECKING:
    from .network import NervousSystem
    from .connection import Path
    from .recordings import Trial

"""Sentinel value assigned to enum-like neuron attributes
(``type``, ``category``, ``modality``, …) when the constituents of a
merged neuron disagree on that attribute. Analyses that switch on those
attributes should branch on this value explicitly rather than silently
treating a merged neuron as its first constituent's value.
"""
MERGED_TYPE = 'merged'

# Enum-like attributes whose merge follows the "all-same → keep / mixed
# → 'merged'" policy in ``NervousSystem.contract_neurons``. Adding an
# attribute here automatically:
#   * captures it in each constituent's snapshot under
#     ``Neuron.constituents[name][attr]``,
#   * applies the merge policy to the surviving neuron,
#   * mirrors the result onto the networkx node attribute dict so paths
#     that reconstruct neurons via ``create_neurons_from(data=True)``
#     propagate it.
# The list is intentionally short — only attributes that behave like
# bounded categorical labels. Numeric properties (degree, length,
# transcript counts) need their own aggregation policy and are not
# covered here.
MERGE_TRACK_ATTRS = ('type', 'category', 'modality')


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
        if not isinstance(name, str):
            raise TypeError("name must be a string")
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
    
class Neuron(Cell, Citable):
    ''' Models a biological neuron'''

    # Attributes excluded from automatic scalar introspection in to_dict()
    _SERIALIZE_SKIP = frozenset({
        'name', 'network', 'in_connections', 'out_connections',
        'trial', 'features', 'spatial_mask', '_data', 'loadings',
        'position', 'transcript', 'group_id', 'citations',
        # `constituents` is a dict (not a numeric scalar) — handled
        # explicitly in to_dict() via the merged-neuron block.
        'constituents',
    })
    def __init__(self, name: str, network: 'NervousSystem', **kwargs):
        """
        Initializes a new instance of the Neuron class.

        Args:
            name (str):
                The name of the neuron.
            network (NervousSystem):
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

        Raises:
            ValueError: If a neuron with the given name already exists in the network.
        """
        if name in network.neurons:
            raise ValueError(f"Neuron with name '{name}' already exists in the network")
        Cell.__init__(self, name, network, **kwargs)
        Citable.__init__(self)  # provides self.citations = {}
        self.network.neurons[name] = self
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
        self.trial = kwargs.pop('trial', {})
        self.features = kwargs.pop('features', {0: 'Ca_max', 1: 'Ca_area', 2: 'Ca_avg',
                         3: 'Ca_time_to_peak', 4: 'Ca_area_to_peak',
                         5: 'Ca_min', 6: 'Ca_onset', 7: 'positive_area', 8: 'positive_time'})
        
        # Loadings from dimensionality reduction (SVD/PCA/NMF)
        # e.g. {"SVD_PC1": 0.45, "SVD_PC2": -0.12, "SVD_PC3": 0.03}
        self.loadings = kwargs.pop('loadings', {})
        
        # self.presynapse = presynapse or []
        # self.postsynapse = postsynapse or {}
        self.spatial_mask = kwargs.pop('spatial_mask', None)
        #self.cable_length = kwargs.pop('cable_length', 1)

    @property
    def is_merged(self) -> bool:
        """True iff this neuron was produced by contracting other neurons.

        Backed by ``self.constituents``: a non-empty dict ⇒ merged.
        Set by ``NervousSystem.contract_neurons``; round-trips through
        graph cloning via the networkx node attribute mirror.
        """
        return bool(getattr(self, 'constituents', None))

    def _constituent_values(self, attr):
        """Sorted list of distinct, non-empty string values of `attr`
        across this neuron's constituents. Foundation for the per-attr
        derived properties below — kept as one helper so the policy
        lives in one place.

        Non-string values (``None``, ``NaN`` from upstream pandas
        loaders, numeric junk) are dropped: these enum-like attributes
        only carry meaningful information when they're string labels,
        and mixed-type sets break ``sorted`` on Python 3 anyway.
        """
        if not getattr(self, 'constituents', None):
            return []
        values = set()
        for meta in self.constituents.values():
            v = meta.get(attr)
            if isinstance(v, str) and v:
                values.add(v)
        return sorted(values)

    @property
    def constituent_types(self):
        """Sorted list of distinct source types across this neuron's
        constituents, derived on-demand from ``self.constituents``.

        Single source of truth: changing ``constituents`` automatically
        updates this list — no parallel field to keep in sync.
        Returns ``[]`` for un-merged neurons.
        """
        return self._constituent_values('type')

    @property
    def constituent_categories(self):
        """Sorted list of distinct source categories across this
        neuron's constituents. ``[]`` for un-merged neurons or for
        merged neurons whose constituents had no category set.
        """
        return self._constituent_values('category')

    @property
    def constituent_modalities(self):
        """Sorted list of distinct source modalities across this
        neuron's constituents. ``[]`` for un-merged neurons or for
        merged neurons whose constituents had no modality set.
        """
        return self._constituent_values('modality')

    def _parent_citables(self):
        """Walk citations up through containing NeuronGroups and the NervousSystem."""
        parents = []
        if self.network is not None:
            for group in self.network.groups.values():
                # O(1) membership via the name->Neuron dict each NeuronGroup maintains
                if isinstance(group, NeuronGroup) and group.neurons.get(self.name) is self:
                    parents.append(group)
            parents.append(self.network)
        return parents

    def to_dict(self) -> Dict[str, Any]:
        """Serialize neuron to a plain Python dictionary.

        Returns a dict with guaranteed keys: ``id``, ``type``, ``group_id``,
        ``degree``, ``has_recordings``.  Optionally includes ``position``,
        ``scalars`` (auto-discovered numeric attributes), and ``loadings``
        (dimensionality-reduction weights).

        Returns:
            Dict[str, Any]: A JSON-compatible dictionary representation.
        """
        d: Dict[str, Any] = {
            "id": self.name,
            "type": getattr(self, 'type', 'Unknown'),
            "group_id": getattr(self, 'group_id', 0),
            "degree": len(self.in_connections) + len(self.out_connections),
            "has_recordings": bool(self.trial),
        }

        # Position — serialize as-is (no LR string normalization)
        if hasattr(self, 'position') and self.position:
            if isinstance(self.position, dict):
                d['position'] = dict(self.position)

        # Numeric scalar attributes (dynamic introspection)
        scalars: Dict[str, Any] = {}
        for attr in vars(self):
            if attr.startswith('_') or attr in self._SERIALIZE_SKIP:
                continue
            try:
                val = getattr(self, attr)
                if isinstance(val, (int, float)) and math.isfinite(val):
                    scalars[attr] = val
                elif (isinstance(val, (list, tuple)) and val
                      and all(isinstance(v, (int, float)) for v in val)):
                    scalars[attr] = list(val)
            except Exception:
                pass
        if scalars:
            d['scalars'] = scalars

        # Loadings from dimensionality reduction
        if hasattr(self, 'loadings') and self.loadings:
            clean_loadings: Dict[str, float] = {}
            for k, v in self.loadings.items():
                try:
                    fv = float(v)
                    if math.isfinite(fv):
                        clean_loadings[k] = fv
                except (ValueError, TypeError):
                    pass
            if clean_loadings:
                d['loadings'] = clean_loadings

        # Citations attached directly to this neuron (not the inherited chain)
        if hasattr(self, 'citations') and self.citations:
            d['citations'] = serialize_citations(self.citations)

        # Merge provenance — `is_merged` is only emitted when actually
        # merged so un-merged neurons stay payload-slim. Each
        # constituent snapshot includes every MERGE_TRACK_ATTRS value
        # captured at merge time.
        if self.is_merged:
            d['is_merged'] = True
            d['constituents'] = [
                {
                    'name': meta.get('name', name),
                    **{a: meta.get(a, '') for a in MERGE_TRACK_ATTRS},
                }
                for name, meta in self.constituents.items()
            ]
            d['constituent_types'] = self.constituent_types
            cats = self.constituent_categories
            if cats:
                d['constituent_categories'] = cats
            mods = self.constituent_modalities
            if mods:
                d['constituent_modalities'] = mods

        return d

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
        # B4: Sync to node attributes so **data works in subnetwork/copy
        nx.set_node_attributes(self.network, {self: {'trial': self.trial}})
        return self.trial[trial_num]

    def load_recording(self, data, trial_num=0, sampling_rate=None, metadata=None):
        """Attach a 1D time-series recording to this neuron as a new Trial.

        Thin convenience over `add_trial()` + setting `trial.recording`; use it
        when you already have a per-neuron trace in memory (ndarray, list,
        pandas Series) and don't want to manage Trial objects by hand.

        Args:
            data: 1D array-like of samples (calcium trace, voltage, etc.).
            trial_num: Trial index (default 0). Overwrites any existing trial
                at this index for this neuron.
            sampling_rate: Sampling rate in Hz; written to ``trial.metadata``.
                Leave as None to keep the module default (F_SAMPLE).
            metadata: Extra metadata dict merged into ``trial.metadata``.

        Returns:
            Trial: The trial object holding the recording.
        """
        trial = self.add_trial(trial_num)
        trial.recording = np.asarray(data)
        if sampling_rate is not None:
            trial.metadata['sampling_rate'] = float(sampling_rate)
        if metadata:
            trial.metadata.update(metadata)
        return trial

    def remove_trial(self, trial_num):
        """
        Removes a trial from the trial dictionary.
        """
        del self.trial[trial_num]

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


class NeuronGroup(Citable):
    ''' This contains a group of neurons in the network'''
    def __init__(self, network, members=None, group_name=None) -> None:
        """
        Initializes a new instance of the NeuronGroup class.

        Parameters:
            network (NervousSystem):
                The owning nervous system. Used to resolve string members.
            members (Iterable[Union[Neuron, str]], optional):
                The members of the group. Each entry may be a ``Neuron`` instance
                or a neuron name (string) — names are resolved via
                ``network.neurons``. Defaults to an empty group.
            group_name (str, optional):
                The name of the neuron group. Auto-generated if omitted.

        Returns:
            None
        """
        Citable.__init__(self)  # provides self.citations = {}
        if group_name is None:
            self.group_name = 'Group-'+ generate_random_string(8)
        else:
            self.group_name = group_name

        if members is None:
            members = []
        else:
            resolved = []
            for m in members:
                if isinstance(m, Neuron):
                    resolved.append(m)
                elif isinstance(m, str):
                    if m not in network.neurons:
                        raise KeyError(
                            f"Neuron {m!r} not found in network {network.name!r}"
                        )
                    resolved.append(network.neurons[m])
                else:
                    raise TypeError(
                        f"NeuronGroup members must be Neuron or str, got {type(m).__name__}"
                    )
            members = resolved
        self.members = members
        self.neurons = {m.name: m for m in members}
        self.network = network
        assert self.group_name not in self.network.groups, f"Group name {self.group_name}\
            already exists in the network"
        self.network.groups.update({self.group_name: self})

    def _parent_citables(self):
        """Walk citations up to the containing NervousSystem."""
        return (self.network,) if self.network is not None else ()

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

    def add_neuron(self, neuron: 'Neuron') -> None:
        """Add a neuron to the group.
        
        Args:
            neuron: Neuron to add.
        """
        if neuron not in self.neurons:
            self.neurons[neuron.name] = neuron

    def remove_neuron(self, neuron: 'Neuron') -> None:
        """Remove a neuron from the group.
        
        Args:
            neuron: Neuron to remove.
        """
        if neuron in self.neurons:
            self.neurons.pop(neuron.name)

    def get_neurons_by_type(self, type: str) -> List['Neuron']:
        """Get all neurons of a specific type.
        
        Args:
            type: Neuron type to filter by.
            
        Returns:
            List of neurons of the specified type.
        """
        return [n for n in self.neurons.values() if n.type == type]

    def get_neurons_by_property(self, key: str, value: Any) -> List['Neuron']:
        """Get neurons with a specific property value.
        
        Args:
            key: Property name.
            value: Property value to match.
            
        Returns:
            List of neurons with matching property value.
        """
        return [n for n in self.neurons.values() if n.get_property(key) == value]

    # ---- Set operations ----

    def union(self, other: 'NeuronGroup', group_name: str = None) -> 'NeuronGroup':
        """Return a new NeuronGroup containing neurons from both groups.

        Args:
            other: Another NeuronGroup to union with.
            group_name: Optional name for the resulting group.

        Returns:
            A new NeuronGroup with the combined members.
        """
        assert isinstance(other, NeuronGroup), "Operand must be a NeuronGroup"
        assert self.network is other.network, "Both groups must belong to the same network"
        merged = {**self.neurons, **other.neurons}
        name = group_name or f"{self.group_name}_union_{other.group_name}"
        return NeuronGroup(self.network, members=list(merged.values()), group_name=name)

    def intersection(self, other: 'NeuronGroup', group_name: str = None) -> 'NeuronGroup':
        """Return a new NeuronGroup containing only neurons present in both groups.

        Args:
            other: Another NeuronGroup to intersect with.
            group_name: Optional name for the resulting group.

        Returns:
            A new NeuronGroup with the shared members.
        """
        assert isinstance(other, NeuronGroup), "Operand must be a NeuronGroup"
        assert self.network is other.network, "Both groups must belong to the same network"
        common_keys = set(self.neurons.keys()) & set(other.neurons.keys())
        members = [self.neurons[k] for k in common_keys]
        name = group_name or f"{self.group_name}_intersect_{other.group_name}"
        return NeuronGroup(self.network, members=members, group_name=name)

    def difference(self, other: 'NeuronGroup', group_name: str = None) -> 'NeuronGroup':
        """Return a new NeuronGroup containing neurons in self but not in other.

        Args:
            other: Another NeuronGroup to subtract.
            group_name: Optional name for the resulting group.

        Returns:
            A new NeuronGroup with the difference members.
        """
        assert isinstance(other, NeuronGroup), "Operand must be a NeuronGroup"
        assert self.network is other.network, "Both groups must belong to the same network"
        diff_keys = set(self.neurons.keys()) - set(other.neurons.keys())
        members = [self.neurons[k] for k in diff_keys]
        name = group_name or f"{self.group_name}_diff_{other.group_name}"
        return NeuronGroup(self.network, members=members, group_name=name)

    def __or__(self, other: 'NeuronGroup') -> 'NeuronGroup':
        """Operator ``|`` — union."""
        return self.union(other)

    def __and__(self, other: 'NeuronGroup') -> 'NeuronGroup':
        """Operator ``&`` — intersection."""
        return self.intersection(other)

    def __sub__(self, other: 'NeuronGroup') -> 'NeuronGroup':
        """Operator ``-`` — difference."""
        return self.difference(other)