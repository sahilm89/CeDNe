"""
Loader utilities for CeDNe

Includes functions to initialize nervous systems, load connectome data, 
neurotransmitters, neuropeptides, transcriptomes, and other biological properties.
"""

__author__ = "Sahil Moza"
__date__ = "2025-04-06"
__license__ = "MIT"

from calendar import c
import warnings
import pickle
import re
from pathlib import Path
import numpy as np
import pandas as pd
import requests
from cedne import Worm, Fly, NervousSystem
from cedne.core import Neuron, Behavior, Session
from cedne.core.context import Context, ExperimentalContext
from cedne.core.animal import Animal
from cedne.core.source import Citation
from .config import *
warnings.filterwarnings("ignore", category=UserWarning, module='openpyxl')


# Canonical SIM vocabulary used throughout CeDNe for the sensory / interneuron /
# motorneuron axis. Every C. elegans loader (cook, witvliet, white_1986) and
# the comparative loaders (platynereis, ciona) normalize to these exact strings
# so downstream analyses don't need per-species mappings.
SIM_SENSORY     = 'sensory'
SIM_INTERNEURON = 'interneuron'
SIM_MOTORNEURON = 'motorneuron'

_COOK_MALE_TYPE_NORM = {
    'SENSORY NEURON':  SIM_SENSORY,    'SENSORY NEURONS':  SIM_SENSORY,
    'INTERNEURON':     SIM_INTERNEURON,'INTERNEURONS':     SIM_INTERNEURON,
    'MOTOR NEURON':    SIM_MOTORNEURON,'MOTOR NEURONS':    SIM_MOTORNEURON,
}


def _neuropal_position_citations():
    """Structured citations for the bundled NeuroPAL-derived position atlas."""
    return [
        Citation(
            key='Skuhersky2022',
            title='Toward a more accurate 3D atlas of C. elegans neurons',
            authors=['Skuhersky et al.'],
            year=2022,
            doi='10.1186/s12859-022-04738-3',
            notes='Source of CeDNe bundled C. elegans anatomical neuron positions.',
        ),
        Citation(
            key='Yemini2020',
            title='NeuroPAL: A Multicolor Atlas for Whole-Brain Neuronal Identification in C. elegans',
            authors=['Yemini et al.'],
            year=2020,
            doi='10.1016/j.cell.2020.12.012',
            notes='Original NeuroPAL atlas and neuron-identification reference.',
        ),
    ]


def _is_default_neuropal_positions_path(path):
    """True when ``path`` is the bundled NeuroPAL-derived position pickle."""
    try:
        return Path(path).resolve() == Path(neuronPositions).resolve()
    except TypeError:
        return False


def _attach_neuropal_position_citations(nn):
    """Attach NeuroPAL position provenance only to neurons with coordinates."""
    for neuron in nn.neurons.values():
        if getattr(neuron, 'position', None) is not None:
            for citation in _neuropal_position_citations():
                neuron.add_citation(citation)


def _atanas_recording_citation():
    """Structured citation for Atanas et al. whole-brain recordings."""
    return Citation(
        key='Atanas2023',
        authors=['Atanas et al.'],
        year=2023,
        doi='10.1016/j.cell.2023.07.035',
        notes='Whole-brain calcium imaging recordings loaded from atanas_whole_brain.',
    )


def _celegans_canonical_types():
    """Return {neuron_name: canonical_SIM_type} from the Cook hermaphrodite
    Cell_list reference (sensory / interneuron / motorneuron only; pharyngeal
    and other non-SIM entries are dropped).

    Used by loaders whose source data does not carry SIM labels per neuron
    (Witvliet) or uses non-canonical labels that can be backfilled by name
    (Cook male PHARYNX bucket).
    """
    with open(cell_list, 'rb') as f:
        df = pickle.load(f)
    name_col = df.columns[0]
    type_col = df.columns[1]
    canonical = {SIM_SENSORY, SIM_INTERNEURON, SIM_MOTORNEURON}
    return {
        str(row[name_col]).strip(): str(row[type_col]).strip()
        for _, row in df.iterrows()
        if str(row[type_col]).strip() in canonical
    }


def makeWorm(name='', import_parameters=None, chem_only=False, gapjn_only=False):
    ''' Utility function to make a Worm based on import parameters.'''
    if import_parameters is None or (import_parameters['style'] == 'cook' and import_parameters['sex'] == 'hermaphrodite'):
        w = Worm(name)
        w.citations.update({'cook_connectome':citations['cook_connectome']})
        nn = NervousSystem(w)
        build_nervous_system(nn, neuron_data=cell_list, \
                            chem_synapses=chemsyns, \
                            elec_synapses=elecsyns, \
                            positions=neuronPositions, \
                            chem_only=chem_only, \
                            gapjn_only=gapjn_only)
    elif (import_parameters['style'] == 'cook' and import_parameters['sex'] == 'male'):
        w = Worm(name)
        w.citations.update({'cook_connectome':citations['cook_connectome']})
        nn = NervousSystem(w)

        if cook_male_cache.exists():
            with open(cook_male_cache, 'rb') as cache_file:
                cache = pickle.load(cache_file)
            labels = cache['labels']
            ntype = cache['ntype']
            l1 = cache['l1']
            adj_chem = cache['adj_chem']
            adj_gapjn = cache['adj_gapjn']
        else:
            input_file = 'SI 5 Connectome adjacency matrices, corrected July 2020.xlsx'

            ## Chemical synapses
            cook_chem = pd.read_excel(cook_connectome / input_file, sheet_name='male chemical', engine='openpyxl')
            colnames = cook_chem.iloc[1, 3:-1].astype(str).tolist()
            labels = cook_chem.loc[2:383]['Unnamed: 2'].tolist()

            ccl = cook_chem.iloc[2:,:2].copy().ffill()

            list_1 = ccl.iloc[:,0].to_list()
            list_2 = ccl.iloc[:,1].to_list()

            ## Correcting SEX_SPECIFIC NEURONS
            ntype = {}
            l1 = {}
            for j, n in enumerate(labels):
                if list_1[j] == 'SEX SPECIFIC':
                    if not list_2[j] == 'HEAD':
                        ntype[n] = list_2[j]
                    else:
                        if n.startswith('CEM'):
                            ntype[n] = 'SENSORY NEURON'
                        elif n.startswith('MCM'):
                            ntype[n] = 'INTERNEURON'
                    l1[n] = 'SEX SPECIFIC'
                else:
                    ntype[n] = list_1[j]
                    l1[n] = list_2[j]

            labels_set = set(labels)

            cook_chem = cook_chem.drop(columns=cook_chem.columns[:3], index=cook_chem.index[:2])
            cook_chem = cook_chem.drop(columns=cook_chem.columns[-1], index=cook_chem.index[-1])
            cook_chem.reset_index(drop=True, inplace=True)
            cook_chem.columns = colnames
            cols = cook_chem.columns.to_list()
            chem_adj = cook_chem.to_numpy()
            adj_chem = {}
            for i, row in enumerate(labels):
                adj_chem[row] = {col1: {"weight": chem_adj[i, j]}
                                 for j, col1 in enumerate(cols)
                                 if col1 in labels_set and chem_adj[i, j] != 0}

            ## Gap junctions
            cook_gapjn = pd.read_excel(cook_connectome / input_file, sheet_name='male gap jn symmetric', engine='openpyxl')
            colnames = cook_gapjn.iloc[1][3:-1].astype(str).tolist()

            row_labels = cook_gapjn.loc[2:383]['Unnamed: 2'].tolist()

            cook_gapjn = cook_gapjn.drop(columns=cook_gapjn.columns[:3], index=cook_gapjn.index[:2])
            cook_gapjn = cook_gapjn.drop(columns=cook_gapjn.columns[-1], index=cook_gapjn.index[-1])
            cook_gapjn.reset_index(drop=True, inplace=True)
            cook_gapjn.columns = colnames
            cols = cook_gapjn.columns.to_list()
            gapjn_adj = cook_gapjn.to_numpy()
            adj_gapjn = {}
            for i, row in enumerate(row_labels):
                if row in labels_set:
                    adj_gapjn[row] = {col1: {"weight": gapjn_adj[i, j]}
                                      for j, col1 in enumerate(cols)
                                      if col1 in labels_set and gapjn_adj[i, j] != 0}

            with open(cook_male_cache, 'wb') as cache_file:
                pickle.dump({'labels': labels, 'ntype': ntype, 'l1': l1,
                             'adj_chem': adj_chem, 'adj_gapjn': adj_gapjn},
                            cache_file, protocol=pickle.HIGHEST_PROTOCOL)

        # Canonicalize per-cell SIM labels. Source strings are uppercase/plural
        # ('SENSORY NEURONS', 'MOTOR NEURON', ...) and a 'PHARYNX' bucket that
        # doesn't encode SIM; backfill that bucket from the Cook hermaphrodite
        # reference which tags the 20 pharyngeal neurons individually.
        cook_h_sim = _celegans_canonical_types()
        canon_type = {}
        for n, raw in ntype.items():
            canon = _COOK_MALE_TYPE_NORM.get(str(raw).strip())
            if canon is None:
                canon = cook_h_sim.get(n)
            canon_type[n] = canon
        nn.create_neurons(labels=labels, type=canon_type, category=l1)
        if not gapjn_only:
            nn.setup_chemical_connections(adj_chem)
        if not chem_only:
            nn.setup_gap_junctions(adj_gapjn)
    elif import_parameters['style'] == 'witvliet':
        ind_dict = {'L1': [1,2,3,4], 'L2':[5] , 'L3':[6], 'adult':[7,8]}
        assert import_parameters['stage'] in ['L1', 'L2', 'L3', 'adult'], "stage should be one of 'L1', 'L2', 'L3', 'adult'"
        assert int(import_parameters['dataset_ind']) in range(1,len(ind_dict[import_parameters['stage']])+1) , f"Dataset id {int(import_parameters['dataset_ind'])} for stage {import_parameters['stage']} should be in {list(range(1,len(ind_dict[import_parameters['stage']])+1))}"

        input_file = 'witvliet_2020_' + str(ind_dict[import_parameters['stage']][int(import_parameters['dataset_ind'])-1]) + ' ' + import_parameters['stage'] + '.xlsx'
        witvliet_input = pd.read_excel(witvliet_connectome / input_file, engine='openpyxl')
        all_labels = set(witvliet_input['pre']) | set(witvliet_input['post'])
        labels = [lab for lab in all_labels if not any(lab.startswith(k) for k in ['BWM-', 'CEPsh', 'GLR'])]

        w = Worm(name=name, stage=import_parameters['stage'])
        w.citations.update({'witvliet_connectome':citations['witvliet_connectome']})
        nn = NervousSystem(w, network='_'.join([import_parameters['style'], import_parameters['stage'], import_parameters['dataset_ind']]))
        # Witvliet's source does not carry per-neuron SIM labels; backfill from
        # the Cook hermaphrodite reference (same naming convention).
        cook_h_sim = _celegans_canonical_types()
        witvliet_types = {lab: cook_h_sim.get(lab) for lab in labels}
        nn.create_neurons(labels=labels, type=witvliet_types)
        witvliet_input.rename(columns={'synapses': 'weight'}, inplace=True)
        fin_input = witvliet_input[witvliet_input['pre'].isin(labels)]
        fin_input = fin_input[fin_input['post'].isin(labels)]
        for _, conn in fin_input.iterrows():
            nn.setup_connections(conn, conn['type'], input_type='edge')
    elif import_parameters['style'] == 'white_1986':
        # Historical/comparison loader. For modern analyses prefer style='cook',
        # which supersedes White 1986 with more neurons (302 vs 279; includes
        # pharynx) and updated synapse counts after re-scoring.
        #
        # The source NeuronConnect.xls encodes each synapse from both sides as
        # S/Sp (send) and R/Rp (receive). We keep S/Sp for chemical synapses
        # to avoid double-counting. EJ is symmetric and mirrored on both
        # directions to match the Cook gap-junction convention. NMJ is skipped
        # (no muscles in the network).
        w = Worm(name=name, stage='Day-1 Adult', sex='Hermaphrodite', genotype='N2')
        w.citations.update({'white_connectome': citations['white_connectome']})
        nn = NervousSystem(w, network='white_1986')

        nt = pd.read_excel(white_connectome / 'NeuronType.xls')
        nt.columns = [c.strip() for c in nt.columns]
        soma_position = {str(r['Neuron']).strip(): float(r['Soma Position'])
                         for _, r in nt.iterrows() if pd.notna(r['Soma Position'])}
        soma_region = {str(r['Neuron']).strip(): str(r['Soma Region']).strip()
                       for _, r in nt.iterrows() if pd.notna(r['Soma Region'])}
        labels = sorted(soma_position)
        # NeuronType.xls has no SIM column; backfill from the Cook hermaphrodite
        # reference (same neuron naming convention).
        cook_h_sim = _celegans_canonical_types()
        white_types = {lab: cook_h_sim.get(lab) for lab in labels}
        nn.create_neurons(labels, type=white_types,
                          soma_position=soma_position, soma_region=soma_region)

        # Source has lowercase 'avfl'/'avfr' in place of AVFL/AVFR; normalize.
        name_fixups = {'avfl': 'AVFL', 'avfr': 'AVFR'}
        def _canon(x):
            s = str(x).strip()
            return name_fixups.get(s, s)

        edges = pd.read_excel(white_connectome / 'NeuronConnect.xls')
        chem_adj = {}
        for _, row in edges[edges['Type'].isin(['S', 'Sp'])].iterrows():
            pre, post = _canon(row['Neuron 1']), _canon(row['Neuron 2'])
            if pre not in nn.neurons or post not in nn.neurons:
                continue
            chem_adj.setdefault(pre, {}).setdefault(post, {'weight': 0})
            chem_adj[pre][post]['weight'] += int(row['Nbr'])
        nn.setup_chemical_connections(chem_adj)

        gap_adj = {}
        for _, row in edges[edges['Type'] == 'EJ'].iterrows():
            a, b = _canon(row['Neuron 1']), _canon(row['Neuron 2'])
            if a not in nn.neurons or b not in nn.neurons:
                continue
            w_ij = int(row['Nbr'])
            gap_adj.setdefault(a, {})[b] = {'weight': w_ij}
            gap_adj.setdefault(b, {})[a] = {'weight': w_ij}
        nn.setup_gap_junctions(gap_adj)
    else:
        raise ValueError("Unsupported connectome style")
    return w

def makeFly(name = '', import_parameters=None):
    if import_parameters is not None and import_parameters['style'] == 'fly_wire':
        f = Fly(name)
        f.citations.update({'fly_wire': citations['fly_wire']})
        nn = NervousSystem(f)

        ## Neurons

        ### Names
        names = pd.read_csv(fly_wire / 'names.csv')
        labs, neuron_types, lab_root_id = names['name'], names['group'], names['root_id']
        neuron_dict = {r:lab for r,lab in zip(lab_root_id, labs)}
        type_dict = {r:ntype for r,ntype in zip(lab_root_id, neuron_types)} 
        
        root_ids = sorted(lab_root_id)
        labels = [neuron_dict[rid] for rid in root_ids]
        neuron_types = {neuron_dict[rid]:type_dict[rid] for rid in root_ids}
        
        ### Positions
        coordinates = pd.read_csv(fly_wire / 'coordinates.csv')
        pos_root_id, position = coordinates['root_id'], coordinates['position']
        position_dict = {neuron_dict[rid]:np.array(list(filter(None, pos.split('[')[-1].split(']')[0].split(' '))), dtype=int) for rid,pos in zip(pos_root_id, position)}
        
        ### Stats
        stats = pd.read_csv(fly_wire / 'cell_stats.csv')
        stats_root_id, nlength, narea, nvolume = stats['root_id'], np.array(stats['length_nm'], dtype=int), np.array(stats['area_nm'], dtype=int), np.array(stats['size_nm'], dtype=int)

        length_dict = {neuron_dict[rid]:nlen for (rid,nlen) in zip(stats_root_id, nlength)}
        area_dict = {neuron_dict[rid]:nare for (rid,nare) in zip(stats_root_id, narea)}
        vol_dict = {neuron_dict[rid]:nvol for (rid,nvol) in zip(stats_root_id, nvolume)}

        nn.create_neurons(labels, type=neuron_types, position=position_dict, length=length_dict, area=area_dict, volume=vol_dict)

        ## Connections
        conns = pd.read_csv(fly_wire / 'connections_no_threshold.csv')
        pre_rid, post_rid, weights, nts = conns['pre_root_id'], conns['post_root_id'], conns['syn_count'], conns['nt_type']

        for pre, post, weight, nt in zip(pre_rid, post_rid, weights, nts ):
            adjacency = {'pre':neuron_dict[pre], 'post':neuron_dict[post], 'weight':weight}
            neurotransmitter = {'neurotransmitter':nt}
            nn.setup_connections(adjacency, connection_type='chemical-synapse', input_type='edge', neurotransmitter=neurotransmitter)
    
    elif import_parameters['style'] == 'Winding_2023':
        f = Fly(name, stage='Larva Instar-1')
        f.citations.update({'winding_connectome': citations['winding_connectome']})
        nn = NervousSystem(f)

        names = pd.read_csv(winding_connectome/ 'annotations.csv')
        base_neuron_names = names['left_id'].tolist() + names['right_id'].tolist()
        base_neuron_type = names['celltype'].tolist() + names['celltype'].tolist()

        numNeurons = len(base_neuron_names)
        neuron_types = {}
        for j in range(numNeurons):
            if base_neuron_names[j] != 'no pair':
                neuron_types[str(base_neuron_names[j])] = str(base_neuron_type[j])

        conns = pd.read_csv(winding_connectome/ 'all-all_connectivity_matrix.csv', index_col= 0)

        neuron_names = [str(n) for n in conns.index]
        nn.create_neurons(neuron_names, neuron_type=[neuron_types[nname] if nname in neuron_types else 'unannotated' for nname in neuron_names])
        
        for pre in conns.index:
            for post in conns.columns:
                weight = conns.loc[pre, post]
                if weight > 0:
                    adjacency = {'pre': str(pre), 'post': str(post), 'weight': float(weight)}
                    nn.setup_connections(adjacency, connection_type='chemical-synapse', input_type='edge')

    return f

def make_ciona():
    a = Animal(name='Ciona intestinalis', species='Ciona intestinalis', common_name='sea squirt', phylum='Chordata', clade='Tunicata')
    a.citations.update({'ciona_connectome': citations['ciona_connectome']})
    nn = NervousSystem(a)
    
    # Load core data
    names_df = pd.read_csv(ciona_connectome / 'nodes.csv')
    names_df.columns = names_df.columns.str.strip().str.lstrip('#').str.strip()
    conns = pd.read_csv(ciona_connectome / 'edges.csv')
    conns.columns = conns.columns.str.strip().str.lstrip('#').str.strip()

    # Load enrichment data
    fig1_xl = pd.read_excel(ciona_connectome / 'elife-16962-fig1-data1-v1.xlsx', sheet_name='Sheet1').ffill()
    fig3_xl = pd.read_excel(ciona_connectome / 'elife-16962-fig3-data1-v1.xlsx', sheet_name='Sheet2')
    
    # Coordinates mapping from Fig 3 (3D)
    pos3d_dict = {}
    for _, row in fig3_xl.iterrows():
        try:
            cell_id = str(row['Cell'])
            pos3d_dict[cell_id] = np.array([float(row['X']), float(row['Y']), float(row['Z'])])
        except:
            continue

    # Biological mapping from Fig 1
    bio_mapping = {}
    for _, row in fig1_xl.iterrows():
        ids_str = str(row['Cell IDs'])
        # Handle ranges like pr1-pr23
        range_match = re.search(r'([a-zA-Z]+)(\d+)-(\1|)(\d+)', ids_str)
        if range_match:
            prefix, start, _, end = range_match.groups()
            for i in range(int(start), int(end) + 1):
                bio_mapping[f'{prefix}{i}'] = {'cell_type': row['Cell Type'], 'annotation': row['Annotation']}
        
        # Handle individual IDs and aliases in parentheses
        # e.g. "ACIN1L (109*), ACIN2L (tail7)"
        items = re.findall(r'([a-zA-Z0-9]+(?:\*[^*]*)?)(?:\s*\(([^)]+)\))?', ids_str)
        for name, alias in items:
            info = {'cell_type': row['Cell Type'], 'annotation': row['Annotation']}
            bio_mapping[name] = info
            if alias:
                # remove * and other symbols
                clean_alias = alias.replace('*', '').strip()
                bio_mapping[clean_alias] = info
                # handle 'tail' -> 'midtail' for nodes.csv compatibility
                if clean_alias.startswith('tail'):
                    bio_mapping[clean_alias.replace('tail', 'midtail')] = info

    # Build final node mapping
    neuron_dict = names_df.set_index('index')['name'].to_dict()
    
    # Only keep anatomical coordinates from Fig 3. The nodes.csv `_pos` values are
    # a compact layout, not anatomical coordinates, so using them here produces a
    # misleading mixed view.
    pos_dict = {}
    for _, row in names_df.iterrows():
        idx = str(row['index'])
        name = str(row['name'])
        if idx in pos3d_dict:
            pos_dict[name] = pos3d_dict[idx]
        elif name in pos3d_dict:
            pos_dict[name] = pos3d_dict[name]

    node_colors = {row['name']: "#" + row['color'][-6:] for _, row in names_df.iterrows()}
    
    neuron_indices = sorted(names_df['index'].tolist())    
    
    # Map fig1 'Annotation' column to canonical SIM vocabulary. Annotation
    # values that don't map cleanly ('Sensory/Interneuron', 'Accessory',
    # 'Ambiguous') leave `type` unset so downstream SIM-axis analyses exclude
    # those cells rather than miscount them.
    ciona_annot_norm = {
        'Sensory':             SIM_SENSORY,
        'Interneuron':         SIM_INTERNEURON,
        'Ciliated Interneuron': SIM_INTERNEURON,
        'Motor neuron':        SIM_MOTORNEURON,
    }

    # Enrichment attributes
    cell_type_dict = {}
    annot_dict = {}
    type_dict = {}
    for nidx in neuron_indices:
        name = neuron_dict[nidx]
        info = bio_mapping.get(name) or bio_mapping.get(str(nidx))
        if info:
            cell_type_dict[name] = info['cell_type']
            annot_dict[name] = info['annotation']
            type_dict[name] = ciona_annot_norm.get(str(info['annotation']).strip())
        else:
            cell_type_dict[name] = 'Other'
            annot_dict[name] = 'Unknown'
            type_dict[name] = None

    nn.create_neurons([neuron_dict[n] for n in neuron_indices],
                      position=pos_dict,
                      color=node_colors,
                      type=type_dict,
                      cell_type=cell_type_dict,
                      annotation=annot_dict)

    ## Connections
    for _, row in conns.iterrows():
        adjacency = {'pre': neuron_dict[row['source']], 'post': neuron_dict[row['target']], 'weight': row['depth']}
        nn.setup_connections(adjacency, connection_type='chemical-synapse', input_type='edge')

    return a

def make_pristionchus(name='', dataset_ind=1):
    """Load the Pristionchus pacificus pharyngeal connectome (Bumbarger et al. 2013).

    Uses the paper's specimen-specific pharyngeal adjacency table.

        dataset_ind=1  -> specimen 107 (default)
        dataset_ind=2  -> specimen 148

    The paper does not assign SIM (sensory / interneuron / motorneuron)
    classes to pharyngeal cells, so neuron `type` is left unset.

    Args:
        name (str): Animal name. Auto-generated if empty.
        dataset_ind (int): 1 (specimen 107, default) or 2 (specimen 148).

    Returns:
        Animal: A Pristionchus pacificus Animal with a single 'pharynx'
        NervousSystem containing chemical synapses.
    """
    specimen_map = {
        1: ('107', 'weight 107'),
        2: ('148', 'weight 148'),
    }
    assert dataset_ind in specimen_map, \
        f"dataset_ind must be one of {list(specimen_map)}; got {dataset_ind!r}"
    specimen, weight_col = specimen_map[dataset_ind]

    df = pd.read_excel(pristionchus_pharynx / 'mmc2.xlsx', sheet_name='Sheet1',
                       header=1, engine='openpyxl')
    adjacency = {}
    if {'presynaptic', 'postsynaptic', weight_col}.issubset(df.columns):
        df = df.dropna(subset=['presynaptic', 'postsynaptic']).copy()
        for _, row in df.iterrows():
            pre = str(row['presynaptic']).strip()
            post = str(row['postsynaptic']).strip()
            if pre in ('nan', '') or post in ('nan', ''):
                continue
            adjacency.setdefault(pre, {})[post] = {'weight': float(row[weight_col])}
    else:
        col_offset = 0 if dataset_ind == 1 else 3
        pre_col = col_offset + 1
        post_col = col_offset + 2
        edge_counts = {}
        for _, row in df.iterrows():
            pre = row.get(pre_col)
            post = row.get(post_col)
            if pd.isna(pre) or pd.isna(post):
                continue
            pre = str(pre).strip()
            post = str(post).strip()
            if not pre or not post:
                continue
            edge_counts[(pre, post)] = edge_counts.get((pre, post), 0) + 1
        for (pre, post), weight in edge_counts.items():
            adjacency.setdefault(pre, {})[post] = {'weight': float(weight)}

    cells = set(adjacency.keys())
    for partners in adjacency.values():
        cells.update(partners.keys())
    cells = sorted(cells)

    if not name:
        name = f'Pristionchus-consensus-{weight_col.replace(" ", "_")}'
    a = Animal(species='Pristionchus pacificus', name=name, stage='Adult',
               specimen=specimen, weight_source=weight_col)
    a.citations.update({'bumbarger_pharynx': citations['bumbarger_pharynx']})
    nn = NervousSystem(a, network='pharynx')
    nn.create_neurons(cells)
    nn.setup_chemical_connections(adjacency)
    return a


def make_platynereis(name=''):
    """Load the Platynereis dumerilii whole-body connectome (Verásztó 2025).

    The 3-day-old larva (~72 hpf nectochaete) whole-body synaptic connectome,
    covering 202 neuronal cell types plus 92 non-neuronal effector types
    (muscles, glands, ciliated / follicle / pigment cells).

    Two `NervousSystem` networks are attached to the returned Animal:

        - 'neurons'    -- fine-grained per-cell network from the 1720-row
                          adjacency matrix, minus unassigned 'fragment' /
                          'SHORTfrg' / 'SHORTfragment' placeholder rows
                          (1624 cells, ~7.5k chemical connections).
        - 'celltypes'  -- grouped cell-type network (294 nodes: 202 + 92,
                          ~2.5k connections).

    Every node carries CeDNe-standard attributes (`type`, `category`,
    `modality`, `neurotransmitter`, `soma_position`, `region`) plus
    domain-specific extras (`is_neuron`, `n_cells`, `cell_kind`,
    `body_segments`, `region_tags`, `projection_tags`, `nt_tags`, `markers`,
    `tags`). `category` is SN/IN/MN for the 202 neuronal types (from
    elife-97964-fig3-data1.txt); `cell_kind` on effectors is the paper-
    authoritative kind from elife-97964-fig3-data2-v1.txt.

    For orphan cells in the fine network (no named celltype in either
    reference), `category` is still filled from the per-neuron SN/IN/MN
    class in elife-97964-fig1-data1.txt.

    Args:
        name (str): Animal name. Defaults to 'Platynereis'.

    Returns:
        Animal: A Platynereis dumerilii Animal with 'neurons' and
        'celltypes' networks attached.
    """
    import itertools

    src = veraszto_connectome

    celltypes    = pd.read_csv(src / 'neuronal_celltypes_table.csv')
    fig1         = pd.read_csv(src / 'elife-97964-fig1-data1.txt', sep='\t')
    fig3         = pd.read_csv(src / 'elife-97964-fig3-data1.txt', sep='\t')
    fig3_nonneu  = pd.read_csv(src / 'elife-97964-fig3-data2-v1.txt', sep='\t')
    annot_matrix = pd.read_csv(src / 'elife-97964-fig3-figsupp2-data1.txt',
                               sep='\t', index_col=0)
    full_adj     = pd.read_csv(src / 'full_connectome_adjacency_matrix.csv', index_col=0)
    grouped_adj  = pd.read_csv(src / 'all_celltypes_synapse_matrix.csv',
                               sep=';', index_col=0)

    # Drop unassigned EM fragments from the fine-grained adjacency. The source
    # has rows labeled 'fragment', 'SHORTfrg', 'SHORTFRG *', 'SHORTfragment *',
    # and 'fragment PDF *' (short-fragment / unassigned neurites that could not
    # be matched to a named cell). These carry no metadata and their
    # connectivity is reconstruction noise.
    frag_pat = re.compile(r'(^|\s)fragment(\s|$)|SHORTfrg|SHORTFRG|SHORTfragment',
                          re.IGNORECASE)
    frag_mask = full_adj.index.astype(str).map(lambda s: bool(frag_pat.search(s)))
    full_adj = full_adj.loc[~frag_mask]
    frag_cols = [c for c in full_adj.columns if frag_pat.search(str(c))]
    if frag_cols:
        full_adj = full_adj.drop(columns=frag_cols)

    def _clean(x):
        if pd.isna(x):
            return None
        return ' '.join(str(x).split())

    # fig3 -> SN/IN/MN class per neuronal celltype (CeDNe canonical `type`).
    category_norm = {
        'Sensory neuron': SIM_SENSORY,
        'interneuron':    SIM_INTERNEURON,
        'motorneuron':    SIM_MOTORNEURON,
    }
    fig3_sim = {
        _clean(ct_name): category_norm.get(_clean(sub['SN_MN_IN'].iloc[0]))
        for ct_name, sub in fig3.groupby('celltype')
    }

    # fig3_nonneu -> authoritative effector kind for the 92 non-neuronal types.
    # One alias: fig3_nonneu has 'meso'; grouped_adj has 'meso 1592020'.
    grouped_names = set(grouped_adj.index)
    nonneu_aliases = {'meso': 'meso 1592020'}
    fig3_nonneu_kind = {}
    for ct_name, sub in fig3_nonneu.groupby('celltype'):
        nm = nonneu_aliases.get(_clean(ct_name), _clean(ct_name))
        fig3_nonneu_kind[nm] = _clean(sub['ANNOT'].iloc[0])

    # celltypes table -> text metadata (incomplete coverage).
    ct_meta = {}
    for _, row in celltypes.iterrows():
        nm = _clean(row['name of cell type'])
        if not nm:
            continue
        ct_meta[nm] = {
            'soma_position':    _clean(row['soma position']),
            'region':           _clean(row['region']),
            'neurotransmitter': _clean(row['transmitter phenotype']),
            'n_cells':          int(row['number of cells']) if pd.notna(row['number of cells']) else None,
            'cell_kind':        _clean(row['Sensory/inter/motor neuron']),
        }

    # annot_matrix -> resolve column names to grouped_adj celltype names.
    # R mangles '-' and '/' to '.' in column names; try all combinations.
    def _resolve_annot_col(col, names):
        if col in names:
            return col
        dot_pos = [i for i, ch in enumerate(col) if ch == '.']
        if not dot_pos:
            return None
        for subs in itertools.product(['-', '/', '.'], repeat=len(dot_pos)):
            arr = list(col)
            for p, ch in zip(dot_pos, subs):
                arr[p] = ch
            cand = ''.join(arr)
            if cand in names:
                return cand
        return None

    annot_col_to_ct = {c: _resolve_annot_col(c, grouped_names) for c in annot_matrix.columns}

    BODY_SEGMENTS = {'ectoderm', 'episphere', 'segment_0', 'segment_1', 'segment_2',
                     'segment_3', 'torso', 'pygidium', 'neurosecretory_plexus'}
    REGION_TAGS   = {'mushroom body', 'Apical_organ', 'Dorsal_sensory_organ',
                     'Dorsolateral_sense_organs', 'antenna', 'eyespot',
                     'mechanosensory_girdle', 'nuchal organ'}
    MODALITY_TAGS = {'rhabdomeric photoreceptor', 'ciliary photoreceptor', 'sensory_cilia',
                     'Uniciliated_penetrating_cell', 'Biciliated_penetrating_cell',
                     'Uniciliated_nonpenetrating_cell', 'Biciliated_nonpenetrating_cell',
                     'Multiciliated_penetrating_cell'}
    PROJECTION_TAGS = {'asymmetric neuron', 'biaxonal', 'decussating', 'commissural',
                       'ipsilateral', 'contralateral', 'pseudounipolar', 'descending',
                       'ascending', 'global_reach', 'head_trunk'}
    NT_TAGS       = {'glutamatergic', 'serotonergic', 'cholinergic', 'adrenergic', 'dopaminergic'}
    MARKER_TAGS   = {'dense cored vesicles', 'siGOLD'}

    annot_per_ct = {}
    for col, ct in annot_col_to_ct.items():
        if not ct:
            continue
        active = [feat for feat, v in annot_matrix[col].items() if v == 1]
        annot_per_ct[ct] = {
            'body_segments':   [f for f in active if f in BODY_SEGMENTS],
            'region_tags':     [f for f in active if f in REGION_TAGS],
            'modality':        next((f for f in active if f in MODALITY_TAGS), None),
            'projection_tags': [f for f in active if f in PROJECTION_TAGS],
            'nt_tags':         [f for f in active if f in NT_TAGS],
            'markers':         [f for f in active if f in MARKER_TAGS],
            'tags':            active,
        }

    # Merge all sources into one record per celltype.
    celltype_attrs = {}
    for nm in set(fig3_sim) | set(fig3_nonneu_kind) | set(ct_meta) | grouped_names:
        meta = ct_meta.get(nm, {})
        ann  = annot_per_ct.get(nm, {})
        is_neuron = nm in fig3_sim
        cell_kind = fig3_nonneu_kind.get(nm) if not is_neuron else None
        if cell_kind is None:
            cell_kind = meta.get('cell_kind')
        celltype_attrs[nm] = {
            'type':             nm if is_neuron else None,
            'category':         fig3_sim.get(nm) if is_neuron else None,
            'cell_type':        nm,
            'modality':         ann.get('modality'),
            'neurotransmitter': meta.get('neurotransmitter'),
            'soma_position':    meta.get('soma_position'),
            'region':           meta.get('region'),
            'is_neuron':        is_neuron,
            'n_cells':          meta.get('n_cells'),
            'cell_kind':        cell_kind,
            'body_segments':    ann.get('body_segments', []),
            'region_tags':      ann.get('region_tags', []),
            'projection_tags':  ann.get('projection_tags', []),
            'nt_tags':          ann.get('nt_tags', []),
            'markers':          ann.get('markers', []),
            'tags':             ann.get('tags', []),
        }

    # fig1 -> per-neuron SN/IN/MN class (closes `type` for orphan cells).
    fig1_norm = {'SN': SIM_SENSORY, 'IN': SIM_INTERNEURON, 'MN': SIM_MOTORNEURON}
    fig1_sim = {n: fig1_norm.get(c)
                for n, c in zip(fig1['neuron'], fig1['neuron_type'])}

    def _neuron_to_celltype(n):
        base = n.split('_')[0]
        return base if base in celltype_attrs else None

    # Build the Animal and both networks.
    if not name:
        name = 'Platynereis'
    a = Animal(species='Platynereis dumerilii', name=name, stage='3-day larva',
               common_name='ragworm', phylum='Annelida')
    a.citations.update({'veraszto_connectome': citations['veraszto_connectome']})

    std_keys = ('type', 'category', 'cell_type', 'modality', 'neurotransmitter',
                'soma_position', 'region', 'is_neuron')
    extra_keys = ('n_cells', 'cell_kind', 'body_segments', 'region_tags',
                  'projection_tags', 'nt_tags', 'markers', 'tags')
    all_keys = std_keys + extra_keys

    # Fine-grained (per-cell) network.
    nn_fine = NervousSystem(a, network='neurons')
    full_cells = list(full_adj.index)
    attr_by_neuron = {k: {} for k in all_keys}
    for n in full_cells:
        ct = _neuron_to_celltype(n)
        meta = celltype_attrs.get(ct, {}) if ct else {}
        for k in all_keys:
            attr_by_neuron[k][n] = meta.get(k)
        attr_by_neuron['cell_type'][n] = ct
        if attr_by_neuron['category'][n] is None and n in fig1_sim:
            attr_by_neuron['category'][n] = fig1_sim[n]
    nn_fine.create_neurons(full_cells, **attr_by_neuron)

    adj_dict = {}
    for pre in full_adj.index:
        posts = full_adj.loc[pre]
        posts = posts[posts > 0]
        if len(posts):
            adj_dict[pre] = {post: {'weight': float(w)} for post, w in posts.items()}
    nn_fine.setup_connections(adj_dict, connection_type='chemical-synapse',
                              input_type='adjacency')

    # Grouped (celltype-level) network.
    nn_ct = NervousSystem(a, network='celltypes')
    ct_labels = list(grouped_adj.index)
    ct_kwargs = {k: {} for k in all_keys}
    for ct in ct_labels:
        meta = celltype_attrs.get(ct, {})
        for k in all_keys:
            ct_kwargs[k][ct] = meta.get(k)
        # grouped_adj entries not in fig3 are non-neuronal per the paper's 202+92 split
        if ct_kwargs['is_neuron'][ct] is None:
            ct_kwargs['is_neuron'][ct] = False
    nn_ct.create_neurons(ct_labels, **ct_kwargs)

    ct_adj = {}
    label_set = set(ct_labels)
    for pre in grouped_adj.index:
        posts = grouped_adj.loc[pre]
        posts = posts[posts > 0]
        if len(posts):
            ct_adj[pre] = {post: {'weight': float(w)}
                           for post, w in posts.items() if post in label_set}
    nn_ct.setup_connections(ct_adj, connection_type='chemical-synapse',
                            input_type='adjacency')

    return a


def load_contactome(nn, stage='adult', matrix_path=None):
    """Layer Brittin 2018 nerve-ring contactome edges onto an existing NervousSystem.

    Contact area (nm^2) is symmetric: we mirror every (a, b) pair to both
    (a, b) and (b, a) with `connection_type='contact'`, matching the
    gap-junction convention. Only neurons already present in `nn` are used;
    the number of cells skipped is returned alongside the edge count.

    Parameters
    ----------
    nn : NervousSystem
        Target network. Typically a nerve-ring subnetwork of a Cook adult
        hermaphrodite for `stage='adult'`, or a bare L4 NervousSystem whose
        neurons were just created from the Brittin label set for
        `stage='L4'`.
    stage : {'adult', 'L4'}
        Selects the sheet inside the Brittin source workbook.
    matrix_path : str or Path, optional
        Override for the default Brittin spreadsheet location.

    Returns
    -------
    (added, skipped) : tuple[int, int]
        Directed edges added (each undirected pair counted twice) and the
        number of matrix cells not present in `nn`.
    """
    src = matrix_path or (brittin_contactome /
                          'Adult and L4 nerve ring neighbors.xlsx')
    sheet = {'adult': 'adult nerve ring neighbors',
             'L4':    'L4 nerve ring neighbors'}[stage]

    mat = pd.read_excel(src, sheet_name=sheet, index_col=0)
    mat = mat.loc[:, ~mat.columns.astype(str).str.startswith('Unnamed')]
    mat = mat.loc[~mat.index.astype(str).isin(['nan'])]
    mat = mat.loc[~mat.index.astype(str).str.startswith('Unnamed')]
    mat = mat.dropna(how='all', axis=0).dropna(how='all', axis=1)
    cells = sorted(set(mat.index) | set(mat.columns))
    mat = mat.reindex(index=cells, columns=cells)

    adj = {}
    for i, a in enumerate(cells):
        for b in cells[i + 1:]:
            v1 = mat.at[a, b]
            v2 = mat.at[b, a]
            vs = [v for v in (v1, v2) if pd.notna(v) and v > 0]
            if not vs:
                continue
            w = float(max(vs))
            adj.setdefault(a, {})[b] = {'weight': w}
            adj.setdefault(b, {})[a] = {'weight': w}

    present = set(nn.neurons)
    filtered = {p: {q: v for q, v in qs.items() if q in present}
                for p, qs in adj.items() if p in present}
    skipped = len(set(adj) - present)

    nn.setup_connections(filtered, connection_type='contact',
                         input_type='adjacency')
    nn.set_property('weight_units', 'nm^2')

    added = sum(len(v) for v in filtered.values())
    nn.worm.citations.update({'brittin_contactome':
                              citations['brittin_contactome']})
    return added, skipped


def build_nervous_system(nn, neuron_data, chem_synapses, elec_synapses, positions, chem_only=False, gapjn_only=False):
        """
        Builds the hermaphrodite nervous system by loading pickle files containing neuron data, chemical synapses,
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
            #meaning, lineage, description = neuron_info.iloc[:,4].to_list(), neuron_info.iloc[:,5].to_list(), neuron_info.iloc[:,6].to_list()
            nn.create_neurons(labels, type=neuron_types, category=categories, modality=modalities, position=locations) #meaning=meaning, lineage=lineage, description=description)
            if _is_default_neuropal_positions_path(positions):
                _attach_neuropal_position_citations(nn)
            assert not all([gapjn_only, chem_only]), "Select at most one of gapjn_only or chem_only attributes to be True."
            if not gapjn_only:
                nn.setup_chemical_connections(chem_adjacency)
            if not chem_only:
                nn.setup_gap_junctions(elec_adjacency)

def load_lineage(neural_network, sex='Hermaphrodite'):
    lineage_meaning_description = pd.read_excel(lineage, sheet_name=sex, engine='openpyxl')
    return(lineage_meaning_description)
## Neurotransmitter tables
suffixes = ['', 'D', 'V', 'L', 'R', 'DL', 'DR', 'VL', 'VR', '01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13']
present = False

# Canonical NT names used throughout CeDNe payloads. Source tables use
# abbreviations (Ach, Glu) that the rest of the system does not recognize.
NT_CANONICAL = {
    'Ach': 'Acetylcholine',
    'ACh': 'Acetylcholine',
    'Glu': 'Glutamate',
}

def canonicalizeNT(name):
    if not isinstance(name, str):
        return name
    return NT_CANONICAL.get(name.strip(), name.strip())

def _readLigandTable(sex='Hermaphrodite'):
    lig_file = DOWNLOAD_DIR / prefix_NT / 'ligand-table.xlsx'
    if sex in ['Hermaphrodite', 'hermaphrodite']:
        return pd.read_excel(lig_file, sheet_name='Hermaphrodite, sorted by neuron', skiprows=7, engine='openpyxl')
    if sex in ['Male', 'male']:
        return pd.read_excel(lig_file, sheet_name='Male neurons, sorted by neuron', skiprows=7, engine='openpyxl')
    raise ValueError("Sex must be 'Hermaphrodite' or 'Male'")

def getLigands(neuron, sex='Hermaphrodite', ligtable=None):
    ''' Returns ligand for each neuron'''
    if ligtable is None:
        ligtable = _readLigandTable(sex=sex)

    a,b = ligtable['Neurotransmitter 1'][ligtable['Neuron']==neuron].to_list(), ligtable['Neurotransmitter 2'][ligtable['Neuron']==neuron].to_list()

    # Only string entries are real transmitter names; NaN / empty cells mean
    # "no known transmitter for this neuron" and must not leak into downstream
    # edge metadata as a NaN float.
    out = []
    if len(a) and isinstance(a[0], str) and a[0].strip():
        out.append(canonicalizeNT(a[0]))
    if len(b) and isinstance(b[0], str) and b[0].strip():
        out.append(canonicalizeNT(b[0]))
    return out

def getLigandsAndReceptors(npr, ligmap, col):
    ''' Returns ligand and receptor for each neuron'''
    receptor_ligand = {}
    #print(npr[col])
    i = npr[col][npr[col]].index
    #print(i)
    rec = npr['gene_name'][i].to_list()
    for r in rec:
        ligands = ligmap['ligand'][ligmap['gene'] == r].to_list()
        if len(ligands)>0:
            receptor_ligand.update ({r: canonicalizeNT(ligands[0])})
        else:
            receptor_ligand.update ({r: ''})
    return receptor_ligand


def loadNeurotransmitters(nn, sex='Hermaphrodite'):
    ''' Loads Neurotransmitters into neurons using Wang et al 2024'''

    npr_file = DOWNLOAD_DIR / prefix_NT / 'GenesExpressing-BATCH-thrs4_use.xlsx'
    npr = pd.read_excel(npr_file, sheet_name='npr', true_values='TRUE', false_values='FALSE', engine='openpyxl')
    ligmap = pd.read_excel(npr_file, sheet_name='ligmap', engine='openpyxl')
    try:
        ligtable = _readLigandTable(sex=sex)
    except (FileNotFoundError, StopIteration):
        ligtable = None

    for n in nn.neurons:
        neuron = nn.neurons[n]
        if not hasattr(neuron, '_preSynapse'):
            neuron.set_property('_preSynapse', [])
        if not hasattr(neuron, '_postSynapse'):
            neuron.set_property('_postSynapse', {})

    for col in npr.columns:
        for suffix in suffixes:
            if col + suffix in nn.neurons:
                neuron = nn.neurons[col + suffix]
                merged_post = dict(neuron._postSynapse)
                merged_post.update(getLigandsAndReceptors(npr, ligmap, col))
                neuron.set_property('_postSynapse', merged_post)
    for n in nn.neurons:
        neuron = nn.neurons[n]
        if ligtable is None:
            ligands = getLigands(n, sex=sex)
        else:
            ligands = getLigands(n, sex=sex, ligtable=ligtable)
        neuron.set_property('_preSynapse', list(neuron._preSynapse) + ligands)

    for e,conn in nn.connections.items():
        if e[0].name in nn.neurons and e[1].name in nn.neurons and conn.connection_type == 'chemical-synapse':
            conn.set_property('ligands', nn.neurons[e[0].name]._preSynapse)
            conn.set_property('receptors', nn.neurons[e[1].name]._postSynapse)
            conn.set_property('putative_neurotrasmitter_receptors', [])
            for rec, lig in conn.receptors.items():
                if lig in conn.ligands:
                    conn.putative_neurotrasmitter_receptors.append((lig, rec))

            # Authoritative per-edge neurotransmitter set: receptor-matched
            # ligands (deduped, canonical names). Falls back to the full
            # presynaptic ligand list when the postsynaptic neuron has no
            # receptor data, so we don't erase transmitters just because the
            # NPR table is sparse.
            matched = []
            seen = set()
            for lig, _rec in conn.putative_neurotrasmitter_receptors:
                if lig and lig not in seen:
                    matched.append(lig)
                    seen.add(lig)
            if not matched:
                for lig in conn.ligands:
                    if lig and lig not in seen:
                        matched.append(lig)
                        seen.add(lig)
            conn.set_property('neurotransmitters', matched)
    nn.worm.citations.update({'neurotransmitter_atlas':citations['neurotransmitter_atlas']})

## Neuropeptides tables

NEUROPEPTIDE_TABLE_MODES = ("old", "new")
NEUROPEPTIDE_RANGE_MODELS = {
    "short": ("Individual NPP-GPCR networks SR", "short"),
    "mid": ("Individual NPP-GPCR networks MR", "mid"),
    "long": ("Individual NPP-GPCR networks LR", "long"),
}
NEUROPEPTIDE_NEW_PAIRS = "neuropeptide_pairs (network identities for Individual_net folders).csv"


def _normalize_neuropeptide_mode(mode):
    mode = (mode or "old").lower()
    if mode not in NEUROPEPTIDE_TABLE_MODES:
        raise ValueError(f"Unknown neuropeptide table mode: {mode}")
    return mode


def _normalize_neuropeptide_range(range_model):
    range_model = (range_model or "long").lower().replace("-", "_")
    if range_model in ("short_range", "sr"):
        range_model = "short"
    elif range_model in ("mid_range", "medium", "medium_range", "mr"):
        range_model = "mid"
    elif range_model in ("long_range", "lr"):
        range_model = "long"
    if range_model not in NEUROPEPTIDE_RANGE_MODELS:
        raise ValueError(f"Unknown neuropeptide range model: {range_model}")
    return range_model


def _neuropeptide_data_roots():
    root = DOWNLOAD_DIR / prefix_NP
    seen = {root}
    yield root

    # Local CeDNe-web development often keeps the source CeDNe repository next
    # to this app, with freshly downloaded tables that are not staged here.
    for base in (TOPDIR.parent, TOPDIR.parent.parent):
        sibling_root = base / "CeDNe" / "data_sources" / "downloads" / "Ripoll-Sanchez_2023"
        if sibling_root not in seen:
            seen.add(sibling_root)
            yield sibling_root


def _neuropeptide_old_root():
    for root in _neuropeptide_data_roots():
        old_root = root / "old"
        if (old_root / "91-NPPGPCR networks").exists():
            return old_root
        if (root / "91-NPPGPCR networks").exists():
            return root
    raise FileNotFoundError("Could not find old Ripoll-Sanchez neuropeptide tables.")


def _neuropeptide_new_root():
    for root in _neuropeptide_data_roots():
        new_root = root / "new"
        if (new_root / NEUROPEPTIDE_NEW_PAIRS).exists():
            return new_root
    raise FileNotFoundError("Could not find new Ripoll-Sanchez neuropeptide tables.")


def _neuropeptide_pair_name(ligand, receptor):
    return f"{ligand}_{receptor}".replace("-", "_").replace(".", "_").replace(" ", "_")


def _read_new_neuropeptide_pairs():
    pairs_path = _neuropeptide_new_root() / NEUROPEPTIDE_NEW_PAIRS
    pairs = pd.read_csv(
        pairs_path,
        sep=r"\s+",
        header=None,
        names=["Ligand", "GPCR"],
        dtype=str,
    )
    pairs["pair_names_NPP"] = [
        _neuropeptide_pair_name(ligand, gpcr)
        for ligand, gpcr in zip(pairs["Ligand"], pairs["GPCR"])
    ]
    pairs["network_number"] = range(1, len(pairs) + 1)
    return pairs


def _read_old_neuropeptide_models():
    old_root = _neuropeptide_old_root()
    lrm = old_root / 'NPP_GPCR_networks_long_range_model_2.csv'
    nid = old_root / '26012022_num_neuronID.txt'
    np_order = old_root / '91-NPPGPCR networks'
    model = pd.read_csv(lrm, encoding='unicode_escape', header=None)
    neuronID = pd.read_csv(nid, encoding='unicode_escape', sep='\t', index_col=0, names=['NID', "Neuron"])
    neuropep_rec = pd.read_csv(np_order, sep=',', index_col=0)
    nidList = neuronID['Neuron'].to_list()

    models = {}
    for i, j in enumerate(range(0, len(model), len(neuronID))):
        models[i + 1] = np.array(model[j:j + len(neuronID)], dtype=np.int8)

    models_dict = {}
    for k, nprc in enumerate(neuropep_rec['pair_names_NPP']):
        npNum = k + 1
        models_dict[nprc] = _matrix_to_neuropeptide_adjacency(models[npNum], nidList, nidList)
    return neuropep_rec['pair_names_NPP'].tolist(), models_dict


def _matrix_to_neuropeptide_adjacency(matrix, row_names, column_names, allowed_neurons=None):
    allowed = set(allowed_neurons) if allowed_neurons is not None else None
    adjacency = {}
    for i, n1 in enumerate(row_names):
        if allowed is not None and n1 not in allowed:
            continue
        adjacency[n1] = {}
        for j, n2 in enumerate(column_names):
            if allowed is not None and n2 not in allowed:
                continue
            adjacency[n1][n2] = {'weight': int(matrix[i][j])}
    return adjacency


def _read_new_neuropeptide_model(pair_row, range_model, allowed_neurons=None):
    new_root = _neuropeptide_new_root()
    folder, suffix = NEUROPEPTIDE_RANGE_MODELS[range_model]
    network_number = int(pair_row["network_number"])
    model_path = (
        new_root / folder /
        f"01022024_neuropeptide_network{network_number:03d}_{suffix}_range_model.csv"
    )
    matrix = pd.read_csv(model_path, index_col=0)
    return _matrix_to_neuropeptide_adjacency(
        matrix.to_numpy(dtype=np.int8),
        matrix.index.astype(str).tolist(),
        matrix.columns.astype(str).tolist(),
        allowed_neurons=allowed_neurons,
    )


def loadNeuropeptides(w, neuropeps: str = 'all', mode: str = "old", range_model: str = "long"):
    ''' Loads Neuropeptides into neurons using Ripoll-Sanchez et al. 2023'''

    mode = _normalize_neuropeptide_mode(mode)
    range_model = _normalize_neuropeptide_range(range_model)
    allowed_neurons = w.neurons.keys() if type(w) == NervousSystem else None

    if mode == "old":
        npepreclist, models_dict = _read_old_neuropeptide_models()
    else:
        pairs = _read_new_neuropeptide_pairs()
        npepreclist = pairs['pair_names_NPP'].tolist()
        models_dict = {}

    if neuropeps == 'all':
        npepreclist_filter = set(npepreclist)
    elif isinstance(neuropeps, str):
        npepreclist_filter = {neuropeps}
    else:
        npepreclist_filter = set(neuropeps)

    if mode == "new":
        for _, pair_row in pairs.iterrows():
            nprc = pair_row["pair_names_NPP"]
            if nprc in npepreclist_filter:
                models_dict[nprc] = _read_new_neuropeptide_model(
                    pair_row,
                    range_model,
                    allowed_neurons=allowed_neurons,
                )

    for nprc in npepreclist:
        if nprc in npepreclist_filter:
            if type(w) == Worm:
                nn_np = NervousSystem(w, network="{}".format(nprc))
                nn_np.build_network(neuron_data=cell_list, adj=models_dict[nprc], label=nprc)
                w.citations.update({'neuropeptide_atlas': citations['neuropeptide_atlas']})
            elif type(w) == NervousSystem:
                w.setup_connections(adjacency=models_dict[nprc], connection_type=nprc)
                w.worm.citations.update({'neuropeptide_atlas': citations['neuropeptide_atlas']})


def getNeuropeptideList(mode: str = "old", range_model: str = "long"):
    ''' Returns the list of available neuropeptide networks '''
    mode = _normalize_neuropeptide_mode(mode)
    _normalize_neuropeptide_range(range_model)
    if mode == "old":
        old_root = _neuropeptide_old_root()
        np_order = old_root / '91-NPPGPCR networks'
        neuropep_rec = pd.read_csv(np_order, sep=',', index_col=0)
        return neuropep_rec['pair_names_NPP'].tolist()
    pairs = _read_new_neuropeptide_pairs()
    return pairs['pair_names_NPP'].tolist()

## Load CENGEN tables
thres_1 = DOWNLOAD_DIR / prefix_CENGEN / 'liberal_threshold1.csv'
thres_2 = DOWNLOAD_DIR / prefix_CENGEN / 'medium_threshold2.csv'
thres_3 = DOWNLOAD_DIR / prefix_CENGEN / 'conservative_threshold3.csv'
thres_4 = DOWNLOAD_DIR / prefix_CENGEN / 'stringent_threshold4.csv'

def returnThresholdDict(th1, th2, th3, th4, nnames, cengen_neurons):
    """
    Generate a dictionary of thresholds using CENGEN levels of sensitivity.

    Args:
        th1 (dict): 
            A dictionary mapping neuron names to their corresponding CENGEN threshold values for level 1.
        th2 (dict): 
            A dictionary mapping neuron names to their corresponding CENGEN threshold values for level 2.
        th3 (dict): 
            A dictionary mapping neuron names to their corresponding CENGEN threshold values for level 3.
        th4 (dict): 
            A dictionary mapping neuron names to their corresponding CENGEN threshold values for level 4.
        nnames (list): 
            A list of neuron names.
        cengen_neurons (dict): 
            A dictionary mapping neuron names to their corresponding indices.

    Returns:
        dict: A dictionary containing four dictionaries, each representing a level of sensitivity. Each inner dictionary maps gene names to a list of threshold values for that gene across all neurons.

    """
    #Load Thresholds
    full_th1, full_th2, full_th3, full_th4 = {}, {}, {}, {}
    for n in cengen_neurons.keys():
        full_th1[n] = th1[cengen_neurons[n]]
        full_th2[n] = th2[cengen_neurons[n]]
        full_th3[n] = th3[cengen_neurons[n]]
        full_th4[n] = th4[cengen_neurons[n]]

    gene_list = full_th1[nnames[0]].keys()

    th1_f = {g:[] for g in gene_list}
    th2_f = {g:[] for g in gene_list}
    th3_f = {g:[] for g in gene_list}
    th4_f = {g:[] for g in gene_list}

    for neuron in nnames:
        for g in gene_list:
            th1_f[g].append(full_th1[neuron][g])
            th2_f[g].append(full_th2[neuron][g])
            th3_f[g].append(full_th3[neuron][g])
            th4_f[g].append(full_th4[neuron][g])

    threshold_dict = {'1': th1_f, '2': th2_f, '3': th3_f, '4': th4_f}
    return threshold_dict

def loadTranscripts(nn, threshold=4):
    """
    Loads transcripts from CENGEN data files and assigns them to neuron objects.
    
    Parameters:
    - nn: Neuron object
    - thres_1, thres_2, thres_3, thres_4: Paths to CSV files containing transcript data
    
    Returns:
    - None
    """
    th1 = pd.read_csv(thres_1,encoding= 'unicode_escape', index_col=1).drop(['Wormbase_ID','Unnamed: 0'], axis = 'columns')
    th1 = th1[th1.columns]>0

    th2 = pd.read_csv(thres_2,encoding= 'unicode_escape', index_col=1).drop(['Wormbase_ID','Unnamed: 0'], axis = 'columns')
    th2 = th2[th2.columns]>0

    th3 = pd.read_csv(thres_3,encoding= 'unicode_escape', index_col=1).drop(['Wormbase_ID','Unnamed: 0'], axis = 'columns')
    th3 = th3[th3.columns]>0

    th4 = pd.read_csv(thres_4,encoding= 'unicode_escape', index_col=1).drop(['Wormbase_ID','Unnamed: 0'], axis = 'columns')
    th4 = th4[th4.columns]>0

    ## Group Names
    groupNames = {}
    for k in nn.neurons.keys():
        if k[-1] in ['L', 'R']:
            if k[-1] == 'L' and k[:-1] + 'R' in nn.neurons.keys():
                groupNames[k] = k[:-1]
            elif k[-1] == 'R' and k[:-1] + 'L' in nn.neurons.keys(): 
                groupNames[k] = k[:-1]
            else:
                groupNames[k] = k
        else:
            groupNames[k] = k

    ## CENGEN neurons 
    suffixes = ['D', 'V', 'L', 'R', 'DL', 'DR', 'VL', 'VR', '01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13']
    cengen_neurons  = {m:'' for m in groupNames.keys()}
    neuron_keys = list(cengen_neurons.keys())
    for m in neuron_keys:
        if m in th1.columns:
            cengen_neurons[m] = m
        elif groupNames[m] in th1.columns:
            cengen_neurons[m] = groupNames[m]
        elif m.startswith('AWC'):
            ## Arbitrary mapping, note the user
            cengen_neurons['AWCL'] = 'AWC_OFF'
            cengen_neurons['AWCR'] = 'AWC_ON'
        else:
            for k in th1.columns:
                if m.startswith(k):
                    cengen_neurons[m] = k
            
            if m.startswith('VD') or m.startswith('DD'):
                cengen_neurons[m] = 'VD_DD'

            if m == 'RMEL' or m == 'RMER':
                cengen_neurons[m] = 'RME_LR' 
            if m == 'RMED' or m == 'RMEV': 
                cengen_neurons[m] = 'RME_DV'
            if m == 'RMDL' or m == 'RMDR':
                cengen_neurons[m] = 'RMD_LR' 
            if m.startswith('RMDD') or m.startswith('RMDV'): 
                cengen_neurons[m] = 'RMD_DV'
            if m == 'IL2L' or m == 'IL2R':
                cengen_neurons[m] = 'IL2_LR'
            if m.startswith('IL2D') or m.startswith('IL2V'): 
                cengen_neurons[m] = 'IL2_DV' 
            


    cengen_neurons['VC04']= 'VC_4_5'
    cengen_neurons['VC05']=  'VC_4_5'
    cengen_neurons['DA09']=  'DA9'

    th_i = [th1, th2, th3, th4]
    for n in nn.neurons:
        nn.neurons[n].set_property('transcript', th_i[threshold-1][cengen_neurons[n]])
    nn.worm.citations.update({'cengen':citations['cengen']})

def get_enriched_neurons(network, target_neurons, excluded_neurons=None, threshold=4):
    """
    Returns the enriched neurons from the target neurons in the given neural network.

    Args:
        network (NeuralNetwork): The neural network object.
        target_neurons (list): List of target neuron names.
        excluded_neurons (list, optional): List of neuron names to exclude. Defaults to None.
        threshold (int, optional): Threshold value. Defaults to 4.

    Returns:
        enriched_neurons (list): List of enriched neuron names.
    """
    enriched_neurons = [
        neuron for neuron in target_neurons
        if network.neurons[neuron].transcript[threshold] > 0
    ]

    if excluded_neurons is not None:
        enriched_neurons = [
            neuron for neuron in enriched_neurons
            if neuron not in excluded_neurons
        ]

    return enriched_neurons

def loadGapJunctions(nn, threshold=4):
    """
    Use CENGEN data to load gap junction transcripts to known gap junctions in the given neural network.

    Parameters:
        nn (NeuralNetwork): The neural network object to update with gap junction transcripts.
        threshold (int, optional): The threshold value to use. Defaults to 4.

    Returns:
        None
    """
    if not hasattr(list(nn.neurons.values())[0], 'transcript'):
        loadTranscripts(nn, threshold)

    gene_names = list(nn.neurons.values())[0].transcript.index.tolist()
    gapjn_subunits = [g for g in gene_names if g.startswith('inx') or g in ['che-7', 'eat-5', 'unc-7', 'unc-9']]
    for e,conn in nn.connections.items():
        if e[0].name in nn.neurons and e[1].name in nn.neurons and conn.connection_type == 'gap-junction':
            n1 = set(e[0].transcript[e[0].transcript == True].index).intersection(gapjn_subunits)
            n2 = set(e[1].transcript[e[1].transcript == True].index).intersection(gapjn_subunits)
            # for g in gene_names:
            #     if g.startswith('inx') or g in ['che-7', 'eat-5', 'unc-7', 'unc-9']:
            #         if e[0].transcript[threshold][g]:
            #             n1.append(g)
            #         if e[1].transcript[threshold][g]:
            #             n2.append(g)
            conn.set_property('putative_gapjn_subunits', set([(e1,e2) for e1 in n1 for e2 in n2]))
    nn.worm.citations.update({'cengen':citations['cengen']})

## Synaptic weights
def loadSynapticWeights(nn):
    """
    Load synaptic weights from an Excel file into the given neural network.

    Parameters:
        nn (NeuralNetwork): The neural network object to update with synaptic weights.
        weightMatrix (str, optional): The path to the Excel file containing synaptic weights. 
        Defaults to leiferFile.

    Returns:
        None
    """
    ## Load synaptic weights from Excel file
    weightMatrix = DOWNLOAD_DIR / prefix_synaptic_weights / "41586_2023_6683_MOESM13_ESM.xls"
    wtMat = pd.read_excel(weightMatrix, index_col=0).T
    for sid in nn.connections.keys():
        if sid[0].name in wtMat:
            if sid[1].name in wtMat[sid[0].name]:
                nn.connections[sid].update_weight(wtMat[sid[0].name][sid[1].name])
            else:
                nn.connections[sid].update_weight(np.nan)
        else:
            nn.connections[sid].update_weight(np.nan)
    nn.worm.citations.update({'sig_prop_atlas':citations['sig_prop_atlas']})
    return wtMat

def download_datasets(key=''):
    """
    Downloads the required datasets from online sources.
    """
    if not key:
        print("Nothing downloaded. Pass key")
    elif key == 'cengen':
        cengen_dir = (DOWNLOAD_DIR / prefix_CENGEN).resolve()
        cengen_dir.mkdir(parents=True, exist_ok=True)
        for link in cengen_links:
            response = requests.get(link, stream=True)
            response.raise_for_status()  # Raises HTTPError for bad responses
            local_dir = cengen_dir
            local_filename = link.split('021821_')[-1]
            with open(local_dir / local_filename, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"Downloaded {local_filename} at {local_dir}")

    # elif key == 'fly_wire':
        # if not os.path.exists(fly_wire):
        #     os.makedirs(fly_wire)
        # for link in flywire_links:
        #     response = requests.get(link, stream=True)
        #     response.raise_for_status()  # Raises HTTPError for bad responses
        #     local_dir = fly_wire

        #     local_filename = link.split('data_product=')[-1].split('&data_version')[0] + '.csv.gz'
        #     with open(local_dir + local_filename, "wb") as f:
        #         for chunk in response.iter_content(chunk_size=8192):
        #             f.write(chunk)
        #     print(f"Downloaded {local_filename} at {local_dir}")

    elif key == 'atanas_whole_brain':
        for stim, location in atanas_whole_brain.items():
            location.mkdir(parents=True, exist_ok=True)

            for suff in atanas_links[stim]:
                link = atanas_link_prefix + suff
                response = requests.get(link, stream=True)
                response.raise_for_status()  # Raises HTTPError for bad responses
                local_dir = location
                local_filename = suff

                with open(local_dir / local_filename, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                print(f"Downloaded {local_filename} at {local_dir}")
    # elif key == 'neuropeptide_atlas':
    else:
        print("Not yet supported. Download manually into the directory.")


def load_recordings(animal, source, network=None, trial_num=0, time_col=0,
                    metadata=None, on_missing='skip'):
    """Attach per-neuron recordings to ``animal`` from a tidy CSV or DataFrame.

    Expected wide layout: first column is time (seconds), every remaining
    column is a 1D trace whose header is the neuron name. Each matched
    column becomes a Trial on that neuron via ``Neuron.load_recording``.
    Sampling rate is inferred from the median inter-sample interval of the
    time column and written to every trial's metadata.

    Args:
        animal: Target ``Animal`` / ``Worm`` / ``Fly``. Must already have at
            least one NervousSystem attached.
        source: CSV path (str / Path) or pandas DataFrame in the layout above.
        network: Name of the target NervousSystem in ``animal.networks``.
            Defaults to the first network (insertion order).
        trial_num: Trial index to create on each matched neuron.
        time_col: Index or column name of the time column (default 0).
        metadata: Extra metadata dict merged into every trial.
        on_missing: 'skip' (default) silently drops columns whose header does
            not match a neuron; 'raise' raises ``KeyError`` instead.

    Returns:
        dict: ``{'matched': [names], 'missing': [names],
                 'sampling_rate': Hz, 'trial_num': trial_num,
                 'network': <NervousSystem>}``
    """
    if isinstance(source, pd.DataFrame):
        df = source.copy()
    else:
        df = pd.read_csv(source)

    if isinstance(time_col, int):
        time_series = df.iloc[:, time_col]
        neuron_cols = [c for i, c in enumerate(df.columns) if i != time_col]
    else:
        time_series = df[time_col]
        neuron_cols = [c for c in df.columns if c != time_col]

    t = np.asarray(time_series, dtype=np.float64)
    dts = np.diff(t)
    if len(dts) == 0 or not np.all(np.isfinite(dts)) or np.median(dts) <= 0:
        raise ValueError("time column must be monotonic and have at least two samples")
    sampling_rate = float(1.0 / np.median(dts))

    if not animal.networks:
        raise ValueError("animal has no networks; load a connectome first")
    if network is None:
        network = next(iter(animal.networks))
    nn = animal.networks[network]

    matched, missing = [], []
    base_meta = {'sampling_rate': sampling_rate}
    if metadata:
        base_meta.update(metadata)

    for col in neuron_cols:
        col_name = str(col)
        if col_name not in nn.neurons:
            missing.append(col_name)
            if on_missing == 'raise':
                raise KeyError(f"column {col_name!r} not found in network {network!r}")
            continue
        trace = np.asarray(df[col], dtype=np.float64)
        nn.neurons[col_name].load_recording(
            trace, trial_num=trial_num, metadata=dict(base_meta))
        matched.append(col_name)

    return {
        'matched': matched,
        'missing': missing,
        'sampling_rate': sampling_rate,
        'trial_num': trial_num,
        'network': nn,
    }


""" This is experimental, not yet tested """
def load_nwb(filepath):
    """
    Loads an NWB file and maps its content to CeDNe core objects.
    
    Supports standard NWB mapping and C. elegans extensions (NWBelegans).
    
    Args:
        filepath (str): Path to the NWB file.
        
    Returns:
        tuple: (NervousSystem, Session)
    """
    try:
        from pynwb import NWBHDF5IO
    except ImportError:
        raise ImportError("pynwb is required for load_nwb. Install it via 'pip install pynwb'.")

    
    
    with NWBHDF5IO(filepath, 'r') as io:
        nwbfile = io.read()
        species = getattr(nwbfile.subject, 'species', '').lower()
        if 'elegans' in species or not species: # Default to Worm if unspecified or elegans
            w = Worm(name=nwbfile.session_id or "NWB_Worm")
            nn = NervousSystem(w)
        else:
            a = Animal(name=nwbfile.session_id or "NWB_Animal", species=species)
            nn = NervousSystem(a)
        
        # Setting up context
        exp_context = ExperimentalContext(
            name="NWB_Context", 
            experimental_params={"description": nwbfile.session_description}
        )
        ctx = Context(name="NWB_Combined_Context", experimental=exp_context)
        
        # Setting up session
        session = Session(
            name=nwbfile.session_id or "NWB_Session",
            context=ctx,
            date=nwbfile.session_start_time,
            experimenter=nwbfile.experimenter,
            lab=nwbfile.lab,
            institution=nwbfile.institution
        )
        
        #Volumetric Imaging (NWBelegans/ophys)
        if 'ophys' in nwbfile.processing:
            ophys_mod = nwbfile.processing['ophys']
            
            # Look for VolumeSegmentation (Cell IDs and Masks)
            for name, obj in ophys_mod.data_interfaces.items():
                if 'Segmentation' in obj.type_hierarchy: # Matches VolumeSegmentation or ImageSegmentation
                    # In NWBelegans, there might be multiple planes/volumes
                    for plane_seg in obj.plane_segmentations.values():
                        for row in plane_seg:
                            neuron_name = str(row.index[0]) # Fallback to index if no name
                            if 'cell_id' in plane_seg.colnames:
                                neuron_name = row['cell_id'].values[0]
                            
                            # Create Neuron if it doesn't exist
                            if neuron_name not in nn.neurons:
                                mask = None
                                if 'voxel_mask' in plane_seg.colnames:
                                    mask = row['voxel_mask'].values[0]
                                elif 'image_mask' in plane_seg.colnames:
                                    mask = row['image_mask'].values[0]
                                    
                                Neuron(neuron_name, nn, spatial_mask=mask)

            # MultiChannelVolumeSeries (Activity Traces)
            if 'Fluorescence' in ophys_mod.data_interfaces:
                fl = ophys_mod.data_interfaces['Fluorescence']
                for roi_resp_series in fl.roi_response_series.values():
                    # Map ROI indices to Neuron objects
                    # This assumes Neurons were created from VolumeSegmentation in order
                    neuron_list = list(nn.neurons.values())
                    for i, neuron in enumerate(neuron_list):
                        if i < roi_resp_series.data.shape[1]:
                            trial = neuron.add_trial(0)
                            trial.recording = roi_resp_series.data[:, i]
                            # Add OpticalChannel metadata if available
                            if hasattr(roi_resp_series, 'imaging_plane'):
                                plane = roi_resp_series.imaging_plane
                                for j, channel in enumerate(plane.optical_channel):
                                    trial.add_metadata(f'optical_channel_{j}', {
                                        'excitation_lambda': channel.excitation_lambda,
                                        'emission_lambda': channel.emission_lambda,
                                        'description': channel.description
                                    })
                            session.add_trial(trial)

        # Spikes (Units), for future implementations, i.e. not C.elegans or perhaps some neurons like AWA?
        if nwbfile.units:
            for i, unit_id in enumerate(nwbfile.units.id):
                neuron_name = f"Unit_{unit_id}"
                if neuron_name not in nn.neurons:
                    neuron = Neuron(neuron_name, nn)
                else:
                    neuron = nn.neurons[neuron_name]
                
                trial = neuron.add_trial(0)
                trial.recording = nwbfile.units['spike_times'][i]
                session.add_trial(trial)

        # Handle Behavior (SpatialSeries)
        if 'behavior' in nwbfile.processing:
            beh_mod = nwbfile.processing['behavior']
            for name, obj in beh_mod.data_interfaces.items():
                if 'SpatialSeries' in obj.type_hierarchy:
                    if not hasattr(nn.worm, 'behavior'):
                        nn.worm.behavior = Behavior()
                    setattr(nn.worm.behavior, name, obj.data[:])

        return nn, session


def load_atanas(condition='Control', max_files=None, network=None):
    """
    Load Atanas et al. (2023) whole-brain calcium imaging data into CeDNe.

    Reads downloaded JSON recordings from the Atanas whole-brain dataset and
    populates Trial objects on neurons in the network. Each JSON file is treated
    as a separate Session with its own ExperimentalContext.

    Args:
        condition (str): 'Control' or 'Heat'. Selects the experimental condition.
        max_files (int, optional): Limit how many JSON files to load (for quick demos).
            If None, loads all available files.
        network (NervousSystem, optional): Existing network to attach recordings to.
            If None, a new worm is created via makeWorm('atanas', chem_only=True).

    Returns:
        dict: {
            'network': NervousSystem,
            'sessions': list of Session objects,
            'neurons_loaded': int (number of neurons with recordings),
            'num_timepoints': int (length of first trace found),
            'condition': str,
            'worm': Worm
        }

    Raises:
        FileNotFoundError: If no Atanas data files are found. Run
            download_data('atanas_whole_brain') first.
        ValueError: If condition is not 'Control' or 'Heat'.
    """
    import json

    if condition not in atanas_whole_brain:
        raise ValueError(
            f"condition must be one of {list(atanas_whole_brain.keys())}, got '{condition}'"
        )

    data_dir = atanas_whole_brain[condition]
    if not data_dir.exists():
        raise FileNotFoundError(
            f"Atanas data directory not found at {data_dir}. "
            "Run download_data('atanas_whole_brain') first."
        )

    # Find available JSON files
    json_files = sorted(data_dir.glob('*.json'))
    if not json_files:
        raise FileNotFoundError(f"No JSON files found in {data_dir}")
    if max_files:
        json_files = json_files[:max_files]

    # Build or use existing network
    if network is None:
        w = makeWorm('atanas', chem_only=True)
        nn = w.networks['Neutral']
    else:
        nn = network
        w = nn.worm

    sessions = []
    neurons_loaded = set()
    num_timepoints = 0

    for trial_num, json_path in enumerate(json_files):
        with open(json_path, 'r') as f:
            data = json.load(f)

        trace_array = data.get('trace_array', {})
        neuron_labels = data.get('neuron_labels', {})

        # Build a reverse map: neuron_label_index → trace_key
        # In the Atanas format, neuron_labels maps index → neuron name,
        # and trace_array maps neuron name → trace
        label_to_name = {}
        if isinstance(neuron_labels, dict):
            label_to_name = {v: v for v in neuron_labels.values()}
        elif isinstance(neuron_labels, list):
            label_to_name = {name: name for name in neuron_labels}

        # Create experimental context for this recording
        exp_context = ExperimentalContext(
            name=f"Atanas_{condition}_{json_path.stem}",
            experimental_params={
                'condition': condition,
                'source_file': json_path.name,
                'dataset': 'Atanas et al. (2023)',
                'doi': 'https://doi.org/10.1016/j.cell.2023.07.035',
                'imaging_modality': 'whole-brain calcium imaging',
            }
        )
        ctx = Context(
            name=f"{condition}_{json_path.stem}",
            experimental=exp_context,
            description=f"Whole-brain calcium imaging, {condition} condition"
        )

        session = Session(
            name=json_path.stem,
            context=ctx,
            condition=condition,
            source_file=json_path.name,
        )

        # Extract Behavior if present
        behavior = None
        behavior_keys = ['velocity', 'angular_velocity', 'head_curvature', 'body_curvature', 'pumping']
        found_behavior = {k: data[k] for k in behavior_keys if k in data and isinstance(data[k], list)}
        
        if found_behavior:
            from cedne.core.behavior import Behavior
            behavior = Behavior(worm=w)
            behavior.metadata['source_file'] = json_path.name
            # Identify timepoints for timestamps
            temp_num_tp = 0
            if isinstance(trace_array, dict) and trace_array:
                temp_num_tp = len(next(iter(trace_array.values())))
            elif isinstance(trace_array, list) and trace_array:
                temp_num_tp = len(trace_array[0])
            
            ts = np.arange(temp_num_tp) * (float(data.get('avg_timestep', 0.01)) * 60.0)
            for k, vals in found_behavior.items():
                if len(vals) == temp_num_tp:
                    behavior.add_variable(k, np.array(vals, dtype=np.float64), timestamps=ts)
            
            # Link behavior to session
            session.behavior = behavior

        # Assign traces to matching neurons
        trace_items = []
        if isinstance(trace_array, dict):
            # Format: { "NeuronName": [trace], ... }
            trace_items = list(trace_array.items())
        elif isinstance(trace_array, list):
            # Format: [ [trace], [trace], ... ]
            # We must map indices to names.
            
            # Path A: Try neuron_labels (list of names or dict index->name)
            if isinstance(neuron_labels, list) and len(neuron_labels) > 0:
                for idx, name in enumerate(neuron_labels):
                    if idx < len(trace_array):
                        trace_items.append((name, trace_array[idx]))
            elif isinstance(neuron_labels, dict) and len(neuron_labels) > 0:
                for idx_str, name in neuron_labels.items():
                    try:
                        idx = int(idx_str)
                        if idx < len(trace_array):
                            trace_items.append((name, trace_array[idx]))
                    except ValueError:
                        pass
            
            # Path B: If Path A yielded nothing, try 'labeled' (dict index -> {label: name, ...})
            if not trace_items:
                labeled = data.get('labeled', {})
                if isinstance(labeled, dict):
                    for idx_str, info in labeled.items():
                        try:
                            idx = int(idx_str)
                            # In labeled dict, 'label' usually contains the neuron name
                            name = None
                            if isinstance(info, dict):
                                name = info.get('label')
                            elif isinstance(info, str):
                                name = info
                            
                            if name and idx < len(trace_array):
                                trace_items.append((name, trace_array[idx]))
                        except (ValueError, TypeError):
                            pass
        
        if not trace_items:
            print(f"Warning: No valid trace mapping found in {json_path}")
            continue

        for neuron_name, trace in trace_items:
            # Match neuron names to network neurons (handle case differences)
            matched_name = None
            if neuron_name in nn.neurons:
                matched_name = neuron_name
            else:
                # Try case-insensitive matching
                for nn_name in nn.neurons:
                    if nn_name.upper() == neuron_name.upper():
                        matched_name = nn_name
                        break

            if matched_name is not None:
                neuron = nn.neurons[matched_name]
                trial = neuron.add_trial(trial_num)
                trial.recording = np.array(trace, dtype=np.float64)
                neuron.add_citation(_atanas_recording_citation())
                trial.metadata.update({
                    'condition': condition,
                    'source_file': json_path.name,
                    'dataset': 'atanas_whole_brain',
                    'sampling_rate': 1.0 / (float(data.get('avg_timestep', 0.01)) * 60.0)
                })
                # Link behavior to trial as well
                if behavior:
                    trial.behavior = behavior
                
                session.add_trial(trial)
                neurons_loaded.add(matched_name)

                if num_timepoints == 0:
                    num_timepoints = len(trace)

        sessions.append(session)

    return {
        'network': nn,
        'sessions': sessions,
        'neurons_loaded': len(neurons_loaded),
        'num_timepoints': num_timepoints,
        'condition': condition,
        'worm': w,
    }
