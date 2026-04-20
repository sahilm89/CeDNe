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
import numpy as np
import pandas as pd
import requests
from cedne import Worm, Fly, NervousSystem
from cedne.core import Neuron, Behavior, Session
from cedne.core.context import Context, ExperimentalContext
from cedne.core.animal import Animal
from .config import *
warnings.filterwarnings("ignore", category=UserWarning, module='openpyxl')

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
        input_file = 'SI 5 Connectome adjacency matrices, corrected July 2020.xlsx'

        ## Chemical synapses
        cook_chem = pd.read_excel(cook_connectome / input_file, sheet_name='male chemical', engine='openpyxl')
        colnames = cook_chem.iloc[1, 3:-1].astype(str).tolist()
        labels = cook_chem.loc[2:383]['Unnamed: 2'].tolist()

        ccl = cook_chem.iloc[2:,:2].copy().ffill()

        list_1 = ccl.iloc[:,0].to_list() #.to_csv('temp_filled.csv', index=False)
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

        # cook_chem.ffill().to_csv('temp_filled.csv', index=False)
        
        cook_chem = cook_chem.drop(columns=cook_chem.columns[:3], index=cook_chem.index[:2])
        cook_chem = cook_chem.drop(columns=cook_chem.columns[-1], index=cook_chem.index[-1])
        cook_chem.reset_index(drop=True, inplace=True)
        cook_chem.columns = colnames
        cols = cook_chem.columns.to_list()
        chem_adj = cook_chem.to_numpy()
        adj_chem = {}
        for i, row in enumerate(labels):
            # idx = cook_chem.columns.get_loc()
            adj_chem[row] = {col1: {"weight": chem_adj[i,j]} for j,col1 in enumerate(cols) if col1 in labels}

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
        
        for i,row in enumerate(row_labels):
            # idx = cook_gapjn.columns.get_loc(col)
            if row in labels:
                adj_gapjn[row] = {col1: {"weight":gapjn_adj[i,j]} for j,col1 in enumerate(cols) if col1 in labels}

        nn.create_neurons(labels=labels, type=ntype, category=l1)
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
        nn.create_neurons(labels=labels)
        witvliet_input.rename(columns={'synapses': 'weight'}, inplace=True)
        fin_input = witvliet_input[witvliet_input['pre'].isin(labels)]
        fin_input = fin_input[fin_input['post'].isin(labels)]
        for _, conn in fin_input.iterrows():
            nn.setup_connections(conn, conn['type'], input_type='edge')
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
    
    # Fallback to nodes.csv 2D positions if 3D not found
    pos_dict = {}
    for _, row in names_df.iterrows():
        idx = str(row['index'])
        name = str(row['name'])
        if idx in pos3d_dict:
            pos_dict[name] = pos3d_dict[idx]
        elif name in pos3d_dict:
            pos_dict[name] = pos3d_dict[name]
        else:
            # Parse 2D array from nodes.csv
            try:
                # nodes.csv has 2D projection
                p2d = np.array([float(x) for x in row['_pos'].split('[')[-1].split(']')[0].split(',')])
                pos_dict[name] = np.array([p2d[0], p2d[1], 0.0]) # Add dummy Z
            except:
                pos_dict[name] = np.array([0.0, 0.0, 0.0])

    node_colors = {row['name']: "#" + row['color'][-6:] for _, row in names_df.iterrows()}
    
    neuron_indices = sorted(names_df['index'].tolist())    
    
    # Enrichment attributes
    type_dict = {}
    annot_dict = {}
    for nidx in neuron_indices:
        name = neuron_dict[nidx]
        info = bio_mapping.get(name) or bio_mapping.get(str(nidx))
        if info:
            type_dict[name] = info['cell_type']
            annot_dict[name] = info['annotation']
        else:
            type_dict[name] = 'Other'
            annot_dict[name] = 'Unknown'

    nn.create_neurons([neuron_dict[n] for n in neuron_indices], 
                      position=pos_dict,
                      color=node_colors,
                      cell_type=type_dict,
                      annotation=annot_dict)

    ## Connections
    for _, row in conns.iterrows():
        adjacency = {'pre': neuron_dict[row['source']], 'post': neuron_dict[row['target']], 'weight': row['depth']}
        nn.setup_connections(adjacency, connection_type='chemical-synapse', input_type='edge')

    return a

def make_platynereis():
    pass


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

def getLigands(neuron, sex='Hermaphrodite'):
    ''' Returns ligand for each neuron'''
    lig_file = DOWNLOAD_DIR / prefix_NT / 'ligand-table.xlsx'
    if sex in ['Hermaphrodite', 'hermaphrodite']:
        ligtable = pd.read_excel(lig_file, sheet_name='Hermaphrodite, sorted by neuron', skiprows=7, engine='openpyxl')
    elif sex in ['Male', 'male']:
        ligtable = pd.read_excel(lig_file, sheet_name='Male neurons, sorted by neuron', skiprows=7, engine='openpyxl')
    else:
        raise ValueError("Sex must be 'Hermaphrodite' or 'Male'")

    a,b = ligtable['Neurotransmitter 1'][ligtable['Neuron']==neuron].to_list(), ligtable['Neurotransmitter 2'][ligtable['Neuron']==neuron].to_list()
    
    if len(a):
        if len(b) and type(b[0])==str:
            return [a[0],b[0]]
        else:
            return [a[0]]
    else:
        return []

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
            receptor_ligand.update ({r: ligands[0]})
        else:
            receptor_ligand.update ({r: ''})
    return receptor_ligand


def loadNeurotransmitters(nn, sex='Hermaphrodite'):
    ''' Loads Neurotransmitters into neurons using Wang et al 2024'''
    
    npr_file = DOWNLOAD_DIR / prefix_NT / 'GenesExpressing-BATCH-thrs4_use.xlsx'
    npr = pd.read_excel(npr_file, sheet_name='npr', true_values='TRUE', false_values='FALSE', engine='openpyxl')
    ligmap = pd.read_excel(npr_file, sheet_name='ligmap', engine='openpyxl')

    for n in nn.neurons:
        neuron = nn.neurons[n]
        if not hasattr(neuron, '_preSynapse'):
            nn.neurons[n]._preSynapse = []
        if not hasattr(neuron, '_postSynapse'):
            nn.neurons[n]._postSynapse = {}

    for col in npr.columns:
        for suffix in suffixes:
            if col + suffix in nn.neurons:
                nn.neurons[col + suffix]._postSynapse.update(getLigandsAndReceptors(npr, ligmap, col))
                #present = True
    for n in nn.neurons:
        nn.neurons[n]._preSynapse += getLigands(n, sex=sex)
    
    for e,conn in nn.connections.items():
        if e[0].name in nn.neurons and e[1].name in nn.neurons and conn.connection_type == 'chemical-synapse':
            conn.set_property('ligands', nn.neurons[e[0].name]._preSynapse)
            conn.set_property('receptors', nn.neurons[e[1].name]._postSynapse)
            conn.set_property('putative_neurotrasmitter_receptors', []) 
            for rec, lig in conn.receptors.items():
                if lig in conn.ligands:
                    conn.putative_neurotrasmitter_receptors.append((lig, rec))
    nn.worm.citations.update({'neurotransmitter_atlas':citations['neurotransmitter_atlas']})

## Neuropeptides tables

def loadNeuropeptides(w, neuropeps:str= 'all'):
    ''' Loads Neuropeptides into neurons using Ripoll-Sanchez et al. 2023'''

    #csvfile = DOWNLOAD_DIR + prefix_NP + 'neuropeptideConnectome.txt'
    lrm = DOWNLOAD_DIR /  prefix_NP / 'NPP_GPCR_networks_long_range_model_2.csv'
    nid = DOWNLOAD_DIR /  prefix_NP / '26012022_num_neuronID.txt'
    np_order = DOWNLOAD_DIR /  prefix_NP / '91-NPPGPCR networks'
    model = pd.read_csv(lrm,encoding= 'unicode_escape', header=None)
    neuronID = pd.read_csv(nid,encoding= 'unicode_escape', sep='\t', index_col=0, names=['NID', "Neuron"]) 
    neuropep_rec = pd.read_csv(np_order, sep=',', index_col=0)
    nidList = neuronID['Neuron'].to_list()

    models_dict = {nprc: {} for nprc in neuropep_rec['pair_names_NPP']}

    models = {}
    for i,j in enumerate(range(0,len(model),len(neuronID))):
        models[i+1] = np.array(model[j:j+len(neuronID)], dtype=np.int8)

    for k, nprc in enumerate(neuropep_rec['pair_names_NPP']):
        npNum = k+1
        for i,n1 in enumerate(nidList):
            models_dict[nprc] [n1] = {}
            for j, n2 in enumerate(nidList):
                models_dict[nprc][n1][n2] = {'weight':models[npNum][i][j]}
    npepreclist = neuropep_rec['pair_names_NPP'].tolist()
    if neuropeps != 'all':
        npepreclist_filter = neuropeps
    else:
        npepreclist_filter = npepreclist

    for nprc, model in zip(npepreclist, models ):
        if nprc in npepreclist_filter:
            if type(w)==Worm:
                print(nprc, model, models_dict[nprc])
                nn_np = NervousSystem(w, network="{}".format(nprc))
                nn_np.build_network(neuron_data=cell_list, adj=models_dict[nprc], label=nprc)
                w.citations.update({'neuropeptide_atlas':citations['neuropeptide_atlas']})
            elif type(w)==NervousSystem:
                w.setup_connections(adjacency=models_dict[nprc], connection_type=nprc)
                w.worm.citations.update({'neuropeptide_atlas':citations['neuropeptide_atlas']})

def getNeuropeptideList():
    ''' Returns the list of available neuropeptide networks '''
    np_order = DOWNLOAD_DIR / prefix_NP / '91-NPPGPCR networks'
    neuropep_rec = pd.read_csv(np_order, sep=',', index_col=0)
    return neuropep_rec['pair_names_NPP'].tolist()

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