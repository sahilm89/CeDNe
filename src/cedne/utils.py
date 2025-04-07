## File containing utility functions for CeDNe

## Dependencies
import os
from re import T
import warnings
import numpy as np
import pandas as pd
import datetime
import copy
import pickle

import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib
from matplotlib.patches import Circle
from matplotlib.text import Annotation
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d.proj3d import proj_transform
from mpl_toolkits.axes_grid1 import make_axes_locatable

from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation, PillowWriter

import textalloc as ta
import cmasher as cmr

import cedne as ced
from cedne import cedne
from cedne import simulator

warnings.filterwarnings('ignore', category=UserWarning, module='openpyxl')

## Directories

def get_root_path():
    """Returns the path to the root directory of the library."""
    return '/'.join(ced.__path__[0].split('/')[:-2])

root_path = get_root_path()

TOPDIR = root_path + '/' ## Change this to cedne and write a function to download data from an online server for heavy data.
DATADIR = TOPDIR + 'data_sources/'
DOWNLOAD_DIR = TOPDIR + 'data_sources/downloads/'
OUTPUT_DIR = TOPDIR + f'Output/{str(datetime.datetime.now()).split(" ")[0]}/'

prefix_NT = 'Wang_2024/'
prefix_CENGEN = 'CENGEN/'
prefix_NP = 'Ripoll-Sanchez_2023/'
prefix_synaptic_weights = 'Randi_2023/'

## Loading and building functions
cell_list = DATADIR + "Cell_list.pkl"
chemsyns = DATADIR + "chem_adj.pkl"
elecsyns = DATADIR + "gapjn_symm_adj.pkl"
neuronPositions = DATADIR + "neuronPosition.pkl"

cook_connectome = DOWNLOAD_DIR + 'cook_2019/'
witvliet_connectome = DOWNLOAD_DIR + 'witvliet_2020/'

lineage = DOWNLOAD_DIR + 'Worm_Atlas/Altun_lineage_corrected.xlsx'

## FlyWire
fly_wire = DOWNLOAD_DIR + 'FlyWire/'

def makeWorm(name='', import_parameters=None, chem_only=False, gapjn_only=False):
    ''' Utility function to make a Worm based on import parameters.'''
    if import_parameters is None or (import_parameters['style'] == 'cook' and import_parameters['sex'] == 'hermaphrodite'):
        w = cedne.Worm(name)
        nn = cedne.NervousSystem(w)
        build_nervous_system(nn, neuron_data=cell_list, \
                            chem_synapses=chemsyns, \
                            elec_synapses=elecsyns, \
                            positions=neuronPositions, \
                            chem_only=chem_only, \
                            gapjn_only=gapjn_only)
    elif (import_parameters['style'] == 'cook' and import_parameters['sex'] == 'male'):
        w = cedne.Worm(name)
        nn = cedne.NervousSystem(w)
        input_file = 'SI 5 Connectome adjacency matrices, corrected July 2020.xlsx'

        ## Chemical synapses
        cook_chem = pd.read_excel(cook_connectome + input_file, sheet_name='male chemical', engine='openpyxl')
        colnames = cook_chem.iloc[1][3:-1]
        labels = cook_chem.loc[2:383]['Unnamed: 2'].tolist()

        ccl = cook_chem.iloc[2:,:2].ffill()

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
        
        cook_chem = cook_chem.drop(columns=cook_chem.columns[:3],index=cook_chem.index[:2])
        #cols = ['/'.join(list(a)) for a in zip(l1_list, l2_list, colnames)]
        cook_chem = cook_chem.drop(columns=cook_chem.columns[-1],index=cook_chem.index[-1])
        cook_chem.columns = colnames
        cols = cook_chem.columns.to_list()
        chem_adj = cook_chem.to_numpy()
        adj_chem = {}
        for i, row in enumerate(labels):
            # idx = cook_chem.columns.get_loc()
            adj_chem[row] = {col1: {"weight": chem_adj[i,j]} for j,col1 in enumerate(cols) if col1 in labels}

        ## Gap junctions
        cook_gapjn = pd.read_excel(cook_connectome + input_file, sheet_name='male gap jn symmetric', engine='openpyxl')
        colnames = cook_gapjn.iloc[1][3:-1]

        row_labels = cook_gapjn.loc[2:383]['Unnamed: 2'].tolist()

        cook_gapjn = cook_gapjn.drop(columns=cook_gapjn.columns[:3],index=cook_gapjn.index[:2])
        #cols = ['/'.join(list(a)) for a in zip(l1_list, l2_list, colnames)]
        cook_gapjn = cook_gapjn.drop(columns=cook_gapjn.columns[-1],index=cook_gapjn.index[-1])
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
    else:
        if import_parameters['style'] == 'witvliet':
            ind_dict = {'L1': [1,2,3,4], 'L2':[5] , 'L3':[6], 'adult':[7,8]}
            assert import_parameters['stage'] in ['L1', 'L2', 'L3', 'adult'], "stage should be one of 'L1', 'L2', 'L3', 'adult'"
            assert int(import_parameters['dataset_ind']) in range(1,len(ind_dict[import_parameters['stage']])+1) , f"Dataset id {int(import_parameters['dataset_ind'])} for stage {import_parameters['stage']} should be in {list(range(1,len(ind_dict[import_parameters['stage']])+1))}"

            input_file = 'witvliet_2020_' + str(ind_dict[import_parameters['stage']][int(import_parameters['dataset_ind'])-1]) + ' ' + import_parameters['stage'] + '.xlsx'
            witvliet_input = pd.read_excel(witvliet_connectome + input_file, engine='openpyxl')
            all_labels = set(witvliet_input['pre'])|set(witvliet_input['post'])
            labels = [lab for lab in all_labels if not any(lab.startswith(k) for k in ['BWM-', 'CEPsh', 'GLR'])]

            w = cedne.Worm(name=name, stage=import_parameters['stage'])
            nn = cedne.NervousSystem(w, network= '_'.join([import_parameters['style'],import_parameters['stage'], import_parameters['dataset_ind']]))
            nn.create_neurons(labels=labels)
            witvliet_input.rename(columns={'synapses': 'weight'}, inplace=True)
            fin_input = witvliet_input[witvliet_input['pre'].isin(labels)]
            fin_input = fin_input[fin_input['post'].isin(labels)]
            for iter, conn in fin_input.iterrows():
                nn.setup_connections(conn, conn['type'], input_type='edge')
                

    return w

def makeFly(name = ''):
    f = cedne.Fly(name)
    nn = cedne.NervousSystem(f)

    ## Neurons

    ### Names
    names = pd.read_csv(fly_wire + 'names.csv')
    labs, neuron_types, lab_root_id = names['name'], names['group'], names['root_id']
    neuron_dict = {r:lab for r,lab in zip(lab_root_id, labs)}
    type_dict = {r:ntype for r,ntype in zip(lab_root_id, neuron_types)} 
    
    root_ids = sorted(lab_root_id)
    labels = [neuron_dict[rid] for rid in root_ids]
    neuron_types = {neuron_dict[rid]:type_dict[rid] for rid in root_ids}
    
    ### Positions
    coordinates = pd.read_csv(fly_wire + 'coordinates.csv')
    pos_root_id, position = coordinates['root_id'], coordinates['position']
    position_dict = {neuron_dict[rid]:np.array(list(filter(None, pos.split('[')[-1].split(']')[0].split(' '))), dtype=int) for rid,pos in zip(pos_root_id, position)}
    
    ### Stats
    stats = pd.read_csv(fly_wire + 'cell_stats.csv')
    stats_root_id, nlength, narea, nvolume = stats['root_id'], np.array(stats['length_nm'], dtype=int), np.array(stats['area_nm'], dtype=int), np.array(stats['size_nm'], dtype=int)

    length_dict = {neuron_dict[rid]:nlen for (rid,nlen) in zip(stats_root_id, nlength)}
    area_dict = {neuron_dict[rid]:nare for (rid,nare) in zip(stats_root_id, narea)}
    vol_dict = {neuron_dict[rid]:nvol for (rid,nvol) in zip(stats_root_id, nvolume)}

    nn.create_neurons(labels, type=neuron_types, position=position_dict, length=length_dict, area=area_dict, volume=vol_dict)

    ## Connections
    conns = pd.read_csv(fly_wire + 'connections_no_threshold.csv')
    pre_rid, post_rid, weights, nts = conns['pre_root_id'], conns['post_root_id'], conns['syn_count'], conns['nt_type']

    for pre, post, weight, nt in zip(pre_rid, post_rid, weights, nts ):
        adjacency = {'pre':neuron_dict[pre], 'post':neuron_dict[post], 'weight':weight}
        neurotransmitter = {'neurotransmitter':nt}
        nn.setup_connections(adjacency, connection_type='chemical-synapse', input_type='edge', neurotransmitter=neurotransmitter)
    return f


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
    lig_file = DOWNLOAD_DIR + prefix_NT +'ligand-table.xlsx'
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
    
    npr_file = DOWNLOAD_DIR + prefix_NT + 'GenesExpressing-BATCH-thrs4_use.xlsx'
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

## Neuropeptides tables

def loadNeuropeptides(w, neuropeps:str= 'all'):
    ''' Loads Neuropeptides into neurons using Ripoll-Sanchez et al. 2023'''

    #csvfile = DOWNLOAD_DIR + prefix_NP + 'neuropeptideConnectome.txt'
    lrm = DOWNLOAD_DIR +  prefix_NP + 'NPP_GPCR_networks_long_range_model_2.csv'
    nid = DOWNLOAD_DIR +  prefix_NP + '26012022_num_neuronID.txt'
    np_order = DOWNLOAD_DIR +  prefix_NP + '91-NPPGPCR networks'

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
            if type(w)==cedne.Worm:
                print(nprc, model, models_dict[nprc])
                nn_np = cedne.NervousSystem(w, network="{}".format(nprc))
                nn_np.build_network(neuron_data=cell_list, adj=models_dict[nprc], label=nprc)
            elif type(w)==cedne.NervousSystem:
                w.setup_connections(adjacency=models_dict[nprc], connection_type=nprc)


## Load CENGEN tables
thres_1 = DOWNLOAD_DIR + prefix_CENGEN + 'liberal_threshold1.csv'
thres_2 = DOWNLOAD_DIR + prefix_CENGEN + 'medium_threshold2.csv'
thres_3 = DOWNLOAD_DIR + prefix_CENGEN + 'conservative_threshold3.csv'
thres_4 = DOWNLOAD_DIR + prefix_CENGEN + 'stringent_threshold4.csv'

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
    for m in cengen_neurons.keys():
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
    weightMatrix = DOWNLOAD_DIR + prefix_synaptic_weights + "41586_2023_6683_MOESM13_ESM.xls"
    wtMat = pd.read_excel(weightMatrix, index_col=0).T
    for sid in nn.connections.keys():
        if sid[0].name in wtMat:
            if sid[1].name in wtMat[sid[0].name]:
                nn.connections[sid].update_weight(wtMat[sid[0].name][sid[1].name])
            else:
                nn.connections[sid].update_weight(np.nan)
        else:
            nn.connections[sid].update_weight(np.nan)
    return wtMat


## Graph contraction functions
def joinLRNodes(nn_old):
    """
    Joins left and right nodes together in the neural network based on specific conditions.
    
    Parameters:
    - nn_old: The original neural network to be processed.
    
    Returns:
    - nn_new: The updated neural network after joining left and right nodes.
    """
    nn_new = nn_old.copy()
    for m in nn_new.neurons:
        if m[-1]=='L' and not m in ['AVL']:
            nn_new.neurons[m].name = m[:-1]
            n=m[:-1]+'R'
            if n in nn_new.neurons:
                neuronPair = [m,n]
                nn_new.contract_neurons(neuronPair, m[:-1])
    return nn_new

# def foldByNeuronType(nn_old):
#     """
#     Folds neurons in the given neural network based on the neuron type.

#     Args:
#         nn_old (NeuralNetwork): The original neural network.

#     Returns:
#         NeuralNetwork: The folded neural network.
    
#     """
        
def foldByNeuronType(nn_old, exceptions=[], self_loops=True, data='clean'):
    """
    Folds neurons in the given neural network based on the neuron type.

    Args:
        nn_old (NeuralNetwork): The original neural network.

    Returns:
        NeuralNetwork: The folded neural network.
    """
    # nn_new = nn_old.copy()
    neuron_class = {}
    argmax = lambda lst: lst.index(max(lst))
    for n in nn_old.neurons:
        _suffs = []
        _clsname = []
        for s in suffixes:
            if n.endswith(s):
                n0 = n[:-len(s)]
                if len(n0)>2:
                    _suffs.append(s)
                    _clsname.append(n0)
                elif len(n0)>1:
                    if (n0[-1] in '0123456789') or (s[0] in '0123456789') or n0 in ['MC']:
                        _suffs.append(s)
                        _clsname.append(n0)
                
        if len(_suffs) > 0:
            _sufflen = [len(s0) for s0 in _suffs]
            j = argmax(_sufflen)
            if not _clsname[j] in neuron_class:
                neuron_class[_clsname[j]] = []
            neuron_class[_clsname[j]].append(n)
        else:
            neuron_class[n] = [n]
    print(neuron_class)
    nn_new = nn_old.fold_network(neuron_class, exceptions=exceptions, self_loops=self_loops, data=data)
    return nn_new

# def foldByNeuronType(nn_old):
#     """
#     Folds neurons in the given neural network based on the neuron type.

#     Args:
#         nn_old (NeuralNetwork): The original neural network.

#     Returns:
#         NeuralNetwork: The folded neural network.

#     The function iterates over each neuron in the neural network and checks if the neuron type matches a specific pattern. If the neuron type matches the pattern, the function groups the neurons together based on the neuron number. The function then contracts the grouped neurons and updates the neuron dictionary.

#     Note:
#         The function assumes that the neuron type is represented by a sequence of numbers at the end of the neuron name.
#     """
#     nn_new = nn_old.copy()
#     for m in nn_old.neurons:
#         stripNums = m.rstrip('0123456789')
#         if (len(m) - len(stripNums)==1 and m[-1]=='1') or (len(m) - len(stripNums)==2 and m[-1]=='1' and m[-2]=='0'):
#             j=1
#             moreInClass=True
#             classNeurons = []
#             while (moreInClass):
#                 if len(m) - len(stripNums) == 1:
#                     if m[:len(stripNums)] + str(j+1) in nn_new.neurons:
#                         classNeurons.append(m[:len(stripNums)] + str(j+1))
#                         j+=1
#                         moreInClass=True
#                     else:
#                         moreInClass=False
#                 else:
#                     if m[:len(stripNums)] + "{:02d}".format(j+1) in nn_new.neurons:
#                         classNeurons.append(m[:len(stripNums)] + "{:02d}".format(j+1))
#                         j+=1
#                         moreInClass=True
#                     else:
#                         moreInClass=False
#             print(m, classNeurons)
#             for n in classNeurons:
#                 if not m[:len(stripNums)] in nn_new.neurons: 
#                     neuronPair = [m,n]
#                     nn_new.contractNeurons(neuronPair)
#                 else:
#                     print(m)
#             nn_new.neurons[m].name = m[:len(stripNums)]
#             nn_new.update_neurons()
#     return nn_new 

def foldDorsoVentral(nn_old):
    """
    This function performs a dorsoventral folding operation on a neural network.
    It takes the old neural network as input and produces a new neural network after the folding operation.
    Parameters:
    - nn_old: The original neural network to be folded dorsoventrally.
    Returns:
    - nn_new: The new neural network after the dorsoventral folding operation.
    """
    foldingDict = {}
    for m in nn_old.neurons:
        if m[-1] == 'D' and not m in ['RID']:
            n = m[:-1] + 'V'
            o = m[:-1]
            if n in nn_old.neurons:
                if o not in foldingDict:
                    foldingDict[o] = [m,n]
        elif m[-2] == 'D' and m[-1] in ['L', 'R']:
            n = m[:-2] + 'V' + m[-1]
            o = m[:-2] + m[-1] 
            if n in nn_old.neurons:
                if o not in foldingDict:
                    foldingDict[o] = [m,n]
    nn_new = nn_old.fold_network(foldingDict)
    # nn_new = nn_old.copy()
    # for m in nn_new.neurons:
    #     if m[-1] == 'D' and not m in ['RID']:
    #         n = m[:-1] + 'V'
    #         o = m[:-1] 
    #         if n in nn_new.neurons:
    #             if o in nn_new.neurons:
    #                 neuronPair1 = [m,n] # to contract D and V
    #                 neuronPair2 = [o,m] # to contract - and D
    #                 nn_new.contract_neurons(neuronPair1)
    #                 nn_new.contract_neurons(neuronPair2)
    #             else:
    #                 nn_new.neurons[m].name = m[:-1]
    #                 neuronPair1 = [m,n] # to contract D and V 
    #                 nn_new.contract_neurons(neuronPair1)
    # nn_new.update_neurons() 
    return nn_new
                

### Plotting functions

def simpleaxis(axes, every=False, outward=False):
    """
    Sets the axis style for the given axes.

    Parameters:
        axes (list or ndarray): The axes to set the style for.
        every (bool, optional): Set the style for every axis. Defaults to False.
        outward (bool, optional): Set the style outward. Defaults to False.

    Returns:
        None
    """
    if not isinstance(axes, (list, np.ndarray)):
        axes = [axes]
    for ax in axes:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        if (outward):
            ax.spines['bottom'].set_position(('outward', 10))
            ax.spines['left'].set_position(('outward', 10))
        if every:
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)
        ax.get_xaxis().tick_bottom()
        ax.get_yaxis().tick_left()
        ax.set_title('')


def groupPosition(pos, type_cat):

    x,y = zip(*pos.values())
    x = np.array(x)
    yax = sorted(set(y))[::-1]

    new_pos = {}
    for i,yi in enumerate(yax):
        new_pos[i] = [(xi,yi) for xi in sorted(x[np.where(y==yi)])]
    # print([(i,len(v)) for (i,v) in new_pos.items()])
    # print(type_cat)
    pos2 = {}
    boxes = {}
    order = ['sensory', 'interneuron', 'motorneuron']
    box_height = 0.025
    for i,o in enumerate(order):
        boxes[o] = {}
        j=0
        if o in type_cat:
            sortedKeys = sorted(type_cat[o].keys())
            if o == "interneuron":
                intOrder = ['layer 1 interneuron', 'layer 2 interneuron', 'layer 3 interneuron', 'category 4 interneuron', 'linker to pharynx', 'pharynx', 'endorgan']
                sortedKeys = [intn for intn in intOrder if intn in sortedKeys]
                # p = [sortedKeys[0]]
                # print(p, sortedKeys)
                # sortedKeys= sortedKeys[1:4] + p + [sortedKeys[4]]
                
            for cat in sortedKeys: 
                cat_pos = []
                # print(cat, len(new_pos[i]), len(type_cat[o][cat]))
                for n in sorted(type_cat[o][cat]):
                #print(n, cat, new_pos[i], i, j )
                    pos2[n] = new_pos[i][j]
                    cat_pos.append(pos2[n])
                    j+=1
                boxes[o][cat] = [(cat_pos[0][0]-0.035, cat_pos[0][1]-(box_height/2)), (cat_pos[-1][0] - cat_pos[0][0]) + 0.07, box_height]

    return pos2, boxes

def plot_spiral(neunet, save=False, figsize=(8,8), font_size=11):
    """
    Generates a spiral layout for the network.

    Parameters:
    - neunet (NeuralNetwork): The neural network object.

    Returns:
    - pos (dict): A dictionary mapping node names to their positions in the graph.
    """
    node_size = 800
    node_color = ['lightgray' for node in neunet.nodes]
    edge_color = []
    edge_weight = []
    for (u,v,attrib_dict) in list(neunet.edges.data()):
        if 'color' in attrib_dict:
            edge_color.append(attrib_dict['color'])
        else:
            edge_color.append('gray') 
        
        if 'weight' in attrib_dict:
            edge_weight.append(attrib_dict['weight'])
        else:
            edge_weight.append(1)

    pos = nx.spiral_layout(neunet, equidistant=True, resolution=0.5)

    fig, ax = plt.subplots(figsize=figsize)
    nx.draw(neunet, node_size = node_size, ax=ax, pos=pos, labels = {node: node.name for node in neunet.nodes}, with_labels=True, node_color = node_color, edge_color=edge_color, font_size=font_size)
    if save:
        plt.savefig(save)
    plt.show()
    plt.close()

def add_gapjn_symbols(ax, pos, edges, line_length=0.05, alpha=1, color='k'):
    for edge in edges:
        start, end = edge[0], edge[1]
        x1, y1 = pos[start]
        x2, y2 = pos[end]
         
        # Calculate the midpoint of the edge
        mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2

        # Calculate direction vector and normalize it
        direction = np.array([x2 - x1, y2 - y1])
        line_length = 0.02*np.linalg.norm(direction)
        direction = direction / np.linalg.norm(direction)

    
        # Calculate the perpendicular direction
        perp_direction = np.array([-direction[1], direction[0]])

        # Calculate the positions of the capacitor lines
        line_start1 = np.array([mid_x, mid_y]) - (line_length / 2) * perp_direction
        line_end1 = np.array([mid_x, mid_y]) + (line_length / 2) * perp_direction

        line_start2 = np.array([mid_x, mid_y]) - (line_length / 2) * perp_direction + direction * 0.02
        line_end2 = np.array([mid_x, mid_y]) + (line_length / 2) * perp_direction + direction * 0.02

        # Draw the capacitor lines
        ax.plot([line_start1[0], line_end1[0]], [line_start1[1], line_end1[1]], color=color, alpha=alpha, lw=2)
        ax.plot([line_start2[0], line_end2[0]], [line_start2[1], line_end2[1]], color=color, alpha=alpha, lw=2)

def add_neuropep_symbols(ax, pos, edges, node_size, color='k', crescent_radius=0.04, offset_factor=1.0):
    # Get node positions
    node_radius = np.sqrt(node_size) / 220
    for edge in edges:
        source, target = edge[0], edge[1]
        source_pos = np.array(pos[source])
        target_pos = np.array(pos[target])
        
        # Calculate direction vector
        direction = target_pos - source_pos
        norm = np.linalg.norm(direction)
        direction /= norm  # Normalize
        
        # Calculate the position for the overlapping circles
        outer_center = target_pos - (crescent_radius* 2 + node_radius)  * direction
        inner_center = outer_center - offset_factor * crescent_radius * direction
        
        # Draw the line
        #ax.plot([source_pos[0], target_pos[0]], [source_pos[1], target_pos[1]], color=color, zorder=1)
        
        # Draw the inner circle (white) to create crescent effect
        inner_circle = Circle(inner_center, crescent_radius, color=color, zorder=2)
        ax.add_patch(inner_circle)
        
        # Draw the outer circle (colored) to create crescent effect
        outer_circle = Circle(outer_center, crescent_radius, color='white', zorder=3)
        ax.add_patch(outer_circle)

def add_circular_arrowheads(ax, pos, edges, node_size, scale_factor=0.001, color='k'):
    for edge in edges:
        start, end = edge[0], edge[1]
        x1, y1 = pos[start]
        x2, y2 = pos[end]
        
        # # Calculate direction vector and normalize it
        # direction = np.array([x2 - x1, y2 - y1])
        # direction = direction / np.linalg.norm(direction)

        # # Calculate the radius of the circle
        # radius = np.sqrt(node_size) * scale_factor

        # # Calculate the position of the circle's center (slightly outside the endpoint)
        # circle_center = np.array([x2, y2]) - (radius + np.sqrt(node_size) * 0.05) * direction

        # # Draw the circle at the end of the arrow
        # circle = Circle(circle_center, radius, color=color, zorder=2)
        # ax.add_patch(circle)

        direction = np.array([x2 - x1, y2 - y1])
        direction = direction / np.linalg.norm(direction)

        # Calculate the radius of the circle
        radius = np.sqrt(node_size) * scale_factor

        # Adjust position to place the circle just outside the node
        node_radius = np.sqrt(node_size) / 2 * 0.01
        circle_center = np.array([x2, y2]) - (node_radius + radius) * direction

        # Draw the circle at the end of the arrow
        circle = Circle(circle_center, radius, color=color, zorder=2)
        ax.add_patch(circle)
            
def plot_shell(neunet, center=None, shells=None, save=False, figsize=(8,8), edge_color_dict=None, edge_alpha_dict=None,\
                node_color_dict=None, edge_labels=False, handles=None, fontsize=11, width_logbase=2, title=None):
    """
    Generates a shell layout for the network.

    Parameters:
    - neunet (NeuralNetwork): The neural network object.

    Returns:
    - pos (dict): A dictionary mapping node names to their positions in the graph.
    """
    MAX_LENGTH_FOR_STRAIGHT_TEXT = 20
    arc_rad = 0.1
    node_size = 800
    arrow_size=20
    rotation = 0 #45
        
    if shells is None:
        if isinstance(center, str):
            assert center in neunet.neurons
            shells = [[neunet.neurons[center]], [neu for nname,neu in neunet.neurons.items() if nname !=center]]
        elif isinstance(center, list):
            #print(center, neunet.neurons.keys(), [c in neunet.neurons.keys() for c in center])
            assert all([c in neunet.neurons.keys() or isinstance(c, cedne.Neuron) for c in center])
            new_center = [c if c in neunet.neurons.keys() else c.name for c in center]
            shells = [[neunet.neurons[c] for c in new_center], [neu for nname,neu in neunet.neurons.items() if nname not in new_center]]
        elif isinstance(center, cedne.Neuron):
            shells = [[neunet.neurons[center.name]], [neu for nname,neu in neunet.neurons.items() if nname != center.name]]
        else:
            raise ValueError("center must be a neuron name or a list of neuron names")
    if node_color_dict is None:
        node_color = ['lightgray' if not hasattr(node, 'color') else node.color for node in neunet.nodes]
    else:
        node_color = [node_color_dict[node] for node in neunet.nodes]

    if edge_color_dict is None:
        edge_color_dict = {edge:'gray' if not hasattr(edge, 'color') else edge.color for edge in neunet.edges}
    if edge_alpha_dict is None:
        edge_alpha_dict = {edge:1 if not hasattr(edge, 'alpha') else edge.alpha for edge in neunet.edges}

    if edge_labels:
        edge_labels = {edge: edge.weight for edge in neunet.edges}
    else:
        edge_labels = {edge: '' for edge in neunet.edges}
    # edge_color = []
    # edge_weight = []
    # for (u,v,attrib_dict) in list(neunet.edges.data()):
    #     edge_color.append(attrib_dict['color'])
    #     edge_weight.append(attrib_dict['weight'])

    pos = nx.shell_layout(neunet, nlist=shells, scale=2)
    fig, ax = plt.subplots(figsize=figsize, layout='constrained')
    ax.set_aspect('equal', 'datalim', anchor='C')
    # print(len(pos))
    # print([(n, pos[neu]) for (n,neu) in neunet.neurons.items()])
    # nx.draw(neunet, node_size = 800, ax=ax, pos=pos, with_labels=False, node_color = node_color, edge_color=edge_color)
    nx.draw_networkx_nodes(neunet, pos=pos, node_size = node_size, ax=ax, node_color = node_color)
    
    if len(neunet.connections.keys()):
        for edge,connection in neunet.connections.items():
            edge_color = edge_color_dict[edge]
            edge_alpha = edge_alpha_dict[edge]
            if connection.connection_type == 'chemical-synapse':
                if (edge[1], edge[0]) in neunet.edges():
                    rad = arc_rad
                else:
                    rad = 0.0
                width = 1 + np.log2(connection.weight)/np.log2(width_logbase)
                nx.draw_networkx_edges(neunet, pos=pos, edgelist=[edge], node_size=node_size, connectionstyle=f'arc3,rad={rad}', arrows=True, arrowstyle='-|>', arrowsize=arrow_size, edge_color=edge_color, alpha=edge_alpha, width=width, ax=ax)
                # add_circular_arrowheads(ax, pos, edges=[edge], node_size=node_size, color='k')
                # add_half_circle_arrowheads(ax, pos, edges=[edge], node_size=node_size, color='k')
            elif connection.connection_type == 'gap-junction':
                width = 1 + np.log2(connection.weight)/np.log2(width_logbase)
                nx.draw_networkx_edges(neunet, pos=pos, edgelist=[edge], node_size=node_size, connectionstyle='arc3,rad=0.0', arrowstyle='-', arrowsize=arrow_size, edge_color='k', alpha=edge_alpha, width=width, ax=ax)
                add_gapjn_symbols(ax, pos, edges=[edge], alpha=edge_alpha, color='k')
            else:
                width = 1 + np.log2(connection.weight)/np.log2(width_logbase)
                nx.draw_networkx_edges(neunet, pos=pos, edgelist=[edge], node_size=node_size, connectionstyle='arc3,rad=0.0', arrowstyle='-', arrowsize=arrow_size, edge_color='k', alpha=edge_alpha, width=width, ax=ax)
                add_neuropep_symbols(ax, pos, edges=[edge], node_size=node_size)
                #add_neuropep_symbols(ax, pos, edges=[edge], alpha=edge_alpha, color='k')
            if edge_labels:
                nx.draw_networkx_edge_labels(neunet, pos=pos, edge_labels=edge_labels, ax=ax)
    else:
        nx.draw_networkx_edges(neunet, pos=pos, node_size=node_size, arrows=True, arrowstyle='-|>', arrowsize=20, ax=ax)

    for node, (x, y) in pos.items():
        label = str(node.name)
        if len(label)>4 and len(pos)>MAX_LENGTH_FOR_STRAIGHT_TEXT:
            if (x>0 and y>0) or (x<0 and y<0):
                plt.text(x, y, label, fontsize=fontsize, rotation=rotation, ha='center', va='center')
            elif (x>0 and y<0) or (x<0 and y>0):
                plt.text(x, y, label, fontsize=fontsize, rotation=-rotation, ha='center', va='center')
            else:
                plt.text(x, y, label, fontsize=fontsize, ha='center', va='center')
        else:
            plt.text(x, y, label, fontsize=fontsize, ha='center', va='center')
    
    plt.axis('off')
    if handles:
        fig.legend(handles=handles,loc='outside center right')
    if save:
        if isinstance(save, str):
            plt.savefig(save)
        elif isinstance(save, bool):
            if not os.path.exists(OUTPUT_DIR):
                os.makedirs(OUTPUT_DIR)
            plt.savefig(OUTPUT_DIR + '_'.join([n.name for n in shells[0]]) + '.svg')
    #plt.show()
    if title:
        fig.suptitle(title)
    return fig
    
    
def plot_layered(interesting_conns, neunet, nodeColors=None, edgeColors = None, save=False, title='', extraNodes=[], extraEdges=[], pos=[], mark_anatomical=False, colorbar=False):
    """
    Generates a graph visualization of a given neural network. Change this function to make more streamlined.

    Parameters:
    - interesting_conns (list): A list of tuples representing the connections between nodes in the neural network.
    - edge_colors (list): A list of values representing the color of each edge in the graph.
    - node_colors (dict): A dictionary mapping node names to their corresponding colors.
    - title (str): The title of the graph.
    - neunet (NeuralNetwork): The neural network object.
    - save (bool, optional): Whether to save the graph as an image file. Defaults to False.
    - extra_nodes (list, optional): A list of additional nodes to include in the graph. Defaults to [].
    - extra_edges (list, optional): A list of additional edges to include in the graph. Defaults to [].
    - pos (dict, optional): A dictionary mapping node names to their positions in the graph. Defaults to [].
    - mark_anatomical (bool, optional): Whether to mark anatomically significant connections. Defaults to False.

    Returns:
    - pos (dict): A dictionary mapping node names to their positions in the graph.
    """
    classcolor = 'green'

    G = nx.Graph()
    G.add_edges_from(interesting_conns)
    if edgeColors is None:
        edge_color=[]
    elif isinstance(edgeColors, str):
        edge_color = [edgeColors for e in interesting_conns]
    elif len(edgeColors) == len(interesting_conns):
        if all (isinstance(ec, str) for ec in edgeColors):
            edge_color = [ec for ec in edgeColors]
        elif all (isinstance(ec, float) for ec in edgeColors):
            #cmap2 = sns.diverging_palette(220, 20, as_cmap=True)
            cmap = plt.get_cmap('PuOr')
            max_color = max(np.abs(edgeColors))
            norm = matplotlib.colors.Normalize(vmin=-max_color,vmax=max_color)
            m = cm.ScalarMappable(norm=norm, cmap=cmap)
            edge_color = [m.to_rgba(col) for col in edgeColors]
    else:
        raise ValueError('edge_colors must either be a string or a list of strings or floats \
            of the same length as interesting_conns')
    
    if nodeColors is None:
        node_color=[]
    elif isinstance(nodeColors, list):
        if all (isinstance(nc, str) for nc in nodeColors):
            node_color = [nc for nc in nodeColors]
    elif isinstance(nodeColors, dict):
        cmap2 = plt.get_cmap('RdYlGn')
        norm2 = matplotlib.colors.Normalize(vmin=-1.5,vmax=1.5)
        o = cm.ScalarMappable(norm=norm2, cmap=cmap2)

    if mark_anatomical:
        for i,e in enumerate(G.edges):
            u,v= e[0], e[1]
            if neunet.has_edge(neunet.neurons[u], neunet.neurons[v]) or neunet.has_edge(neunet.neurons[v], neunet.neurons[u]):
                if (neunet.neurons[u], neunet.neurons[v], 'gapjn') in neunet.edges or (neunet.neurons[v], neunet.neurons[u], 'gapjn') in neunet.edges:
                    edge_color[i] = 'k'
                else:
                    edge_color[i] = 'blue'

    edge_labels = {}
    neuronTypes = ['motorneuron', 'interneuron', 'sensory']
    node_color = {'sensory': [], 'interneuron':[], 'motorneuron':[]}
    node_alpha = {'sensory': [], 'interneuron':[], 'motorneuron':[]}

    edgeList_1, edgeList_2 = [], []
    colors_1, colors_2 = [], []
    node_shapes = {'sensory': 'o', 'interneuron':'s', 'motorneuron':'H'}
    node_types = {'sensory': [], 'interneuron':[], 'motorneuron':[]}
    #node_shapes = {'sensory': [], 'interneuron':[], 'motorneuron':[]}
    
    edges_within = []
    edges_across = []
    color_within = []
    color_across = []
    for e,c in zip(G.edges, edge_color):
        if neunet.neurons[e[0]].type == neunet.neurons[e[1]].type:
            edges_within.append(e)
            color_within.append(c)
        else:
            edges_across.append(e)
            color_across.append(c) 
            
    if len(extraEdges):
        G.add_edges_from(extraEdges)
    
    oriNodes = tuple(G.nodes)
    for n in extraNodes:
        if not n in G.nodes:
            G.add_node(n)

    # if not len(extraNodes):
    #     nodelist = G.nodes
    # else:
    #     nodelist = extraNodes
    categories = {} 
    for n in G.nodes:
        # print(n)
        if n in oriNodes:
            if n in nodeColors.keys():
                #print("Y", n, nodeColors.keys())
                if nodeColors is not None:
                    if isinstance(nodeColors[n], str):
                        node_color[neunet.neurons[n].type].append(nodeColors[n])
                    elif isinstance(nodeColors[n], float):
                        node_color[neunet.neurons[n].type].append(o.to_rgba(nodeColors[n]))
                    else:
                        raise ValueError('node_colors must either be a string or a list of strings or floats \
                            of the same length as nodelist')
                else:
                    node_color[neunet.neurons[n].type].append(nodeColors[n])
                    node_alpha[neunet.neurons[n].type].append(0.8)
            else:
                #print(n, nodeColors.keys())
                node_color[neunet.neurons[n].type].append('lightgray') 
                node_alpha[neunet.neurons[n].type].append(0.8)
            node_types[neunet.neurons[n].type].append(n)
        # else:
        #     node_color[nn.neurons[n].type].append("None")
        #     node_alpha[nn.neurons[n].type].append(None)
        categories[n] = (neunet.neurons[n].type, neunet.neurons[n].category)
        nx.set_node_attributes(G, {n:{"layer":neuronTypes.index(neunet.neurons[n].type)}})
        # print(n, neuronTypes.index(neunet.neurons[n].type))
        #G.nodes[n].layer = nn.neurons[n].type 
    print(G.nodes())
    cats = set(categories.values())
    type_cat = {}
    for n,cat in categories.items():
        cat_alt = cat[1]
        if not cat[0] in type_cat:
            type_cat[cat[0]] = {cat_alt:[n]}
        else:
            if not cat_alt in type_cat[cat[0]]:
                type_cat[cat[0]][cat_alt] = [n]
            else:
                type_cat[cat[0]][cat_alt].append(n) 

    array_op = lambda x, sx: np.array([x[0]*sx, x[1]])
    sx=2
    #pos =  nx.spring_layout(G, k=1, iterations=70)# 
    
    if not len(pos):
        pos =  nx.multipartite_layout(G,subset_key="layer", align='horizontal')
        # print(pos)
        pos = {p:array_op(pos[p],sx) for p in pos}
        print(pos)
        pos, boxes = groupPosition(pos, type_cat)

    #print(edges_within, edges_across)
    fig, ax = plt.subplots(figsize=(40,4))
    #edge_color = 'k'
    #nx.draw(G, node_size = 1000, ax=ax, pos=pos, with_labels=True, edge_color=edge_color, node_color = node_color)#, width=weights, font_size=12)
    #plotNodes = []
    node_size=  2000
    for type in node_types:
        nx.draw_networkx_nodes(node_types[type], node_shape=node_shapes[type], node_size = node_size, ax=ax, pos=pos, node_color = node_color[type],  linewidths=1, alpha=0.8)#node_alpha[type]))#, width=weights, font_size=12)
    for node, (x, y) in pos.items():
        label = str(node)
        if len(label)>3:
            plt.text(x, y, label, fontsize=20, rotation=45, ha='center', va='center')
        else:
            plt.text(x, y, label, fontsize=20, ha='center', va='center')

    # nx.draw_networkx_labels(G, pos=pos, labels={o:o for o in oriNodes if len(o)<4}, ax=ax, font_size=20)
    # nx.draw_networkx_labels(G, pos=pos, labels={o:o for o in oriNodes if len(o)>3}, ax=ax, font_size=20, rotation=45)
    #nx.draw_networkx_edges(G, edgelist=interesting_conns, ax=ax, pos=pos, edge_color=edge_color, node_size = node_size, arrows=True, connectionstyle='arc3, rad=0.3', width=2)#colors_2, style='--')
    nx.draw_networkx_edges(G, edgelist=edges_within, ax=ax, pos=pos, edge_color=color_within, node_size = node_size, width=2, arrows=True, connectionstyle='arc3, rad=0.3')#colors_2, style='--')
    nx.draw_networkx_edges(G, edgelist=edges_across, ax=ax, pos=pos, edge_color=color_across, node_size = node_size, width=2, arrows=True)#, connectionstyle='arc3, rad=0.3')#colors_2, style='--')

    within_extra = []
    across_extra = []
    for e in extraEdges:
        if neunet.neurons[e[0]].type == neunet.neurons[e[1]].type:
            within_extra.append(e)
        else:
            across_extra.append(e)
    
    nx.draw_networkx_edges(G, edgelist = within_extra, ax=ax, edge_color='gray', pos=pos, alpha=0.2, width=1, arrows=True, connectionstyle='arc3, rad=0.3')
    nx.draw_networkx_edges(G, edgelist = across_extra, ax=ax, edge_color='gray', pos=pos, alpha=0.2, width=1)
    #nx.draw_networkx_edge_labels(G, pos, label_pos=0.5, edge_labels=edge_labels, ax=ax)
    if colorbar:
        divider = make_axes_locatable(ax)
        
        # cax = inset_axes(ax, loc='upper right', width="100%", height="100%",
        #                bbox_to_anchor=(0.75,0.85,.05,.05), bbox_transform=ax.transAxes) #
        # cax.set_xticks([-1,0,1])
        # cax.set_title("$\\Delta$ correlation", fontsize=24)
        # #cax = divider.append_axes("right", size="1%", pad=0.05)    
        # cbar = plt.colorbar(m, cax= cax, orientation='horizontal')
        # cbar.ax.tick_params(labelsize=24) 

        cax2 = divider.append_axes("bottom", size="2%", pad=0.1)
        cbar2= plt.colorbar(o, cax=cax2, orientation='horizontal')
        cbar2.ax.tick_params(labelsize='xx-large') 

    for types in boxes:
        for cat, pars in boxes[types].items():
            xy, width, height = pars[0], pars[1], pars[2]
            rect = matplotlib.patches.Rectangle(xy,width,height, linewidth=1, edgecolor='gray', facecolor='none', alpha=0.75, linestyle='dashed')
            if cat in ['linker to pharynx']:
                cat = "link. to phar."
                cat = cat.replace(' ', '\n')
            elif cat in ['sex-specific neuron']:
                cat = 'sex-spec'
                #cat = cat.replace(' ', '\n')]
            for suffix in ["interneuron", "motor neuron"]:
                if suffix in cat:
                    cat = cat.replace(suffix,'')
            ax.text(xy[0], xy[1]-0.002, cat, #-height
            horizontalalignment='left',
            verticalalignment='top', fontsize=28, color=classcolor)
            # Add the patch to the Axes
            ax.add_patch(rect)
    simpleaxis(ax, every=True)
    ax.set_title(title, y=1.01, fontsize=40)
    ax.set_xticks([])
    ax.set_yticks([])
    # plt.colorbar(m)
    plt.tight_layout()
    plt.subplots_adjust(top=2)
    if not save==False:
        plt.savefig(save)
    
    plt.show()
    plt.close()
    
    return pos

def is_left_neuron(n):
    ''' Returns if a neuron is a left neuron. This works for Worms only for now.'''
    if n[-1] == 'L' and n not in ['ADL', 'AVL']:
        return True
            
def plot_position(nn, axis='AP-DV', highlight=None, booleanDictionary=None, title='', label='all', save=False, figsize=(20,3), limit=None):
    """
    A function to plot the positions of neurons based on their coordinates and attributes.

    Parameters:
        nn: NeuronNetwork object representing the neurons to be plotted.
        axis: String, the orientation of the plot (default: 'AP-DV').
        special: List of special neurons to highlight.
        booleanDictionary: Dictionary mapping boolean values to colors.
        title: String, the title of the plot.
        save: Boolean, whether to save the plot as an image.

    Returns:
        None
    """

    if booleanDictionary is None:
        active_color, passive_color = "#CC5500", "lightgrey"
        booleanDictionary = {True: active_color, False: passive_color}

    coords = ['AP', 'DV', 'LR']
    nlabels = np.array([n for n in nn.neurons])
    pos = [[] for i in range(len(nlabels))]
    if highlight is None:
        highlight = []
    elif isinstance(highlight, list):
        if isinstance(highlight[0], str):
            assert all([n in nn.neurons for n in highlight]), "Neuron(s) not found in the network."
        elif isinstance(highlight[0], list):
            assert all([n in nn.neurons for j in range(len(highlight)) for n in highlight[j]]), "Neuron(s) not found in the network."
        else:
            raise ValueError("Highlight must be a list of strings or a list of lists.")
            
    for i,n in enumerate(nlabels):
        if n in ['CANL']:
            pos[i] = [510, 20, -1]
        elif n in ['CANR']:
            pos[i] = [490, 20, 1]
        else:
            for x in coords:
                if x in nn.neurons[n].position: 
                    pos[i].append(nn.neurons[n].position[x])
                else:
                    raise ValueError(f"Neuron {n} has no position in axis {x}.")
    pos = np.array(pos)

    if limit:
        # limit should be a dict like {'AP': (0, 150)} or {'DV': (-50, 50), 'AP': (0, 200)}
        mask = np.ones(len(pos), dtype=bool)
        for coord, (low, high) in limit.items():
            axis_idx = coords.index(coord)
            mask &= (pos[:, axis_idx] >= low) & (pos[:, axis_idx] <= high)
        # Filter positions and labels
        pos = pos[mask]
        nlabels = nlabels[mask]

    posDict = dict(zip(coords, pos.T))
    posDict['LR'] = -posDict['LR'] # flipping LR for plotting in the correct direction
    posDict['DV'] = -posDict['DV'] # flipping DV for plotting in the correct direction

    xax, yax = axis.split('-')
    if xax in coords:
        x = posDict[xax]
    elif xax[::-1] in coords:
        x = -posDict[xax[::-1]]
    else:
        raise ValueError(f"Invalid axis: {xax}, use some combination of {coords}")

    if yax in coords:
        y = posDict[yax]
    elif yax[::-1] in coords:
        y = -posDict[yax[::-1]]
    else:
        raise ValueError(f"Invalid axis: {yax}, use some combination of {coords}")
    
    x = np.array(x)
    y = np.array(y)

    if isinstance(highlight[0], str):
        f,ax = plt.subplots(figsize=figsize, dpi=300, layout='constrained')
        plt.axis('off')
        facecolors = np.array([booleanDictionary[n in highlight] for n in nlabels])
        alphas = np.array([1 if n in highlight else 0.25 for n in nlabels])
        boolList = np.array([n in highlight for n in nlabels])
        #print([(n,facecolors[j]) for j,n in enumerate(nlabels) if n in highlight])
        ax.scatter(x[~boolList], y[~boolList], s=200, facecolor=facecolors[~boolList], edgecolor=facecolors[~boolList], alpha=alphas[~boolList], zorder=1)
        ax.scatter(x[boolList], y[boolList], s=200, facecolor=facecolors[boolList], edgecolor=facecolors[boolList], alpha=alphas[boolList], zorder=2)
        if label:
            ta.allocate_text(f,ax,x[boolList], y[boolList],
                        nlabels[boolList],
                        x_scatter=x[boolList], y_scatter=y[boolList],
                        textsize='xx-large', linecolor='gray', avoid_label_lines_overlap=True)

    elif isinstance(highlight[0], list):
        buffer = 1.1
        f,ax = plt.subplots( dpi=300, layout='constrained',figsize=figsize) #figsize=(2,0.3), dpi=300, layout='constrained'
        plt.axis('off') 
        plt.xlim(min(x)*buffer,max(x)*buffer)
        plt.ylim(min(y)*buffer,max(y)*buffer)
        # ax.set_aspect('equal')
        trans=ax.transData.transform
        trans2=f.transFigure.inverted().transform

        facecolors = cmr.take_cmap_colors('cmr.tropical', len(highlight), cmap_range=(0.15, 0.85), return_fmt='hex')
        color_dict = {n:[] for n in nlabels}
        alpha_dict = {n:0.25 for n in nlabels}
        boolList = np.array([False]*len(nlabels))
        for i,n in enumerate(nlabels):
            for j,hlc in enumerate(highlight):
                if n in hlc:
                    color_dict[n].append(facecolors[j])
                    alpha_dict[n] = 1
                    if label == 'all':
                        boolList[i] = True
                    elif label == 'left' and is_left_neuron(n):
                        boolList[i] = True
            if len(color_dict[n]) == 0:
                color_dict[n].append("lightgrey")
        piesize=0.3
        if not label == 'none':
            # if isinstance(label, int):
            #     randnum = np.random.default_rng()
            #     boolList = randnum.choice(boolList, size=label, replace=False)
            ta.allocate_text(f,ax,x[boolList], y[boolList],
                        nlabels[boolList],
                        x_scatter=x[boolList], y_scatter=y[boolList],
                        textsize='xx-large', linecolor='gray', avoid_label_lines_overlap=True)
        # print(boolList, ~boolList, x[boolList])
        # ax.scatter(x[~boolList], y[~boolList], s=200)
        # ax.scatter(x[boolList], y[boolList], s=200)
        for i,n in enumerate(nlabels):
            xx,yy=trans((x[i],y[i])) # figure coordinates
            xa,ya=trans2((xx,yy)) # axes coordinates
            #plot_pie(n, (x[i], y[i]), a, color_dict, alpha_dict, piesize)
            a = plt.axes([xa-piesize/2,ya-piesize/2, piesize, piesize])
            a.set_aspect('equal')
            plot_pie(n, (0,0), a, color_dict, alpha_dict, piesize=piesize)
        #     ax.pie([1/len(color_dict[n])]*len(color_dict[n]), center = (x[i], y[i]), colors=color_dict[n], radius = piesize, counterclock=False, wedgeprops={'alpha': alpha_dict[n], 'zorder': len(color_dict[n])})


    #Add the legend to the plot
    legend_ax = f.add_axes([0.95, 0.1, 0.1, 0.1])
    # legend_ax.set_xlim(0, 1)
    # legend_ax.set_ylim(0, 1)
    legend_ax.set_xlabel(xax, fontsize="x-large")
    legend_ax.set_ylabel(yax, fontsize="x-large")
    legend_ax.set_xticks([])
    legend_ax.set_yticks([])
    legend_ax.set_aspect('equal')
    simpleaxis(legend_ax)
    ax.set_title(title, fontsize="xx-large")
    # f.draw_without_rendering()
    #ax.set_xlim((-0.9,-0.))
    if save:
        plt.savefig(save)
    plt.show()

def plot_pie(n, center, ax, color_dict, alpha_dict, pie_division = None, piesize=1): 
    # radius for pieplot size on a scatterplot
    if not pie_division:
        pie_division = [1/len(color_dict[n])]*len(color_dict[n])
    ax.pie(pie_division, center = center, colors=color_dict[n], radius = piesize, counterclock=False, wedgeprops={'alpha': alpha_dict[n], 'zorder': len(color_dict[n])})

class Annotation3D(Annotation):
    '''Annotate the point xyz with text s'''

    def __init__(self, s, xyz, *args, **kwargs):
        Annotation.__init__(self,s, xy=(0,0), *args, **kwargs)
        self._verts3d = xyz

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.xy=(xs,ys)
        Annotation.draw(self, renderer)

def annotate3D(ax, s, *args, **kwargs):
    '''add anotation text s to to Axes3d ax'''

    tag = Annotation3D(s, *args, **kwargs)
    ax.add_artist(tag)
    return tag

def plot_position_3D(nn, highlight=None, booleanDictionary=None, title='', label=True, save=False):
    """
    A function to plot the positions of neurons based on their coordinates and attributes.

    Parameters:
        nn: NeuronNetwork object representing the neurons to be plotted.
        axis: String, the orientation of the plot (default: 'AP-DV').
        special: List of special neurons to highlight.
        booleanDictionary: Dictionary mapping boolean values to colors.
        title: String, the title of the plot.
        save: Boolean, whether to save the plot as an image.

    Returns:
        None
    """
    coords = ['AP', 'DV', 'LR']
    if booleanDictionary is None:
        active_color, passive_color = "#CC5500", "lightgrey"
        booleanDictionary = {True: active_color, False: passive_color}
    nlabels = np.array([n for n in nn.neurons])
    pos = [[] for i in range(len(nlabels))]
    if highlight is None:
        highlight = []
    else:
        assert all([n in nn.neurons for n in highlight]), "Neuron(s) not found in the network."
            
    for i,n in enumerate(nlabels):
        if n in ['CANL']:
            pos[i] = [510, 20, -1]
        elif n in ['CANR']:
            pos[i] = [490, 20, 1]
        else:
            for x in coords:
                if x in nn.neurons[n].position: 
                    pos[i].append(nn.neurons[n].position[x])
                else:
                    raise ValueError(f"Neuron {n} has no position in axis {x}.")
    pos = np.array(pos)
    posDict = dict(zip(coords, pos.T))
    posDict['LR'] = -posDict['LR'] # flipping LR for plotting in the correct direction
    posDict['DV'] = -posDict['DV'] # flipping DV for plotting in the correct direction
    facecolors = np.array([booleanDictionary[n in highlight] for n in nlabels])
    alphas = np.array([1 if n in highlight else 0.25 for n in nlabels])
    boolList = np.array([n in highlight for n in nlabels])

    xax, yax, zax = coords
    x = np.array(posDict[xax])
    y = np.array(posDict[yax])
    z = np.array(posDict[zax])
    
    xscale =np.max(x)-np.min(x)
    yscale =np.max(y)-np.min(y)

    # Create the figure and GridSpec
    f = plt.figure(figsize=(xscale,yscale), layout='constrained')
    gs = GridSpec(10, 10, figure=f)

    # Create the 3D subplots using GridSpec
    ax = f.add_subplot(gs[:8,:], projection='3d')
    ax.set_box_aspect(aspect = (10,2,1))
    plt.axis('off') 
    ax_leg = f.add_subplot(gs[8,8], projection='3d')
    ax_leg.grid(visible=False, which='both', axis='both')
    ax_leg.set_xticks([])
    ax_leg.set_yticks([])
    ax_leg.set_zticks([])

    ax_leg.set_aspect('equal')
    sc_bg = ax.scatter(x[~boolList], y[~boolList], z[~boolList], s=25, facecolor=facecolors[~boolList], edgecolor=facecolors[~boolList], alpha=alphas[~boolList], zorder=1)
    sc_fg = ax.scatter(x[boolList], y[boolList], z[boolList], s=25, facecolor=facecolors[boolList], edgecolor=facecolors[boolList], alpha=alphas[boolList], zorder=2)

    arrowprops = dict(
    arrowstyle="-",
    #connectionstyle="angle,angleA=0,angleB=90,rad=10"
    )
    annot_bg = {}
    for i,n in enumerate(nlabels):
        lr_sign = 1 if z[i]>0 else -1
        if n in highlight:
            annotate3D(ax, s=n, xyz=(x[i], y[i], z[i]), fontsize=10, xytext=(np.random.uniform(-30,30),lr_sign*30),
               textcoords='offset points', ha='right',va='bottom',arrowprops=arrowprops) 
        else:
            annot_bg[i] = annotate3D(ax, s=n, xyz=(x[i], y[i], z[i]), fontsize=10, xytext=(np.random.uniform(-30,30),lr_sign*30),
               textcoords='offset points', ha='right',va='bottom',arrowprops=arrowprops)
            annot_bg[i].set_visible(False)

    ax_leg.set_xlabel(coords[0])
    ax_leg.set_ylabel(coords[1])
    ax_leg.set_zlabel(coords[2])

    ax.set_xlim3d(np.min(x),np.max(x))
    ax.set_ylim3d(np.min(y), np.max(y))
    ax.set_zlim3d(np.min(z), np.max(z))
    ax.set_title(title, fontsize="xx-large")

    ax.shareview(ax_leg)

    # def update_annot(ind):
    #     #pos_xy = sc_bg.get_offsets()[ind["ind"][0]]
    #     #print(pos_xy)
    #     xs, ys, zs = sc_bg._offsets3d
    #     vxs, vys, vzs = proj_transform(xs, ys, zs, ax.get_proj())
    #     sorted_z_indices = np.argsort(vzs)[::-1]
    #     for j in sorted_z_indices[ind["ind"]]:
    #         annot_bg[j].set_visible(True)
    #     #annot_bg.set_text(nlabels[sorted_z_indices[ind['ind'][0]]])

    #     #annot_bg.xy = pos_xy
    #     #print(ind["ind"])
    #     #text = "{}, {}".format(" ".join(list(map(str,ind["ind"]))), 
    #                         #" ".join([nlabels[n] for n in ind["ind"]]))
    #     #text = ", ".join([nlabels[n] for n in ind["ind"]]) #nlabels[n]

    #     #annot_bg.set_text(text)
    #     #annot_bg.get_bbox_patch().set_facecolor('lightgrey')
    #     #annot_bg.get_bbox_patch().set_alpha(0.4)
    

    # def hover(event):
    #     #vis = annot_bg.get_visible()
    #     if event.inaxes == ax:
    #         cont, ind = sc_bg.contains(event)
    #         if cont:
    #             update_annot(ind)
    #             # for n in ind["ind"]:
    #             #     annot_bg[n].set_visible(True)
    #             f.canvas.draw_idle()
    #         else:
    #             #if vis:
    #             for n in ind["ind"]:
    #                 annot_bg[n].set_visible(False)
    #             f.canvas.draw_idle()

    # f.canvas.mpl_connect("motion_notify_event", hover)

    if save:
        plt.savefig(save)
    plt.show()

def make_hypermotifs(motif, length, join_at):
    ''' 
    Makes hypermotifs from given set of 3-node graph motifs
     
    Motifs must have integers as node names starting from 1.
    '''

    assert all([isinstance(n, int) for n in motif.nodes]), "All nodes must have integer node names"
    assert sorted(motif.nodes) == [*range(1,len(motif.nodes)+1)], "Nodes must be numbered 1 through number of nodes in the motif."
    assert isinstance(motif, nx.classes.digraph.DiGraph)

    motif_set = [motif.copy() for _ in range(length)]
    hypermotif = nx.union_all(motif_set, rename=(f'{j+1}.' for j in range(length)))

    join_indices = [(f'{l}.{j[0]}', f'{l+1}.{j[1]}') for j in join_at for l in range(1,length)]
    join_indices_copy = join_indices[:]
    copy_join = []
    if len(join_indices):
        _, right_ori = list(zip(*join_indices_copy))
        assert len(set(right_ori)) == len(right_ori), "Trying to contract one node to two different nodes"
        
        while len(copy_join)<len(join_indices): 
            left, right = list(zip(*join_indices_copy))
            ind_copy = [j for j,r in enumerate(right) if r not in left]
            ind = [right_ori.index(right[i]) for i in ind_copy]
            copy_join +=ind
            for j in ind_copy[::-1]:
                join_indices_copy.pop(j)
        
        mapping = {}
        for j in copy_join:
            ja = join_indices[j]
            nx.contracted_nodes(hypermotif, ja[0], ja[1], copy=False)
            mapping.update({f'{ja[0]}': f'{ja[0]}-{ja[1]}'})
        nx.relabel_nodes(hypermotif, mapping, copy=False)
    return hypermotif

def return_triads():
    triads = (
    "003",
    "012",
    "102",
    "021D",
    "021U",
    "021C",
    "111D",
    "111U",
    "030T",
    "030C",
    "201",
    "120D",
    "120U",
    "120C",
    "210",
    "300",
    )

    triad_graphs = {t:nx.triad_graph(t) for t in triads}
    for t in triad_graphs:
        triad_graphs[t] = nx.relabel_nodes(triad_graphs[t], mapping={'a':1, 'b':2, 'c':3}, copy=True)
    return triad_graphs
        
def randomize_graph(G, seed=None, mode='edge-swap', multiplier='auto'):
    g_copy = copy.deepcopy(G)
    if multiplier == 'auto':
        multiplier = int(np.log(len(G.edges)))
    else:
        if not isinstance(multiplier, int):
            raise ValueError("Multiplier must be an integer")
    if mode == 'edge-swap':
        nswap = int(multiplier*len(G.edges))
        nx.directed_edge_swap(g_copy, nswap=nswap, max_tries=nswap*100, seed=seed)
    elif mode == 'configuration-model':
        nodes = g_copy.nodes()
        in_degree = [g_copy.in_degree(n) for n in nodes]
        out_degree = [g_copy.out_degree(n) for n in nodes]
        g_copy = nx.directed_configuration_model(in_degree, out_degree, seed=seed)
    elif mode == 'num-nodes-edges':
        g_copy = nx.gnm_random_graph(len(g_copy.nodes()), len(g_copy.edges()), seed=seed, directed=True)
    return g_copy

def addBranch():
    ''' Add parallel and serial branches to a graph.'''


def plot_simulation_results(results, twinx=True):
    rate_model, inputs, rates = results
    f, ax = plt.subplots(figsize=(2.5, 2.5), layout='constrained')
    for node in rates:
        ax.plot(rate_model.time_points, rates[node], label=node.name, lw=2)
    # ax.set_ylim((-15,15))

    if twinx:
        ax2 = ax.twinx()
    else:
        ax2=ax
    for inp in inputs:
        ax2.plot(rate_model.time_points, [inp.process_input(t) for t in rate_model.time_points], ls='--', color='k')
    ax2.axhline(y=0, ls='--', alpha=0.25, color='gray')
    # ax2.set_ylim((-4,4))
    for inp in inputs:
        if isinstance(inp, simulator.StepInput):
            ax2.axvline(x=inp.tstart, ls='--', color='gray', alpha=0.25)
            ax2.axvline(x=inp.tend, ls='--', color='gray', alpha=0.25)
    # ax.set_ylim((-1,1))
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Rate')
    # ax.set_xticks([0,30,60,90])
    simpleaxis(ax)
    f.legend(loc='outside upper center', ncol=len(rates), frameon=False)
    return f

def compare_simulation_results(results1, results2, twinx=True):
    rate_model, inputs, rates1 = results1
    _, _, rates2 = results2
    f, ax = plt.subplots(figsize=(2.5, 2.5), layout='constrained')
    colors = {node: plt.cm.viridis(i/len(rates1)) for i, node in enumerate(rates1)}
    for node in rates1:
        ax.plot(rate_model.time_points, rates1[node], label=node.label or node, lw=2, color=colors[node])
        ax.plot(rate_model.time_points, rates2[node], label=node.label or node, lw=2, color=colors[node], ls='--')
    if twinx:
        ax2 = ax.twinx()
    else:
        ax2=ax
    for inp in inputs:
        ax2.plot(rate_model.time_points, [inp.process_input(t) for t in rate_model.time_points], ls='--', color='k')
    ax2.axhline(y=0, ls='--', alpha=0.25, color='gray')
    # ax2.set_ylim((-4,4))
    for inp in inputs:
        if isinstance(inp, simulator.StepInput):
            ax2.axvline(x=inp.tstart, ls='--', color='gray', alpha=0.25)
            ax2.axvline(x=inp.tend, ls='--', color='gray', alpha=0.25)
    # ax.set_ylim((-1,1))
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Rate')
    # ax.set_xticks([0,30,60,90])
    simpleaxis(ax)
    f.legend(loc='outside upper center', ncol=len(rates1), frameon=False)
    return f

def hierarchical_alignment(conns):
    ntype = ['sensory', 'interneuron', 'motorneuron']
    ntype_pairs = [(n1, n2) for n1 in ntype for n2 in ntype]
    conn_types = {n:0 for n in ntype_pairs}
    for edge in conns:
        if (edge[0].type in ntype) and (edge[1].type in ntype):
            conn = (edge[0].type, edge[1].type)
            conn_types[conn] +=1
    feedforward =  conn_types[('sensory', 'interneuron')] + conn_types[('sensory', 'motorneuron')] + conn_types[('interneuron', 'motorneuron')] 
    feedback = conn_types[('interneuron', 'sensory')] + conn_types[('motorneuron', 'interneuron')] + conn_types[('motorneuron', 'sensory')]
    lateral = 0#conn_types[('sensory', 'sensory')] + conn_types[('interneuron', 'interneuron')] + conn_types[('motorneuron', 'motorneuron')]
    print(feedforward, feedback)
    #return (feedforward+lateral)/(feedforward+lateral+feedback) if (feedforward+lateral+feedback) else 0
    return (feedforward-feedback)/(feedforward+feedback+lateral) if (feedforward+feedback+lateral) else 0