"""
Loader utilities for CeDNe

Includes functions to initialize nervous systems, load connectome data, 
neurotransmitters, neuropeptides, transcriptomes, and other biological properties.
"""

__author__ = "Sahil Moza"
__date__ = "2025-04-06"
__license__ = "MIT"

import os
import pickle
import numpy as np
import pandas as pd
import warnings
from cedne import Worm, Fly, NervousSystem
from .config import *
warnings.filterwarnings("ignore", category=UserWarning, module='openpyxl')

def makeWorm(name='', import_parameters=None, chem_only=False, gapjn_only=False):
    ''' Utility function to make a Worm based on import parameters.'''
    if import_parameters is None or (import_parameters['style'] == 'cook' and import_parameters['sex'] == 'hermaphrodite'):
        w = Worm(name)
        nn = NervousSystem(w)
        build_nervous_system(nn, neuron_data=cell_list, \
                            chem_synapses=chemsyns, \
                            elec_synapses=elecsyns, \
                            positions=neuronPositions, \
                            chem_only=chem_only, \
                            gapjn_only=gapjn_only)
    elif (import_parameters['style'] == 'cook' and import_parameters['sex'] == 'male'):
        w = Worm(name)
        nn = NervousSystem(w)
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

            w = Worm(name=name, stage=import_parameters['stage'])
            nn = NervousSystem(w, network= '_'.join([import_parameters['style'],import_parameters['stage'], import_parameters['dataset_ind']]))
            nn.create_neurons(labels=labels)
            witvliet_input.rename(columns={'synapses': 'weight'}, inplace=True)
            fin_input = witvliet_input[witvliet_input['pre'].isin(labels)]
            fin_input = fin_input[fin_input['post'].isin(labels)]
            for iter, conn in fin_input.iterrows():
                nn.setup_connections(conn, conn['type'], input_type='edge')
                

    return w

def makeFly(name = ''):
    f = Fly(name)
    nn = NervousSystem(f)

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
            if type(w)==Worm:
                print(nprc, model, models_dict[nprc])
                nn_np = NervousSystem(w, network="{}".format(nprc))
                nn_np.build_network(neuron_data=cell_list, adj=models_dict[nprc], label=nprc)
            elif type(w)==NervousSystem:
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