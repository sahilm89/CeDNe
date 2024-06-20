## File containing utility functions for CeDNe

## Dependencies

import numpy as np
import pandas as pd
from . import cedne
import matplotlib.pyplot as plt
import networkx as nx
import matplotlib.cm as cm
import matplotlib.colors as colors
import matplotlib
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import textalloc as ta

## Directories

TOPDIR = '/Users/sahilmoza/Documents/Codes/CedNe/' ## Change this to cedne and write a function to download data from an online server for heavy data.
DATADIR = TOPDIR + 'data_sources/'
DOWNLOAD_DIR = TOPDIR + 'data_sources/downloads/'

## Loading and building functions
cell_list = DATADIR + "Cell_list.pkl"
chemsyns = DATADIR + "chem_adj.pkl"
elecsyns = DATADIR + "gapjn_symm_adj.pkl"
neuronPositions = DATADIR + "neuronPosition.pkl"
def makeWorm(name=''):
    w = cedne.Worm(name)
    nn = cedne.NervousSystem(w)
    nn.build_nervous_system(neuron_data=cell_list, chem_synapses=chemsyns, elec_synapses=elecsyns, positions=neuronPositions)
    return w

## Neurotransmitter tables
prefix_NT = 'Wang_2019/'
npr_file = DOWNLOAD_DIR + prefix_NT + 'GenesExpressing-BATCH-thrs4_use.xlsx'
lig_file = DOWNLOAD_DIR + prefix_NT +'ligand-table.xlsx'
npr = pd.read_excel(npr_file, sheet_name='npr', true_values='TRUE', false_values='FALSE', engine='openpyxl')
ligmap = pd.read_excel(npr_file, sheet_name='ligmap', engine='openpyxl')
ligtable = pd.read_excel(lig_file, sheet_name='Hermaphrodite, sorted by neuron', skiprows=7, engine='openpyxl')

suffixes = ['', 'D', 'V', 'L', 'R', 'DL', 'DR', 'VL', 'VR', '01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13']
present = False

def getLigands(ligtable, neuron):
    ''' Returns ligand for each neuron'''
    a,b = ligtable['Neurotransmitter 1'][ligtable['Neuron']==neuron].to_list(), ligtable['Neurotransmitter 2'][ligtable['Neuron']==neuron].to_list()
    if len(b) and type(b[0])==str:
        return [a[0],b[0]]
    else:
        return [a[0]]

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


def loadNeurotransmitters(nn, npr=npr, ligmap=ligmap):
    ''' Loads Neurotransmitters into neurons using Wang et al 2024'''
    for n in nn.neurons:
        neuron = nn.neurons[n]
        if not hasattr(neuron, 'preSynapse'):
            nn.neurons[n].preSynapse = []
        if not hasattr(neuron, 'postSynapse'):
            nn.neurons[n].postSynapse = {}

    for col in npr.columns:
        for suffix in suffixes:
            if col + suffix in nn.neurons:
                nn.neurons[col + suffix].postSynapse.update(getLigandsAndReceptors(npr, ligmap, col))
                #present = True
    for n in nn.neurons:
        nn.neurons[n].preSynapse += getLigands(ligtable, n)

## Neuropeptides tables
prefix_NP = 'Ripoll-Sanchez_2023/'
csvfile = DOWNLOAD_DIR + prefix_NP + 'neuropeptideConnectome.txt'
lrm = DOWNLOAD_DIR +  prefix_NP + 'NPP_GPCR_networks_long_range_model_2.csv'
nid = DOWNLOAD_DIR +  prefix_NP + '26012022_num_neuronID.txt'
np_order = DOWNLOAD_DIR +  prefix_NP + '91-NPPGPCR networks'

def loadNeuropeptides(w, lrm=lrm, nid=nid, np_order=np_order, neuropeps= 'all'):
    ''' Loads Neuropeptides into neurons using Ripoll-Sanchez et al. 2023'''
    model = pd.read_csv(lrm,encoding= 'unicode_escape')
    neuronID = pd.read_csv(nid,encoding= 'unicode_escape', sep='\t', index_col=0, names=['NID', "Neuron"]) 
    neuropep_rec = pd.read_csv(np_order, sep=',', index_col=0)
    nidList = np.array(neuronID['Neuron'].to_list())

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
                nn_np.build_network(neurons=cell_list, adj=models_dict[nprc], label=nprc)
            elif type(w)==cedne.NervousSystem:
                w.setup_connections(adjacency_matrix=models_dict[nprc], edge_type=nprc)


## Load CENGEN tables
prefix_CENGEN = 'CENGEN/'
thres_1 = DOWNLOAD_DIR + prefix_CENGEN + 'liberal_threshold1.csv'
thres_2 = DOWNLOAD_DIR + prefix_CENGEN + 'medium_threshold2.csv'
thres_3 = DOWNLOAD_DIR + prefix_CENGEN + 'conservative_threshold3.csv'
thres_4 = DOWNLOAD_DIR + prefix_CENGEN + 'stringent_threshold4.csv'

def returnThresholdDict(th1, th2, th3, th4, nnames, cengen_neurons):
    """
    Generate a dictionary of thresholds using CENGEN levels of sensitivity.

    Args:
        th1 (dict): A dictionary mapping neuron names to their corresponding CENGEN threshold values for level 1.
        th2 (dict): A dictionary mapping neuron names to their corresponding CENGEN threshold values for level 2.
        th3 (dict): A dictionary mapping neuron names to their corresponding CENGEN threshold values for level 3.
        th4 (dict): A dictionary mapping neuron names to their corresponding CENGEN threshold values for level 4.
        nnames (list): A list of neuron names.
        cengen_neurons (dict): A dictionary mapping neuron names to their corresponding indices.

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

## Load synaptic weights from Excel file
prefix_synaptic_weights = 'Randi_2023/'
weight_file = DOWNLOAD_DIR + prefix_synaptic_weights + "41586_2023_6683_MOESM13_ESM.xls"

def loadSynapticWeights(nn, weightMatrix = weight_file):
    """
    Load synaptic weights from an Excel file into the given neural network.

    Parameters:
        nn (NeuralNetwork): The neural network object to update with synaptic weights.
        weightMatrix (str, optional): The path to the Excel file containing synaptic weights. 
            Defaults to leiferFile.

    Returns:
        None
    """
    wtMat = pd.read_excel(weightMatrix, index_col=0, engine='openpyxl').T
    for sid in list(nn.edges):
        if sid[0].name in wtMat:
            if sid[1].name in wtMat[sid[0].name]:
                nn.connections[sid].updateWeight(wtMat[sid[0].name][sid[1].name])


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

def plot_spiral(neunet, save=False):
    """
    Generates a spiral layout for the network.

    Parameters:
    - neunet (NeuralNetwork): The neural network object.

    Returns:
    - pos (dict): A dictionary mapping node names to their positions in the graph.
    """
    node_color = ['lightgray' for node in neunet.nodes]
    edge_color = []
    edge_weight = []
    for (u,v,attrib_dict) in list(neunet.edges.data()):
        edge_color.append(attrib_dict['color'])
        edge_weight.append(attrib_dict['weight'])

    pos = nx.spiral_layout(neunet, equidistant=True, resolution=0.5) 
    fig, ax = plt.subplots(figsize=(8,8))
    nx.draw(neunet, node_size = 1200, ax=ax, pos=pos, labels = {node: node.name for node in neunet.nodes}, with_labels=True, node_color = node_color, edge_color=edge_color, font_size=12)
    if save:
        plt.savefig(save)
    plt.show()
    plt.close()

def plot_shell(neunet, center=None, save=False, figsize=(8,8), fontsize=11):
    """
    Generates a shell layout for the network.

    Parameters:
    - neunet (NeuralNetwork): The neural network object.

    Returns:
    - pos (dict): A dictionary mapping node names to their positions in the graph.
    """
    if isinstance(center, str):
        assert center in neunet.neurons
        shells = [[neunet.neurons[center]], [neu for nname,neu in neunet.neurons.items() if nname !=center]]
    elif isinstance(center, list):
        #print(center, neunet.neurons.keys(), [c in neunet.neurons.keys() for c in center])
        assert all([c in neunet.neurons.keys() for c in center])
        shells = [[neunet.neurons[c] for c in center], [neu for nname,neu in neunet.neurons.items() if nname not in center]]
    else:
        raise ValueError("center must be a neuron name or a list of neuron names")
    node_color = ['lightgray' if not hasattr(node, 'color') else node.color for node in neunet.nodes]
    edge_color = ['lightgray' if not hasattr(edge, 'color') else edge.color for edge in neunet.edges]
    # edge_color = []
    # edge_weight = []
    # for (u,v,attrib_dict) in list(neunet.edges.data()):
    #     edge_color.append(attrib_dict['color'])
    #     edge_weight.append(attrib_dict['weight'])

    pos = nx.shell_layout(neunet, nlist=shells)
    fig, ax = plt.subplots(figsize=figsize)
    # print(len(pos))
    # print([(n, pos[neu]) for (n,neu) in neunet.neurons.items()])
    nx.draw(neunet, node_size = 800, ax=ax, pos=pos, with_labels=False, node_color = node_color, edge_color=edge_color)

    for node, (x, y) in pos.items():
        label = str(node.name)
        if len(label)>3:
            if (x>0 and y>0) or (x<0 and y<0):
                plt.text(x, y, label, fontsize=fontsize, rotation=45, ha='center', va='center')
            elif (x>0 and y<0) or (x<0 and y>0):
                plt.text(x, y, label, fontsize=fontsize, rotation=-45, ha='center', va='center')
            else:
                plt.text(x, y, label, fontsize=fontsize, ha='center', va='center')
        else:
            plt.text(x, y, label, fontsize=fontsize, ha='center', va='center')
    if save:
        plt.savefig(save)
    plt.show()
    plt.close()
    
    
def plot_layered(interesting_conns, neunet, nodeColors=None, edgeColors = None, save=False, title='', extraNodes=[], extraEdges=[], pos=[], mark_anatomical=False):
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
            norm = matplotlib.colors.Normalize(vmin=-1.,vmax=1.)
            m = cm.ScalarMappable(norm=norm, cmap=cmap)
            edge_color = [m.to_rgba(col) for col in edgeColors]
    else:
        raise ValueError('edge_colors must either be a string or a list of strings or floats \
            of the same length as interesting_conns')
    
    if nodeColors is None:
        node_color=[]
    elif all (isinstance(nc, str) for nc in nodeColors):
        node_color = [nc for nc in nodeColors]
    else:
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
        if n in oriNodes:
            if n in nodeColors.keys():
                #print("Y", n, nodeColors.keys())
                # node_color[neunet.neurons[n].type].append(o.to_rgba(nodeColors[n]))
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
        
        #G.nodes[n].layer = nn.neurons[n].type 
    
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
        pos = {p:array_op(pos[p],sx) for p in pos}
        pos, boxes = groupPosition(pos, type_cat)
        
        print(boxes)
    
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
    # divider = make_axes_locatable(ax)
    
    # cax = inset_axes(ax, loc='upper right', width="100%", height="100%",
    #                bbox_to_anchor=(0.75,0.85,.05,.05), bbox_transform=ax.transAxes) #
    # cax.set_xticks([-1,0,1])
    # cax.set_title("$\\Delta$ correlation", fontsize=24)
    # #cax = divider.append_axes("right", size="1%", pad=0.05)    
    # cbar = plt.colorbar(m, cax= cax, orientation='horizontal')
    # cbar.ax.tick_params(labelsize=24) 

    #cax2 = divider.append_axes("bottom", size="1%", pad=0.05)
    # cbar2= plt.colorbar(o, cax=cax2, orientation='horizontal')
    # cbar2.ax.tick_params(labelsize='xx-large') 

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
    #ax.set_title(title, y=0.01, fontsize=20)
    ax.set_xticks([])
    ax.set_yticks([])
    #plt.colorbar(m)
    plt.tight_layout()
    plt.subplots_adjust(top=2)
    if not save==False:
        plt.savefig(save)
    
    plt.show()
    plt.close()
    
    return pos

active_color, passive_color = "#CC5500", "lightgrey"
booleanDictionary = {True: active_color, False: passive_color}


def plot_position(nn, axis='AP-DV', highlight=None, booleanDictionary=booleanDictionary, title='', label=True, save=False):
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
    print([(n,facecolors[j]) for j,n in enumerate(nlabels) if n in highlight])

    f, ax = plt.subplots(figsize=(20,3))
    boolList = np.array([n in highlight for n in nlabels])

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

    ax.scatter(x[~boolList], y[~boolList], s=200, facecolor=facecolors[~boolList], edgecolor=facecolors[~boolList], alpha=alphas[~boolList], zorder=1)
    ax.scatter(x[boolList], y[boolList], s=200, facecolor=facecolors[boolList], edgecolor=facecolors[boolList], alpha=alphas[boolList], zorder=2)
    plt.axis('off')
    ax.set_title(title, fontsize="xx-large")

    if label:
        ta.allocate_text(f,ax,x[boolList], y[boolList],
                    nlabels[boolList],
                    x_scatter=x[boolList], y_scatter=y[boolList],
                    textsize=11, linecolor='gray', avoid_label_lines_overlap=True)

    # Add the legend to the plot
    legend_ax = f.add_axes([0.85, 0.1, 0.1, 0.1])
    # legend_ax.set_xlim(0, 1)
    # legend_ax.set_ylim(0, 1)
    legend_ax.set_xlabel(xax, fontsize="x-large")
    legend_ax.set_ylabel(yax, fontsize="x-large")
    legend_ax.set_xticks([])
    legend_ax.set_yticks([])
    legend_ax.set_aspect('equal')
    simpleaxis(legend_ax)
    #ax.set_xlim((-0.9,-0.))
    if save:
        plt.savefig(save)
    plt.show()

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
                nn_new.contractNeurons(neuronPair)
    return nn_new

def foldByNeuronType(nn_old):
    """
    Folds neurons in the given neural network based on the neuron type.

    Args:
        nn_old (NeuralNetwork): The original neural network.

    Returns:
        NeuralNetwork: The folded neural network.

    The function iterates over each neuron in the neural network and checks if the neuron type matches a specific pattern. If the neuron type matches the pattern, the function groups the neurons together based on the neuron number. The function then contracts the grouped neurons and updates the neuron dictionary.

    Note:
        The function assumes that the neuron type is represented by a sequence of numbers at the end of the neuron name.
    """
    nn_new = nn_old.copy()
    for m in nn_old.neurons:
        stripNums = m.rstrip('0123456789')
        if (len(m) - len(stripNums)==1 and m[-1]=='1') or (len(m) - len(stripNums)==2 and m[-1]=='1' and m[-2]=='0'):
            j=1
            moreInClass=True
            classNeurons = []
            while (moreInClass):
                if len(m) - len(stripNums) == 1:
                    if m[:len(stripNums)] + str(j+1) in nn_new.neurons:
                        classNeurons.append(m[:len(stripNums)] + str(j+1))
                        j+=1
                        moreInClass=True
                    else:
                        moreInClass=False
                else:
                    if m[:len(stripNums)] + "{:02d}".format(j+1) in nn_new.neurons:
                        classNeurons.append(m[:len(stripNums)] + "{:02d}".format(j+1))
                        j+=1
                        moreInClass=True
                    else:
                        moreInClass=False
            print(m, classNeurons)
            for n in classNeurons:
                if not m[:len(stripNums)] in nn_new.neurons: 
                    neuronPair = [m,n]
                    nn_new.contractNeurons(neuronPair)
                else:
                    print(m)
            nn_new.neurons[m].name = m[:len(stripNums)]
            nn_new.update_neurons()
    return nn_new 

def foldDorsoVentral(nn_old):
    """
    This function performs a dorsoventral folding operation on a neural network.
    It takes the old neural network as input and produces a new neural network after the folding operation.
    Parameters:
    - nn_old: The original neural network to be folded dorsoventrally.
    Returns:
    - nn_new: The new neural network after the dorsoventral folding operation.
    """
    nn_new = nn_old.copy()
    for m in nn_new.neurons:
        if m[-1] == 'D' and not m in ['RID']:
            n = m[:-1] + 'V'
            o = m[:-1] 
            if n in nn_new.neurons:
                if o in nn_new.neurons:
                    neuronPair1 = [m,n] # to contract D and V
                    neuronPair2 = [o,m] # to contract - and D
                    nn_new.contractNeurons(neuronPair1)
                    nn_new.contractNeurons(neuronPair2)
                else:
                    nn_new.neurons[m].name = m[:-1]
                    neuronPair1 = [m,n] # to contract D and V 
                    nn_new.contractNeurons(neuronPair1)
    nn_new.update_neurons() 
    return nn_new