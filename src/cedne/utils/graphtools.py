"""
Graph manipulation, motif analysis, and transformation utilities for CeDNe

Includes neuron folding, triad enumeration, graph randomization, 
and hierarchical feedforward/feedback alignment measures.
"""
__author__ = "Sahil Moza"
__date__ = "2025-04-06"
__license__ = "MIT"

import copy
import networkx as nx
import numpy as np


suffixes = ['', 'D', 'V', 'L', 'R', 'DL', 'DR', 'VL', 'VR', '01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13']

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

def is_left_neuron(n):
    ''' Returns if a neuron is a left neuron. This works for Worms only for now.'''
    if n[-1] == 'L' and n not in ['ADL', 'AVL']:
        return True
    

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
    pass

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