# %%
from cedne import simulator
from cedne import optimizer
from cedne import utils
from cedne import cedne
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
import os
import json

# %% [markdown]
# # Train parameters layer by layer, not all at once. That is the benefit of using hierarchical sequences over time!

# %%
if not os.path.isdir(utils.OUTPUT_DIR):
    os.makedirs(utils.OUTPUT_DIR)

# %%
ntype = ['sensory', 'interneuron', 'motorneuron']
facecolors = ['#FF6F61', '#FFD700', '#4682B4']
ntype_pairs = set([tuple(sorted([nt1, nt2])) for nt1 in ntype for nt2 in ntype])
colors= plt.cm.magma(np.linspace(0,1,len(ntype_pairs)))
type_color_dict = {p:color for (p,color) in zip(ntype_pairs, colors)}

# %%
w = utils.makeWorm(chem_only=True)
nn_chem = w.networks["Neutral"]

# %%
triad_motifs = utils.return_triads()
motif = triad_motifs['030T']
motif = utils.nx.relabel_nodes(motif, {1:1, 2:3, 3:2})

# %%
hseq = utils.make_hypermotifs(motif, 3, [(3,1)])
hseq = utils.nx.relabel_nodes(hseq, {'1.3-2.1':'2.1', '2.3-3.1':'3.1'})
hseq = utils.nx.convert_node_labels_to_integers(hseq, first_label=1, ordering='sorted', label_attribute='nodename')
all_ffgs = nn_chem.search_motifs(hseq)

# %%
jsons = {}
for js in os.listdir('/Users/sahilmoza/Documents/Postdoc/Yun Zhang/data/SteveFlavell-NeuroPAL-Cell/Control/'):
    with open ("/Users/sahilmoza/Documents/Postdoc/Yun Zhang/data/SteveFlavell-NeuroPAL-Cell/Control/{}".format(js), 'r') as f:
        jsons['Atanas et al (2023) ' +  js] = json.load(f)

# %%
command_interneurons = ['AVAL', 'AVAR', 'AVBL', 'AVBR', 'AVDL', 'AVDR', 'AVEL', 'AVER']

# %%
edgelist = []
input_neurons = []

mot_edgelabels = {node:[] for node in hseq}
neuron_layers = {node:[] for node in hseq}
for ffg in all_ffgs:
    nodelist = {node:None for node in hseq}
    for med, ned in ffg.items():
        for m,n in zip(med, ned):
            nodelist[m] = n.name
    # if nodelist[5] in command_interneurons:
        edgelist+= [(e[0], e[1], 0) for e in ffg.values() if not (e[0], e[1], 0) in edgelist]
        # input_neurons.append(nodelist[1])
        if nn_chem.neurons[nodelist[1]].type == 'sensory':
            input_neurons.append(nodelist[1])
        for m in nodelist:
            neuron_layers[m].append(nodelist[m])
input_neurons = list(sorted(set(input_neurons)))
neuron_layers = {layer: list(set(neuron_layers[layer])) for layer in neuron_layers}

# %%
nn_chem_sub_pre = nn_chem.subnetwork(connections=edgelist)

# %%
counter = 0
optim_neurs = {}
for n in nn_chem.neurons:
    # conn_neu_names = set([m2.name for m1,m2,id in nn_chem.neurons[n].outgoing()]) | set([m1.name for m1,m2,id in nn_chem.neurons[n].incoming()])
    conn_neu_names = set([m1.name for m1,m2,id in nn_chem.neurons[n].incoming()])
    fracs = []
    pres = []
    for database,p in jsons.items():
        if not database in optim_neurs.keys():
            optim_neurs[database] = []
        sortedKeys = sorted ([int(x) for x in (p['labeled'].keys())])
        labelledNeurons = {p['labeled'][str(x)]['label']:x for x in sortedKeys if not '?' in p['labeled'][str(x)]['label']}
        if len(conn_neu_names):
            frac = len(set(labelledNeurons.keys()) & conn_neu_names)/len(conn_neu_names)
            if n in labelledNeurons.keys() and frac>0.33:
                fracs.append(frac)
                pres.append(n in set(labelledNeurons.keys()))
                optim_neurs[database].append(n)
    if len(fracs)>0:
        counter+=1
        print(counter, n, nn_chem.neurons[n].type, list(zip(pres,fracs)))
# %%
def optimizable_neurons(common_set, nn_chem_sub):
    neurons_for_optim = []
    for n1 in nn_chem_sub.neurons:
        if nn_chem_sub.neurons[n1].type in ['sensory', 'interneuron', 'motorneuron']:
            optimizable = n1 in common_set# or (any([key[0].name in common_set for key in nn_chem_sub.neurons[n1].incoming().keys()]) and any([key[1].name in common_set for key in nn_chem_sub.neurons[n1].outgoing().keys()]))
            if optimizable:
                neurons_for_optim.append(n1)
    return neurons_for_optim



# %%
measuredNeurons = {}
neuron_labels = []
optim_neurs = {js:[] for js in jsons.keys()}
for js, p in jsons.items():
    sortedKeys = sorted ([int(x) for x in (p['labeled'].keys())])
    labelledNeurons = {p['labeled'][str(x)]['label']:x for x in sortedKeys if not '?' in p['labeled'][str(x)]['label']} # Removing unsure hits
    measuredNeurons[js] = {m:i for i,m in enumerate(set(labelledNeurons))}
    
    nlabs = list(measuredNeurons[js].keys())
    neuron_labels+=nlabs
    common_set = set(nn_chem_sub_pre.neurons).intersection(set(nlabs))
    frac_common = len(common_set)/len(nn_chem_sub_pre.neurons)
    # neurons_for_optim = optimizable_neurons(common_set, nn_chem_sub_pre)
    # optim_neurs[js] = neurons_for_optim

# %%
num_trials = 25
window_length = 100
window_step = 50
best_models = {}
best_loss = {}
best_params = {}
for datab_in, database in enumerate(jsons.keys()):
    ## Subnetwork and optimize
    best_models[database] = {}
    best_loss[database] = {}
    best_params[database] = {}
    nn_chem_sub = nn_chem.subnetwork(neurons=optim_neurs[database])
    # nn_chem_sub = nn_chem.subnetwork(connections=edgelist)

    ## Parameter Setup

    tconstants = [1] *len(nn_chem_sub.nodes)
    input_nodes = [nn_chem_sub.neurons[n] for n in input_neurons]

    weights = {e:1 for e in nn_chem_sub.edges}
    gains = {node:1.0 for node in nn_chem_sub.nodes}
    baselines = {node:0. for node in nn_chem_sub.nodes}
    time_constants = {n:t for n,t in zip(nn_chem_sub.nodes, tconstants)}
    num_timepoints = len(jsons[database]['trace_array'][measuredNeurons[database][list(measuredNeurons[database].keys())[0]]])
    time_points_all = np.arange(num_timepoints)#jsons[database]['max_t'])

    gain_base = {n.name:1 for n in nn_chem_sub.nodes} # units = _
    tconst_base = {n.name:1 for n in nn_chem_sub.nodes} # units = time_points ^-1
    base_base = {n.name:1 for n in nn_chem_sub.nodes}
    wts_base = {(n[0].name, n[1].name, n[2]):1 for n in nn_chem_sub.edges}

    gain_lims = np.array([-1,1])
    tconst_lims = np.array([2,50])
    base_lim = np.array([-2,2])
    wts_lim = np.array([-2,2])

    for neuron in nn_chem_sub.neurons:
            if neuron in measuredNeurons[database]:
                nn_chem_sub.neurons[neuron].set_property('amplitude', jsons[database]['trace_array'][measuredNeurons[database][neuron]])
                
    ## Inputs
    inputs = []
    time_points = np.arange(num_timepoints)
    for inp in input_nodes:
        if hasattr(inp, 'amplitude'):
            input_value = {t:inp.amplitude[j] for j,t in enumerate(time_points)}
            inputs.append(simulator.TimeSeriesInput([inp], input_value))

    ## Initialize rate model
    rate_model = simulator.RateModel(nn_chem_sub, input_nodes, weights, gains, time_constants, baselines, static_neurons=input_nodes, \
                                            time_points=time_points, inputs=inputs)

    node_parameter_bounds =  {'gain': {rn:gain_base[n.name]*gain_lims for n,rn in rate_model.neurons.items() if not n in input_nodes}, \
                                'time_constant': {rn:tconst_base[n.name]*tconst_lims for n,rn in rate_model.neurons.items() if not n in input_nodes},
                                'baseline': {rn:base_base[n.name]*base_lim for n,rn in rate_model.neurons.items() if not n in input_nodes}}
    
    edge_parameter_bounds = {'weight': {(e[0], e[1], e[2]): wts_base[(e[0].name, e[1].name, e[2])]*wts_lim for e in rate_model.edges}}

    real = {rate_model.neurons[node]:data['amplitude'] for node,data in nn_chem_sub.nodes(data=True) if 'amplitude' in data}
    rate_model.real = real
    
    vars_to_fit = [rn for rn in real.keys() if not rn in [rate_model.neurons[n] for n in input_nodes]]
    o = optimizer.OptunaOptimizer(rate_model, real, optimizer.mean_squared_error, node_parameter_bounds, edge_parameter_bounds, vars_to_fit, num_trials=num_trials, study_name=f"{database}_{window_length}_{window_step}_{num_trials}")

    ## First pass with short windows to loosely get the parameter space. 
    for win_ind in np.arange(((num_timepoints-window_length)//window_step) + 1):
        
        time_points = np.arange(win_ind*window_step, (win_ind*window_step) + window_length )
        rate_model.time_points = time_points
        ## Set vars to fit iteratively in a loop and add network layer by layer?
        try:
            best_param, best_model = o.optimize()
            best_models[database][win_ind] = best_model
            best_loss[database][win_ind] = o.study.best_value
            best_params[database][win_ind] = best_param
        except Exception as e:
            print(f"Optimization failed: {e}")
    print(f"{datab_in} out of {len(jsons.keys())} done.")