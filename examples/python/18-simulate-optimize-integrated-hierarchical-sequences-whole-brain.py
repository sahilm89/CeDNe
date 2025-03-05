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
import sys

# %%
if not os.path.isdir(utils.OUTPUT_DIR):
    os.makedirs(utils.OUTPUT_DIR)

njobs = int(sys.argv[1])
# %%
ntype = ['sensory', 'interneuron', 'motorneuron']
facecolors = ['#FF6F61', '#FFD700', '#4682B4']
ntype_pairs = set([tuple(sorted([nt1, nt2])) for nt1 in ntype for nt2 in ntype])
colors= plt.cm.magma(np.linspace(0,1,len(ntype_pairs)))
type_color_dict = {p:color for (p,color) in zip(ntype_pairs, colors)}

# %%
w = utils.makeWorm(chem_only=True)
nn_chem = w.networks["Neutral"]

# w_both = utils.makeWorm()
# nn_both = w_both.networks["Neutral"] 

# w_gapjn = utils.makeWorm(gapjn_only=True)
# nn_gapjn = w.networks["Neutral"]

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
# mot_edgelabels = {node:[] for node in hseq}
# for ffg in all_ffgs:
#     nodelist = {node:None for node in hseq}
#     for med, ned in ffg.items():
#         for m,n in zip(med, ned):
#             nodelist[m] = n.name
#     for node in nodelist:
#         mot_edgelabels[node].append(nodelist[node])

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
    if nodelist[5] in command_interneurons:
        edgelist+= [(e[0], e[1], 0) for e in ffg.values() if not (e[0], e[1], 0) in edgelist]
        input_neurons.append(nodelist[1])
        for m in nodelist:
            neuron_layers[m].append(nodelist[m])
input_neurons = list(sorted(set(input_neurons)))
neuron_layers = {layer: list(set(neuron_layers[layer])) for layer in neuron_layers}

# %%
nn_chem_sub = nn_chem.subnetwork(connections=edgelist)

# %%
jsons = {}
for js in os.listdir('../..//data_sources/downloads/Atanas_2023/Control/'):
    with open ("../..//data_sources/downloads/Atanas_2023/Control/{}".format(js), 'r') as f:
        jsons['Atanas et al (2023) ' +  js] = json.load(f)

# %%
measuredNeurons = {}
neuron_labels = []
for js, p in jsons.items():
    sortedKeys = sorted ([int(x) for x in (p['labeled'].keys())])
    labelledNeurons = {p['labeled'][str(x)]['label']:x for x in sortedKeys if not '?' in p['labeled'][str(x)]['label']} # Removing unsure hits
    measuredNeurons[js] = {m:i for i,m in enumerate(set(labelledNeurons))}
    neuron_labels+=measuredNeurons[js].keys()
neuron_labels = sorted(set(neuron_labels))


# %%
num_trials = 2000
best_models = {}

gain_range = (1,10)
tconstant_range = (1,10)
baseline_range = (-2, 2)
weight_range = (-2, 2)

for database in jsons.keys():
    ## Subnetwork and optimize
    nn_chem_sub = nn_chem.subnetwork(connections=edgelist)

    ## Parameter Setup
    inputs = []
    tconstants = [1] *len(nn_chem_sub.nodes)
    input_nodes = [nn_chem_sub.neurons[n] for n in input_neurons]

    weights = {e:1 for e in nn_chem_sub.edges}
    gains = {node:1.0 for node in nn_chem_sub.nodes}
    baselines = {node:0. for node in nn_chem_sub.nodes}
    time_constants = {n:t for n,t in zip(nn_chem_sub.nodes, tconstants)}
    num_timepoints = len(jsons[database]['trace_array'][measuredNeurons[database][list(measuredNeurons[database].keys())[0]]])
    for neuron in nn_chem_sub.neurons:
        if neuron in measuredNeurons[database]:
            nn_chem_sub.neurons[neuron].set_property('amplitude', jsons[database]['trace_array'][measuredNeurons[database][neuron]][:num_timepoints])
    time_points = np.arange(num_timepoints)#jsons[database]['max_t'])

    ## Inputs
    for inp in input_nodes:
        if hasattr(inp, 'amplitude'):
            input_value = {t:inp.amplitude[t] for t in time_points}
            inputs.append(simulator.TimeSeriesInput([inp], input_value))

    ## Initialize rate model
    rate_model = simulator.RateModel(nn_chem_sub, input_nodes, weights, gains, time_constants, baselines, static_nodes=input_nodes, \
                                        time_points=time_points, inputs=inputs)
    
    node_parameter_bounds =  {'gain': {rn:gain_range for n,rn in rate_model.node_dict.items() if not n in input_nodes}, \
                                'time_constant': {rn:tconstant_range for n,rn in rate_model.node_dict.items() if not n in input_nodes},
                                'baseline': {rn:baseline_range for n,rn in rate_model.node_dict.items() if not n in input_nodes}}
    edge_parameter_bounds = {'weight': {e:weight_range for e in rate_model.edges}}
    
    real = {rate_model.node_dict[node]:data['amplitude'] for node,data in nn_chem_sub.nodes(data=True) if 'amplitude' in data}
    vars_to_fit = [rn for rn in real.keys() if not rn in [rate_model.node_dict[n] for n in input_nodes]]
    
    ## Setting parameter bounds for the paramters of interest and set the rest to default to simulate. Use a noisy output to fit.
    o = optimizer.OptunaOptimizer(rate_model, real, optimizer.mean_squared_error, node_parameter_bounds, edge_parameter_bounds, vars_to_fit, num_trials=num_trials, njobs=njobs)
    best_params, best_model = o.optimize()
    best_fit = best_model.simulate()

    best_models[database] = (best_params, best_model)
    
    plot_rows = [k for k in best_fit.keys() if not str(k.label) in input_neurons and hasattr(nn_chem_sub.neurons[str(k.label)], 'amplitude')]
    f, ax = plt.subplots(figsize=(10,2*len(plot_rows)), nrows=len(plot_rows), sharex=True, layout='constrained')
    # for k, (n, node) in enumerate(nodelist):
    for j,k in enumerate(plot_rows):
        ax[j].plot(nn_chem_sub.neurons[str(k.label)].amplitude, label=f'{k.label}-{nn_chem_sub.neurons[str(k.label)].name}', color='gray')
        ax1 = ax[j]
        ax1.plot(best_fit[k], color='orange')
        utils.simpleaxis(ax[j])
        ax[j].set_title(f'{np.corrcoef(nn_chem_sub.neurons[str(k.label)].amplitude, best_fit[k])[0,1]}')
        ax[j].legend(frameon=False)
    f.suptitle(f'{database}')
    plt.savefig(f'{database}_best_model.png')
    plt.close()

# %%
var_dict = {}
for database, (pars, mod) in best_models.items():
    for key, val in pars.items():
        par, *rest = key.split(':')
        if par not in var_dict:
            var_dict[par] = {}
        if not tuple(rest) in var_dict[par]:
            var_dict[par][tuple(rest)] = []
        var_dict[par][tuple(rest)].append(val)

# %%
f, ax = plt.subplots(figsize=(24,8), nrows= len(var_dict)-1, layout='constrained', sharex=True)
for j, (par, vars) in enumerate(sorted(var_dict.items(), key=lambda x:x[0])):
    xticks = []
    if not par == 'weight':
        for k, (n, val) in enumerate(vars.items()):
            ax[j].scatter([k]*len(val), val)
            xticks.append('-'.join(n))
        ax[j].set_xticks(np.arange(len(xticks)), xticks, rotation=45)
        utils.simpleaxis(ax[j])
        ax[j].set_title(par)
plt.savefig('best_pars.png')
plt.close()
# %%
f, ax = plt.subplots(figsize=(20,12), layout='constrained', nrows=len(var_dict['weight'])//100+1)
xticks=[]
for j, (n, val) in enumerate(var_dict['weight'].items()):
    ax[j//100].scatter([j%100]*len(val), val)
    xticks.append('-'.join(n))
    # ax[j//100].set_xticks(np.arange(len(xticks)), xticks, rotation=45)
utils.simpleaxis(ax)
plt.savefig('best_weights.png')
plt.close()

# %%
f, ax = plt.subplots(figsize=(36,8), layout='constrained', nrows=2)
xticks_1=[]
xticks_2=[]
k1=0
k2 = 0
for j, (n, val) in enumerate(var_dict['weight'].items()):
    if n[1] in command_interneurons:
        ax[0].scatter([k1]*len(val), val)
        xticks_1.append('-'.join(n))
        k1+=1
    if n[0] in command_interneurons:
        ax[1].scatter([k2]*len(val), val)
        xticks_2.append('-'.join(n))
        k2+=1
ax[0].set_xticks(np.arange(len(xticks_1)), xticks_1, rotation=45, fontsize='x-large')
ax[1].set_xticks(np.arange(len(xticks_2)), xticks_2, rotation=45, fontsize='x-large')
utils.simpleaxis(ax)

plt.savefig('command_int_weights.png')
plt.close()

# %%
# min_motif = ['1.1', '1.2', '2.1', '2.2', '3.1']
# tconstants = [1, 1, 1, 1,1,1,1]
# input_nodes = [min_motif[0]]

# weights = {e:1 for e in hseq.edges}
# gains = {node:1.0 for node in hseq.nodes}
# baselines = {node:0. for node in hseq.nodes}
# time_constants = {n:t for n,t in zip(hseq.nodes, tconstants)}

# # countdown = 10
# for database in jsons.keys():
#     nn_chem_sub = nn_chem.subnetwork(connections=all_edges)
#     all_ffgs = nn_chem_sub.search_motifs(hseq)
#     num_timepoints = len(jsons[database]['trace_array'][measuredNeurons[database][list(measuredNeurons[database].keys())[0]]])
#     for neuron in nn_chem_sub.neurons:
#         if neuron in measuredNeurons[database]:
#             nn_chem_sub.neurons[neuron].set_property('amplitude', jsons[database]['trace_array'][measuredNeurons[database][neuron]])
    
#     by_motif = {}
#     for j,ffg in enumerate(all_ffgs):
#         nodelist = []
#         for edge in sorted(edges):
#             if hasattr(nn_chem_sub.neurons[ffg[edge][0].name], 'amplitude') and hasattr(nn_chem_sub.neurons[ffg[edge][1].name], 'amplitude'):
#                 nodelist+= [(edge[0], ffg[edge][0].name), (edge[1], ffg[edge][1].name)]
#         nodelist = sorted(set(nodelist))
#         if nodelist:# and countdown>0:
#             if all(n in list(zip(*nodelist))[0] for n in min_motif):
                
#                 cedne.GraphMap(ffg, hseq, nn_chem_sub, map_type='edge')
#                 inputs = []
#                 time_points = np.arange(0,jsons[database]['max_t'])
#                 for inp in input_nodes:
#                     if hasattr(nn_chem_sub.neurons[hseq.nodes[inp]['map'].name], 'amplitude'):
#                         input_value = {t:nn_chem_sub.neurons[hseq.nodes[inp]['map'].name].amplitude[t] for t in time_points}
#                         inputs.append(simulator.TimeSeriesInput(input_nodes, input_value))
#                 rate_model = simulator.RateModel(hseq, input_nodes, weights, gains, time_constants, baselines, static_nodes=input_nodes, time_points=time_points, inputs=inputs)
                
#                 node_parameter_bounds =  {'gain': {rn:(0, 2) for n,rn in rate_model.node_dict.items() if not n in input_nodes}, 'time_constant': {rn:(1, 10) for n,rn in rate_model.node_dict.items() if not n in input_nodes}, 'baseline': {rn:(0, 2) for n,rn in rate_model.node_dict.items() if not n in input_nodes}}
#                 edge_parameter_bounds = {'weight': {e:(-2, 2) for e in rate_model.edges}}
                
#                 real = {rate_model.node_dict[node]:data['map'].amplitude for node,data in hseq.nodes(data=True) if hasattr(data['map'], 'amplitude')}
#                 vars_to_fit = [rn for rn in real.keys() if not rn in input_nodes]
                
#                 ## Setting parameter bounds for the paramters of interest and set the rest to default to simulate. Use a noisy output to fit.
#                 o = optimizer.OptunaOptimizer(rate_model, real, optimizer.mean_squared_error, node_parameter_bounds, edge_parameter_bounds, vars_to_fit, num_trials=1e3)
#                 best_params, best_model = o.optimize()
#                 best_fit = best_model.simulate()
                
#                 f, ax = plt.subplots(figsize=(10,2*len(hseq.nodes)), nrows=len(hseq.nodes), sharex=True)
#                 # for k, (n, node) in enumerate(nodelist):
#                 for j,k in enumerate(best_fit.keys()):
#                     if k.label in list(zip(*nodelist))[0]:
#                         ax[j].plot(nn_chem_sub.neurons[hseq.nodes[k.label]['map'].name].amplitude, label=f'{k.label}-{hseq.nodes[k.label]["map"].name}', color='gray')
#                         ax1 = ax[j]
#                         ax1.plot(best_fit[k], color='orange')
#                         utils.simpleaxis(ax[j])
#                         ax[j].set_title(f'{np.corrcoef(nn_chem_sub.neurons[hseq.nodes[k.label]["map"].name].amplitude, best_fit[k])[0,1]}')
#                         ax[j].legend(frameon=False)
#                 f.suptitle(f'{database}')
#                 plt.show()
                # countdown-=1

# %%
# triads = utils.return_triads()
# G = triads['030T']
# weights = {(1, 3): -3., (3, 2): -1, (1, 2): -3}

# input_nodes = [1]
# gains = {node:1.0 for node in G.nodes}
# tconstants = [10, 10, 1]
# time_constants = {n:t for n,t in zip(G.nodes, tconstants)}
# rate_model = simulator.RateModel(G, input_nodes, weights, gains, time_constants, static_nodes=input_nodes)

# initial_rates = [0., 0., 0.]
# max_t = 90
# time_points = np.linspace(0, max_t, 451)

# inp1_value = 1
# input_value = {t:inp1_value*np.sin((t/max_t)*2*np.pi) for t in time_points}
# inp_vals = [input_value[t] for t in time_points]
# input1= simulator.TimeSeriesInput(input_nodes, input_value)

# inputs = [input1]

# rates = rate_model.simulate(time_points, inputs)

# f = utils.plot_simulation_results((rate_model, inputs, rates), twinx=False)