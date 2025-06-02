# %%
from cedne import simulator
from cedne import optimizer
from cedne import utils
from cedne import cedne
from collections import Counter
import numpy as np2
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
print(conn_neu_names)

# %%
jsons[database]

# %%
counter = 0
optimizable_neurs = {}
for n in nn_chem.neurons:
    # conn_neu_names = set([m2.name for m1,m2,id in nn_chem.neurons[n].outgoing()]) | set([m1.name for m1,m2,id in nn_chem.neurons[n].incoming()])
    conn_neu_names = set([m1.name for m1,m2,id in nn_chem.neurons[n].incoming()])
    fracs = []
    pres = []
    for database,p in jsons.items():
        if not database in optimizable_neurs.keys():
            optimizable_neurs[database] = []
        sortedKeys = sorted ([int(x) for x in (p['labeled'].keys())])
        labelledNeurons = {p['labeled'][str(x)]['label']:x for x in sortedKeys if not '?' in p['labeled'][str(x)]['label']}
        if len(conn_neu_names):
            frac = len(set(labelledNeurons.keys()) & conn_neu_names)/len(conn_neu_names)
            if n in labelledNeurons.keys() and frac>0.33:
                fracs.append(frac)
                pres.append(n in set(labelledNeurons.keys()))
                optimizable_neurs[database].append(n)
    if len(fracs)>0:
        counter+=1
        print(counter, n, nn_chem.neurons[n].type, list(zip(pres,fracs)))
        


# %%
len(optimizable_neurs['Atanas et al (2023) 2022-06-14-01.json'])

# %%
for n in nn_chem.neurons:
    nn.neurons[n].get_neighbours()
    for database in jsons.keys():
        jsons[database]

# %%
def optimizable_neurons(common_set, nn_chem_sub):
    neurons_for_optim = []
    for n1 in nn_chem_sub.neurons:
        if nn_chem_sub.neurons[n1].type in ['sensory', 'interneuron', 'motorneuron']:
            # print([key[0].name for key in nn_chem_sub.neurons[n1].incoming().keys()])
            optimizable = n1 in common_set# or (any([key[0].name in common_set for key in nn_chem_sub.neurons[n1].incoming().keys()]) and any([key[1].name in common_set for key in nn_chem_sub.neurons[n1].outgoing().keys()]))
            if optimizable:
                neurons_for_optim.append(n1)
    return neurons_for_optim

# %%
jsons = {}
for js in os.listdir('/Users/sahilmoza/Documents/Postdoc/Yun Zhang/data/SteveFlavell-NeuroPAL-Cell/Control/'):
    with open ("/Users/sahilmoza/Documents/Postdoc/Yun Zhang/data/SteveFlavell-NeuroPAL-Cell/Control/{}".format(js), 'r') as f:
        jsons['Atanas et al (2023) ' +  js] = json.load(f)

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
    neurons_for_optim = optimizable_neurons(common_set, nn_chem_sub_pre)
    optim_neurs[js] = neurons_for_optim
    # nn_chem_sub = nn_chem_sub_pre.subnetwork(neuron_names=neurons_for_optim)

# %%
# measuredNeurons = {}
# neuron_labels = []
# for js, p in jsons.items():
#     sortedKeys = sorted ([int(x) for x in (p['labeled'].keys())])
#     labelledNeurons = {p['labeled'][str(x)]['label']:x for x in sortedKeys if not '?' in p['labeled'][str(x)]['label']} # Removing unsure hits
#     measuredNeurons[js] = {m:i for i,m in enumerate(set(labelledNeurons))}
#     neuron_labels+=measuredNeurons[js].keys()
# neuron_labels = sorted(set(neuron_labels))

# %%
num_trials = 15
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
    #     nn_chem_sub = nn_chem.subnetwork(neurons=optim_neurs[database])
    nn_chem_sub = nn_chem.subnetwork(connections=edgelist)

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
        
        # node_parameter_bounds =  {'gain': {rn:gain_base[n.name]*gain_lims for n,rn in rate_model.neurons.items() if not n in input_nodes}, \
        #                             'time_constant': {rn:tconst_base[n.name]*tconst_lims for n,rn in rate_model.neurons.items() if not n in input_nodes},
        #                             'baseline': {rn:base_base[n.name]*base_lim for n,rn in rate_model.neurons.items() if not n in input_nodes}}
        # edge_parameter_bounds = {'weight': {(e[0].name.name, e[1].name.name, e[2]): wts_base[(e[0].name.name, e[1].name.name, e[2])]*wts_lim for e in rate_model.edges}}
        
        # real = {rate_model.neurons[node]:data['amplitude'] for node,data in nn_chem_sub.nodes(data=True) if 'amplitude' in data}
        #[win_ind*window_step: (win_ind*window_step) + window_length]

        ## Setting parameter bounds for the paramters of interest and set the rest to default to simulate. Use a noisy output to fit.
        # o = optimizer.OptunaOptimizer(rate_model, real, optimizer.mean_squared_error, node_parameter_bounds, edge_parameter_bounds, vars_to_fit, num_trials=num_trials)
        #o = optimizer.ScipyOptimizer(rate_model, real, optimizer.mean_squared_error, node_parameter_bounds, edge_parameter_bounds, vars_to_fit, num_trials=num_trials)
        ## Set vars to fit iteratively in a loop and add network layer by layer?
        try:
            best_param, best_model = o.optimize()
            # print(f"Optimization successful for {database}. Best parameters: {best_params}")
            # best_fit = best_model.simulate()
            # print("Simulation successful")

            best_models[database][win_ind] = best_model
            best_loss[database][win_ind] = o.study.best_value
            best_params[database][win_ind] = best_param
            # print("Plotting results") 
            # plot_rows = [k for k in best_fit.keys() if not str(k.name) in input_neurons and hasattr(nn_chem_sub.neurons[str(k.name)], 'amplitude')]
            # f, ax = plt.subplots(figsize=(10,2*len(plot_rows)), nrows=len(plot_rows), sharex=True, layout='constrained')
            # # for k, (n, node) in enumerate(nodelist):
            # for j,k in enumerate(plot_rows):
            #     ax[j].plot(time_points, np.array(nn_chem_sub.neurons[str(k.name)].amplitude)[time_points], label=f'{k.name}-{nn_chem_sub.neurons[str(k.name)].name}', color='gray')
            #     ax1 = ax[j]
            #     ax1.plot(time_points, best_fit[k], color='orange')
            #     utils.simpleaxis(ax[j])
            #     ax[j].set_title(f'{np.corrcoef(np.array(nn_chem_sub.neurons[str(k.name)].amplitude)[time_points], best_fit[k])[0,1]}')
            #     ax[j].legend(frameon=False)
            # f.suptitle(f'{database}')
            # plt.show()
        except Exception as e:
            print(f"Optimization failed: {e}")
        # best_params, best_model = o.optimize()
    print(f"{datab_in} out of {len(jsons.keys())} done.")

# %%
for database in jsons:
    # with open(f"{utils.OUTPUT_DIR}/{database}_best_models.json", 'w') as f:
    #     json.dump(best_models[database], f)
    with open(f"{utils.OUTPUT_DIR}/{database}_best_loss.json", 'w') as f:
        json.dump(best_loss[database], f)
    with open(f"{utils.OUTPUT_DIR}/{database}_best_params.json", 'w') as f:
        json.dump(best_params[database], f)

# %%
f, ax = plt.subplots(figsize=(10,10), nrows=len(jsons.keys()), sharex=True, layout='constrained')
for j,database in enumerate(jsons):
    loss = []
    for k in sorted(best_loss[database].keys()):
        loss.append(best_loss[database][k])
    ax[j].plot(loss, label=database)
    ax[j].legend()
    utils.simpleaxis(ax[j])
plt.show()

# %%
for inp in inputs:
    print(inp, inp.input_neurons[0].process_inputs(0))

# %%
# num_trials = 10
# best_models = {}
# for database in jsons.keys():
#     ## Subnetwork and optimize
#     #nn_chem_sub = nn_chem.subnetwork(connections=edgelist)
#     nn_chem_sub = nn_chem.subnetwork(neurons=optim_neurs[database])

#     ## Parameter Setup
#     inputs = []
#     tconstants = [1] *len(nn_chem_sub.nodes)
#     input_nodes = [nn_chem_sub.neurons[n] for n in input_neurons]

#     weights = {e:1 for e in nn_chem_sub.edges}
#     gains = {node:1.0 for node in nn_chem_sub.nodes}
#     baselines = {node:0. for node in nn_chem_sub.nodes}
#     time_constants = {n:t for n,t in zip(nn_chem_sub.nodes, tconstants)}
#     num_timepoints = len(jsons[database]['trace_array'][measuredNeurons[database][list(measuredNeurons[database].keys())[0]]])
#     for neuron in nn_chem_sub.neurons:
#         if neuron in measuredNeurons[database]:
#             nn_chem_sub.neurons[neuron].set_property('amplitude', jsons[database]['trace_array'][measuredNeurons[database][neuron]][:num_timepoints])
#     time_points = np.arange(num_timepoints)#jsons[database]['max_t'])

#     ## Inputs
#     for inp in input_nodes:
#         if hasattr(inp, 'amplitude'):
#             input_value = {t:inp.amplitude[t] for t in time_points}
#             inputs.append(simulator.TimeSeriesInput([inp], input_value))
    
#     node_parameters={'gain':gains, 'time_constant':time_constants, 'baseline':baselines}
#     edge_parameters={'weight':weights}

#     ## Initialize rate model
#     rate_model = simulator.JaxRateModel(nn_chem_sub, input_nodes, node_parameters=node_parameters, edge_parameters=edge_parameters, static_nodes=input_nodes, \
#                                         time_points=time_points)
    
#     node_parameter_bounds =  {'gain': {rn:(-1, 1) for n,rn in rate_model.neurons.items() if not n in input_nodes}, \
#                                 'time_constant': {rn:(1, 5) for n,rn in rate_model.neurons.items() if not n in input_nodes},
#                                 'baseline': {rn:(0, 2) for n,rn in rate_model.neurons.items() if not n in input_nodes}}
#     edge_parameter_bounds = {'weight': {e:(-2, 2) for e in rate_model.edges}}
    
#     real = {rate_model.neurons[node]:data['amplitude'] for node,data in nn_chem_sub.nodes(data=True) if 'amplitude' in data}
#     vars_to_fit = [rn for rn in real.keys() if not rn in [rate_model.neurons[n] for n in input_nodes]]
    
#     ## Setting parameter bounds for the paramters of interest and set the rest to default to simulate. Use a noisy output to fit.
#     #o = optimizer.OptunaOptimizer(rate_model, real, optimizer.mean_squared_error, node_parameter_bounds, edge_parameter_bounds, vars_to_fit, num_trials=num_trials)
#     o = optimizer.JaxOptimizer(rate_model, real, optimizer.mean_squared_error, node_parameter_bounds, edge_parameter_bounds, vars_to_fit, num_trials=num_trials)
#     ## Set vars to fit iteratively in a loop and add network layer by layer?
    
#     best_params, best_model = o.optimize()
#     best_fit = best_model.simulate()

#     best_models[database] = (best_params, best_model)
    
#     plot_rows = [k for k in best_fit.keys() if not str(k.name) in input_neurons and hasattr(nn_chem_sub.neurons[str(k.name)], 'amplitude')]
#     f, ax = plt.subplots(figsize=(10,2*len(plot_rows)), nrows=len(plot_rows), sharex=True, layout='constrained')
#     # for k, (n, node) in enumerate(nodelist):
#     for j,k in enumerate(plot_rows):
#         ax[j].plot(nn_chem_sub.neurons[str(k.name)].amplitude, label=f'{k.name}-{nn_chem_sub.neurons[str(k.name)].name}', color='gray')
#         ax1 = ax[j]
#         ax1.plot(best_fit[k], color='orange')
#         utils.simpleaxis(ax[j])
#         ax[j].set_title(f'{np.corrcoef(nn_chem_sub.neurons[str(k.name)].amplitude, best_fit[k])[0,1]}')
#         ax[j].legend(frameon=False)
#     f.suptitle(f'{database}')
#     plt.show()

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
len(nn_chem_sub.neurons)*3 + len(nn_chem_sub.edges)*1

# %%
for m in best_models:
    sorted_k = sorted(best_models[m].keys())[-1]
    model1 = best_models[m][sorted_k]
    model1.time_points = np.arange(num_timepoints)
    res = model1.simulate()

    f, ax = plt.subplots(figsize=(10,2*len(res.keys())), nrows=len(res.keys()), sharex=True, layout='constrained')
    # for k, (n, node) in enumerate(nodelist):
    for j,k in enumerate(res.keys()):
        utils.simpleaxis(ax[j])
        if hasattr(nn_chem_sub.neurons[str(k.name)], 'amplitude'):
            ax[j].plot(np.arange(num_timepoints), np.array(nn_chem_sub.neurons[str(k.name)].amplitude), color='gray')
            ax[j].set_title(f'{np.corrcoef(np.array(nn_chem_sub.neurons[str(k.name)].amplitude)[np.arange(num_timepoints)], res[k])[0,1]}')
        # ax[j].plot(time_points, np.array(nn_chem_sub.neurons[str(k.name)].amplitude)[time_points], label=f'{k.name}-{nn_chem_sub.neurons[str(k.name)].name}', color='gray')
        ax1 = ax[j]
        ax1.plot(np.arange(num_timepoints), res[k], color='orange', label=f'{k.name}-{nn_chem_sub.neurons[str(k.name)].name}')
        ax1.legend(frameon=False)
    f.suptitle(f'{database}')
    plt.show()

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
plt.show()

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
plt.show()

# %%
f, ax = plt.subplots(figsize=(20,12), layout='constrained', nrows=len(var_dict['weight'])//100+1)
xticks=[]
for j, (n, val) in enumerate(var_dict['weight'].items()):
    ax[j//100].scatter([j%100]*len(val), val)
    xticks.append('-'.join(n))
    # ax[j//100].set_xticks(np.arange(len(xticks)), xticks, rotation=45)
utils.simpleaxis(ax)
plt.show()

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

plt.show()

# %%
min_motif = ['1.1', '1.2', '2.1', '2.2', '3.1']
tconstants = [1, 1, 1, 1,1,1,1]
input_nodes = [min_motif[0]]

weights = {e:1 for e in hseq.edges}
gains = {node:1.0 for node in hseq.nodes}
baselines = {node:0. for node in hseq.nodes}
time_constants = {n:t for n,t in zip(hseq.nodes, tconstants)}

# countdown = 10
for database in jsons.keys():
    nn_chem_sub = nn_chem.subnetwork(connections=all_edges)
    all_ffgs = nn_chem_sub.search_motifs(hseq)
    num_timepoints = len(jsons[database]['trace_array'][measuredNeurons[database][list(measuredNeurons[database].keys())[0]]])
    for neuron in nn_chem_sub.neurons:
        if neuron in measuredNeurons[database]:
            nn_chem_sub.neurons[neuron].set_property('amplitude', jsons[database]['trace_array'][measuredNeurons[database][neuron]])
    
    by_motif = {}
    for j,ffg in enumerate(all_ffgs):
        nodelist = []
        for edge in sorted(edges):
            if hasattr(nn_chem_sub.neurons[ffg[edge][0].name], 'amplitude') and hasattr(nn_chem_sub.neurons[ffg[edge][1].name], 'amplitude'):
                nodelist+= [(edge[0], ffg[edge][0].name), (edge[1], ffg[edge][1].name)]
        nodelist = sorted(set(nodelist))
        if nodelist:# and countdown>0:
            if all(n in list(zip(*nodelist))[0] for n in min_motif):
                
                cedne.GraphMap(ffg, hseq, nn_chem_sub, map_type='edge')
                inputs = []
                time_points = np.arange(0,jsons[database]['max_t'])
                for inp in input_nodes:
                    if hasattr(nn_chem_sub.neurons[hseq.nodes[inp]['map'].name], 'amplitude'):
                        input_value = {t:nn_chem_sub.neurons[hseq.nodes[inp]['map'].name].amplitude[t] for t in time_points}
                        inputs.append(simulator.TimeSeriesInput(input_nodes, input_value))
                rate_model = simulator.RateModel(hseq, input_nodes, weights, gains, time_constants, baselines, static_nodes=input_nodes, time_points=time_points, inputs=inputs)
                
                node_parameter_bounds =  {'gain': {rn:(0, 5) for n,rn in rate_model.neurons.items() if not n in input_nodes}, 'time_constant': {rn:(0, 20) for n,rn in rate_model.neurons.items() if not n in input_nodes}, 'baseline': {rn:(0, 3) for n,rn in rate_model.neurons.items() if not n in input_nodes}}
                edge_parameter_bounds = {'weight': {e:(-10, 10) for e in rate_model.edges}}
                
                real = {rate_model.neurons[node]:data['map'].amplitude for node,data in hseq.nodes(data=True) if hasattr(data['map'], 'amplitude')}
                vars_to_fit = [rn for rn in real.keys() if not rn in input_nodes]
                
                ## Setting parameter bounds for the paramters of interest and set the rest to default to simulate. Use a noisy output to fit.
                o = optimizer.OptunaOptimizer(rate_model, real, optimizer.mean_squared_error, node_parameter_bounds, edge_parameter_bounds, vars_to_fit, num_trials=1e3)
                best_params, best_model = o.optimize()
                best_fit = best_model.simulate()
                
                f, ax = plt.subplots(figsize=(10,2*len(hseq.nodes)), nrows=len(hseq.nodes), sharex=True)
                # for k, (n, node) in enumerate(nodelist):
                for j,k in enumerate(best_fit.keys()):
                    if k.name in list(zip(*nodelist))[0]:
                        ax[j].plot(nn_chem_sub.neurons[hseq.nodes[k.name]['map'].name].amplitude, label=f'{k.name}-{hseq.nodes[k.name]['map'].name}', color='gray')
                        ax1 = ax[j]
                        ax1.plot(best_fit[k], color='orange')
                        utils.simpleaxis(ax[j])
                        ax[j].set_title(f'{np.corrcoef(nn_chem_sub.neurons[hseq.nodes[k.name]['map'].name].amplitude, best_fit[k])[0,1]}')
                        ax[j].legend(frameon=False)
                f.suptitle(f'{database}')
                plt.show()
                # countdown-=1

# %%
triads = utils.return_triads()
G = triads['030T']
weights = {(1, 3): -3., (3, 2): -1, (1, 2): -3}

input_nodes = [1]
gains = {node:1.0 for node in G.nodes}
tconstants = [10, 10, 1]
time_constants = {n:t for n,t in zip(G.nodes, tconstants)}
rate_model = simulator.RateModel(G, input_nodes, weights, gains, time_constants, static_nodes=input_nodes)

initial_rates = [0., 0., 0.]
max_t = 90
time_points = np.linspace(0, max_t, 451)

inp1_value = 1
input_value = {t:inp1_value*np.sin((t/max_t)*2*np.pi) for t in time_points}
inp_vals = [input_value[t] for t in time_points]
input1= simulator.TimeSeriesInput(input_nodes, input_value)

inputs = [input1]

rates = rate_model.simulate(time_points, inputs)

f = utils.plot_simulation_results((rate_model, inputs, rates), twinx=False)

# %%


# %%
k

# %%



