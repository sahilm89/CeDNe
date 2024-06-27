# %% [markdown]
# # Fold network and load data in the neurons

# %% [markdown]
# 
# While there are whole organism based datasets are increasingly becoming common for C elegans, several datasets are not currently available for every individual worm neuron, but a subset. Even for whole organism level datasets, data can often be found organized by neuron classes.
# For example, datasets can be found with neuron class based on left/right position (AWCL and AWCR into AWC), or by dorsal and ventral position (RMEDL/RMEVL), or by lineage/function (e.g. amphids or ventral cord neurons (VC)) or by some combination of these. Here we show by taking the example of the CENGEN dataset how the network can be folded across these axes, preserving the data and combining the connections across the axis of folding.

# %%
from cedne import utils
import pandas as pd

# %%
w = utils.makeWorm()
w.stage='L4'
w.sex='Hermaphrodite'
w.genotype='N2'
nn = w.networks['Neutral']

# %% [markdown]
# ## Loading CENGEN data into a dataframe (Using threshold 4 data)

# %%
transcripts = pd.read_csv(utils.thres_4,encoding= 'unicode_escape', index_col=1).drop(['Wormbase_ID','Unnamed: 0'], axis = 'columns')

# %% [markdown]
# ## Creating the folding dictionary

# %% [markdown]
# The folding dictionary has the class name as the key and a list of neuron names that belong to that class as the values.
# All the neurons in the values of this dictionary will be folded together into a single "class neuron" with its connections and 
# attributes being the union of the connections and attributes of its member neurons. 
# **Note that AWC_OFF has been mapped to AWCL and AWC_ON to AWCR for the graph.**

# %%
suffixes = ['', 'D', 'V', 'L', 'R', 'DL', 'DR', 'VL', 'VR', '01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13']
neuron_class = {}
for n in transcripts.columns:
    neuron_class[n] = []
    for s in suffixes:
        if (n+s) in nn.neurons:
            neuron_class[n].append(n+s)
    if n == 'AWC_OFF':
        neuron_class[n].append('AWCL')
    if n == 'AWC_ON':
        neuron_class[n].append('AWCR')
    if n == 'VD_DD':
        for m in nn.neurons:
            if m.startswith('VD') or m.startswith('DD'):
                neuron_class[n].append(m)
    if n == 'RME_LR':
        for m in nn.neurons:
            if m == 'RMEL' or m == 'RMER':
                neuron_class[n].append(m)
    if n == 'RME_DV':
        for m in nn.neurons:
            if m == 'RMED' or m == 'RMEV':
                neuron_class[n].append(m)
    if n == 'RMD_LR':
        for m in nn.neurons:
            if m == 'RMDL' or m == 'RMDR':
                neuron_class[n].append(m)
    if n == 'RMD_DV':
        for m in nn.neurons:
            if m.startswith('RMDD') or m.startswith('RMDV'):
                neuron_class[n].append(m)
    if n == 'IL2_LR':
        for m in nn.neurons:
            if m == 'IL2L' or m == 'IL2R':
                neuron_class[n].append(m)
    if n == 'IL2_DV':
        for m in nn.neurons:
            if m.startswith('IL2D') or m.startswith('IL2V'):
                neuron_class[n].append(m)

for p in ['DA09']:
    neuron_class['DA'].remove(p)
for p in ['VB01', 'VB02']:
    neuron_class['VB'].remove(p)
for p in ['VC04', 'VC05']:
    neuron_class['VC'].remove(p)
for p in ['DB01']:
    neuron_class['DB'].remove(p) 
for p in ['VA12']:
    neuron_class['VA'].remove(p)

neuron_class['VC_4_5'] = ['VC04', 'VC05']
neuron_class['DA9'] = ['DA09']

# %%
neuron_class

# %% [markdown]
# ## Folding the network

# %%
list(nn.neurons)

# %%
nn.fold_network(neuron_class)

# %% [markdown]
# ## Loading cengen transcripts into the network

# %%
for n in nn.neurons:
    nn.neurons[n].set_property('transcript', transcripts[n])

# %%
## Giving the sex specific neurons an interneuron type for positioning on graph.
sex_neurons = ['CAN']
for n in nn.neurons:
    if n in sex_neurons:
        nn.neurons[n].type = 'interneuron'

# %% [markdown]
# ## Plotting the network with the CENGEN transcript data

# %%
transcript_name = 'gpa-4'
save = False #outputDir + nprc + '.pdf'
conns = [(e[0].name, e[1].name) for e in nn.edges]
connNodes = list(nn.neurons.keys())
nodeColors = {n:('orange' if nn.neurons[n].transcript[transcript_name]>0 else 'lightgray') for n in connNodes}

pos = utils.plot_layered(conns, neunet=nn, nodeColors=nodeColors, title= '', save= save)

# %% [markdown]
# ASI looks interesting...

# %% [markdown]
# ## Plotting the subnetwork of the neuron 'ASI'

# %%
conns = nn.neurons['ASI'].get_connections()
g = nn.subnetwork(connections=conns)
utils.plot_shell(g, center='ASI')

# %%


# %%



