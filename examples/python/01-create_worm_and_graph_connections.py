# %% [markdown]
# # Creating a Worm Object and graphing its connectivity. 
# ### This is a basic example of loading a Worm object and graphing its connectivity. Here we also apply an attribute to a subset of the neurons and then subset the network based on this attribute.

# %%
from cedne import utils

# %% [markdown]
# ### Create a worm and its nervous system

# %%
w = utils.makeWorm()
nn = w.networks["Neutral"]

# %% [markdown]
# ### Set a property on a subset of neurons called "sensory-stim". This could in principle be a stimlus or any measurement made from the worm nervous system.

# %%
for n in nn.neurons:
    if nn.neurons[n].type == 'sensory' and nn.neurons[n].modality == 'amphid':
        nn.neurons[n].set_property('sensory-stim', True)

# %% [markdown]
# ### Subgraph the nervous system based on this property

# %%
g = nn.return_network_where(neurons_have = {'sensory-stim': True})

# %% [markdown]
# ### Plot the network as a spiral: Orange connections are chemical synapses and gray connections are gap junctions.

# %%
utils.plot_spiral(g)


