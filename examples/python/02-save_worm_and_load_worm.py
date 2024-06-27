# %%
from cedne import cedne
from cedne import utils

# %% [markdown]
# ### Make a worm and add a property

# %%
## Make a worm
w = utils.makeWorm()
nn = w.networks["Neutral"]

## Adding some properties to the neurons (see Example 1)
for n in nn.neurons:
    if nn.neurons[n].type == 'sensory' and nn.neurons[n].modality == 'amphid':
        nn.neurons[n].set_property('sensory-stim', True)

# %% [markdown]
# ### Save the worm

# %%
w.save("./worm.pkl")

# %% [markdown]
# ### Load the worm

# %%
w_loaded =  cedne.load_worm("./worm.pkl")
nn_loaded = w_loaded.networks["Neutral"]

# %% [markdown]
# ### New property has been stored in the neurons

# %%
g = nn_loaded.return_network_where(neurons_have = {'sensory-stim': True})
utils.plot_spiral(g)

# %%



