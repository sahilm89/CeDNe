# %% [markdown]
# # View positions and select neurons based on transcripts
# 
# It can also be valuable to spread out the classes into the whole connectome and visualize the position of neurons with respect 
# to the whole nervous system and the body of the worm.

# %% [markdown]
# 
# ### Here we look at ways to view positions of specific neurons in the neuronal network in different orientations, and select a subset of neurons based on their transcripts.

# %%
from cedne import utils

# %% [markdown]
# ### Load the worm and neural network as usual

# %%
w = utils.makeWorm()
nn = w.networks["Neutral"]

# %% [markdown]
# ### This function allows plotting to view the positions of the neurons and highlight a subset of neurons

# %%
mec_17 = ['ALML', 'ALMR', 'AVM', 'PLML', 'PLMR', 'PVM']

# %%
utils.plot_position(nn, highlight=mec_17, axis='AP-VD', title='mec-17', save="mec-17.svg") # Possible combinations of axis: RL-AP, AP-LR, AP-DV, DV-LR, AP-RL, LR-DV, LR-AP, DV-AP

# %% [markdown]
# ### Now we load the CENGEN transcipts

# %%
cengen_threshold_level = 4
utils.loadTranscripts(nn, threshold=cengen_threshold_level)
transcript_of_interest =  'sre-4'

# %%
neurons_of_interest = [n.name for n,t in nn.neurons_have('transcript').items() if t[transcript_of_interest]]

# %%
utils.plot_position(nn, highlight=neurons_of_interest, axis='AP-LR', title=transcript_of_interest, save=transcript_of_interest+".pdf")

# %% [markdown]
# ### We can also look at intersections of several CENGEN transcripts together, which can sometimes be useful to design promoters.

# %%
transcripts_of_interest =  ['sre-3', 'sre-4']

# %%
neurons_of_interest = [n.name for n,t in nn.neurons_have('transcript').items() if all (t[transcript_of_interest] for transcript_of_interest in transcripts_of_interest)]

# %%
utils.plot_position(nn, highlight=neurons_of_interest, axis='AP-LR')

# %% [markdown]
# ### View positions of neurons of interest and neighbours in a 3D plot.

# %% [markdown]
# 

# %% [markdown]
# 


