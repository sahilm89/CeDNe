
# CeDNe: C elegans Data Network

CeDNe (pronounced Sid-nee) is a Python Library for multi-omic integration of neuroscience data from
C elegans. CeDNe has been built on top of networkx package in order to leverage its graph structure
and Python's object oriented capabilities together to create graph-based data containers that are
intuitive to keep neuroscience data for analysis. Due to this structure, CeDNe acts as a multi-omic
tool to integrate different sources of information together and facilitates contextualizing new data
using existing information about C elegans. CeDNe is easy to use with only a bit of programming
experience and several examples have been provided. CeDNe can easily integrate into existing Python
based analyses pipelines.

## Examples

1. examples/notebooks/01-load_and_graph.ipynb
2. examples/notebooks/02-load_neuropeptide_and_find_intersection_graphs.ipynb
3. examples/notebooks/03-neuron-combinations-to-intersecting-transcripts.ipynb

## TODO

1. More inbuilt layers in the network.
    1.1 Integrate behavioral components deeper.
2. Clean up the neurotransmitter table.
3. Add G-protein-Neuropeptide relations.
4. Write down paths from one neuron to another at different levels of depth (direct, 1 path away, etc.)
5. Easy Loading functions for loading custom data.


Fenyves.