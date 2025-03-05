# TODO

## Cleanups

- [ ] !!! Clean up the neurotransmitter table.

## Data integration

- [ ] !!! Add reading and writing to and from NWB format. Add that to the paper also.
- [ ] !!! Write about csv/json integration.
- [ ] !!! Add publication from where data is taken in a note section.

## New Classes

- [ ] ! Integrate behavioral components.
- [ ] !! Other graph elements (Map another graph to an object)
- [ ] ! Mapping functions.

## Core functions

- [ ] !! Write down paths from one neuron to another at different levels of depth (direct, 1 path away, etc.)
- [ ] !!! Formalizing connecting the StimResponse and Trial classes.

## ML stuff
- [ ] !!! Jax based optimization.
- [ ] !!! Latin hypercube sampling for hyperparameter optimization.

## Utility functions

- [ ] !! Easy Loading functions for loading custom data.
- [ ] !! Make movies for time series data.
- [ ] !! Export graph for plotting elsewhere.

## Examples

- [ ] !!! Run the neuropeptide example with the L4 connectivity.
- [ ] !!! Example on Louvain modularity and structural analysis.
- [ ] !!! Neurotransmitter check if paper uses connectivity information and do a shuffle to see correlation between connectivity and neurotransmitter identity.
- [ ] !!! Example for gap junction subunits and check with Bhattacharya et al paper.
- [ ] !!! Time series make movies.

## Storage

- [ ] ! HDF5 storage of data.
- [ ] !! neo4j+json+hdf5 for data persistance instead of pickles. neo4j for storing relationships, json for metadata and hdf5 for large matrices.

## Security

- [ ] !! Pickle warning
- [ ] ! Add some tests for sanitizing the pickle files

## Issue tracking and support

## Community involvement

- [ ] !! Plugin development
- [ ] !! Dataset uploads
- [ ] !! Contribution guidelines
- [ ] !! Evolving models framework (as a network): Setup for root models and submodels, etc.
