'''
Configuration file for CeDNe utils module'''

__author__ = "Sahil Moza"
__date__ = "2025-04-06"
__license__ = "MIT"

import datetime
import cedne as ced

def get_root_path():
    """Returns the path to the root directory of the library."""
    return '/'.join(ced.__path__[0].split('/')[:-2])

root_path = get_root_path()

# Download locations
TOPDIR = root_path + '/' ## Change this to cedne and write a function to download data from an online server for heavy data.
DATADIR = TOPDIR + 'data_sources/'
DOWNLOAD_DIR = TOPDIR + 'data_sources/downloads/'
OUTPUT_DIR = TOPDIR + f'Output/{str(datetime.datetime.now()).split(" ")[0]}/'

prefix_NT = 'Wang_2024/'
prefix_CENGEN = 'CENGEN/'
prefix_NP = 'Ripoll-Sanchez_2023/'
prefix_synaptic_weights = 'Randi_2023/' #Signal propagation atlas

## Loading and building functions
cell_list = DATADIR + "Cell_list.pkl"
chemsyns = DATADIR + "chem_adj.pkl"
elecsyns = DATADIR + "gapjn_symm_adj.pkl"
neuronPositions = DATADIR + "neuronPosition.pkl"

cook_connectome = DOWNLOAD_DIR + 'cook_2019/'
witvliet_connectome = DOWNLOAD_DIR + 'witvliet_2020/'

lineage = DOWNLOAD_DIR + 'Worm_Atlas/Altun_lineage_corrected.xlsx'

## FlyWire
fly_wire = DOWNLOAD_DIR + 'FlyWire/'

#Download Links
#CENGEN
cengen_links = ['https://cengen.org/storage/021821_liberal_threshold1.csv',
'https://cengen.org/storage/021821_medium_threshold2.csv',
'https://cengen.org/storage/021821_conservative_threshold3.csv',
'https://cengen.org/storage/021821_stringent_threshold4.csv']

#FlyWire
flywire_links = ['https://codex.flywire.ai/api/download?data_product=consolidated_cell_types&data_version=783&dataset=fafb',
                 'https://codex.flywire.ai/api/download?data_product=classification&data_version=783&dataset=fafb',
                 'https://codex.flywire.ai/api/download?data_product=cell_stats&data_version=783&dataset=fafb',
                 'https://codex.flywire.ai/api/download?data_product=connections_princeton_no_threshold&data_version=783&dataset=fafb',
                 'https://codex.flywire.ai/api/download?data_product=coordinates&data_version=783&dataset=fafb',
                 'https://codex.flywire.ai/api/download?data_product=names&data_version=783&dataset=fafb'
                 'https://codex.flywire.ai/api/download?data_product=neurons&data_version=783&dataset=fafb'
                 ]



## Citation Tables
citations = {
    'cengen': ['https://doi.org/10.1016/j.neuron.2018.07.042',
               'https://doi.org/10.1016/j.cell.2021.06.023',
            ],
    'fly_wire': ['https://doi.org/10.1038/s41586-024-07558-y',
                 'https://doi.org/10.1038/s41586-024-07686-5',
                 'https://doi.org/10.1016/j.cell.2018.06.019',
                 'https://doi.org/10.1038/s41592-021-01183-7',
                 'https://doi.org/10.1007/978-3-030-00934-2_36',
                 'https://doi.org/10.1016/j.cell.2024.03.016',
                 'https://doi.org/10.1038/s41586-024-07981-1',
                 'https://doi.org/10.1038/s41586-024-07968-y',
                 'https://www.biorxiv.org/content/10.1101/2025.06.10.658788'
                ],
    'cook_connectome' : ['https://doi.org/10.1038/s41586-019-1352-7'],
    'witvliet_connectome': ['https://doi.org/10.1038/s41586-021-03778-8'],
    'sig_prop_atlas': ['https://doi.org/10.1038/s41586-023-06683-4'],
    'neuropeptide_atlas': ['https://doi.org/10.1016/j.neuron.2023.09.043'],
    'neurotransmitter_atlas': ['https://doi.org/10.7554/eLife.95402.3'],
    'atanas_whole_brain': ['https://doi.org/10.1016/j.cell.2023.07.035'],
    'altun_neurotrasmitters_receptors': ['https://doi.org/10.3908/wormatlas.5.202'],
    'lineage': 'https://doi.org/10.1016/0012-1606(77)90158-0'
    }