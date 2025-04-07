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

TOPDIR = root_path + '/' ## Change this to cedne and write a function to download data from an online server for heavy data.
DATADIR = TOPDIR + 'data_sources/'
DOWNLOAD_DIR = TOPDIR + 'data_sources/downloads/'
OUTPUT_DIR = TOPDIR + f'Output/{str(datetime.datetime.now()).split(" ")[0]}/'

prefix_NT = 'Wang_2024/'
prefix_CENGEN = 'CENGEN/'
prefix_NP = 'Ripoll-Sanchez_2023/'
prefix_synaptic_weights = 'Randi_2023/'

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