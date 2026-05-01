'''
Configuration file for CeDNe utils module'''

__author__ = "Sahil Moza"
__date__ = "2025-04-06"
__license__ = "MIT"

import datetime
import cedne as ced
from pathlib import Path

root_path = Path(ced.__path__[0])

TOPDIR = root_path.parents[1]
DATADIR = TOPDIR / 'data_sources'
DOWNLOAD_DIR = DATADIR / 'downloads'
OUTPUT_DIR = TOPDIR / 'Output' / f'{str(datetime.datetime.now()).split(" ")[0]}'

prefix_NT = 'Wang_2024/'
prefix_CENGEN = 'CENGEN/'
prefix_NP = 'Ripoll-Sanchez_2023/'
prefix_synaptic_weights = 'Randi_2023/' #Signal propagation atlas

## Loading and building functions
cell_list = DATADIR / "Cell_list.pkl"
chemsyns = DATADIR / "chem_adj.pkl"
elecsyns = DATADIR / "gapjn_symm_adj.pkl"
neuronPositions = DATADIR / "neuronPosition.pkl"

# One-shot cache for the Cook male loader. The raw supplementary workbook is
# slow to parse (~5s via openpyxl); we preprocess it into a single pickle
# (labels, type dict, category dict, sparse chem adj, sparse gap-jn adj) the
# first time makeWorm(style='cook', sex='male') runs, and read from that
# cache on every subsequent call.
cook_male_cache = DATADIR / "cook_male_data.pkl"

cook_connectome = DOWNLOAD_DIR / 'cook_2019'
witvliet_connectome = DOWNLOAD_DIR / 'witvliet_2020'
white_connectome = DOWNLOAD_DIR / 'white_1986'

lineage = DOWNLOAD_DIR / 'Worm_Atlas' / 'Altun_lineage_corrected.xlsx'

## FlyWire
fly_wire = DOWNLOAD_DIR / 'FlyWire'

## Atanas whole brain
atanas_whole_brain = {
    'Control': DOWNLOAD_DIR / 'Atanas_2023' / 'Control',
    'Heat': DOWNLOAD_DIR / 'Atanas_2023' / 'Heat'
}

# Winding connectome - Fly Larva Instar-1
winding_connectome = DOWNLOAD_DIR / 'Winding_2023'

# Ciona connectome
ciona_connectome = DOWNLOAD_DIR / 'Ryan_2016'

# Pristionchus pacificus pharyngeal connectome
pristionchus_pharynx = DOWNLOAD_DIR / 'Bumbarger_2013'

# Platynereis dumerilii whole-body synaptic connectome (3-day larva)
veraszto_connectome = DOWNLOAD_DIR / 'Veraszto_2025'

# Brittin et al. 2018 C. elegans nerve-ring contactome (adult + L4)
brittin_contactome = DOWNLOAD_DIR / 'Brittin_2018'

# Skuhersky et al. 2022 NeuroPAL-derived C. elegans 3D anatomical atlas
skuhersky_neuropal = DOWNLOAD_DIR / 'Skuhersky_2022'

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

# Atanas et al
atanas_links = {'Control' : ['2022-06-14-01.json', '2022-07-20-01.json', '2023-01-19-22.json',
'2022-06-14-07.json', '2022-07-26-01.json',	'2023-01-23-01.json',
'2022-06-14-13.json', '2022-08-02-01.json',	'2023-01-23-08.json',
'2022-06-28-01.json', '2023-01-09-28.json',	'2023-01-23-15.json',
'2022-06-28-07.json', '2023-01-17-01.json',	'2023-01-23-21.json',
'2022-07-15-06.json', '2023-01-19-08.json',	'2023-03-07-01.json',
'2022-07-15-12.json', '2023-01-19-15.json'], 'Heat' :  ['2022-12-21-06.json',	'2023-01-09-15.json',	'2023-01-16-15.json',
'2023-01-05-01.json',	'2023-01-09-22.json',	'2023-01-16-22.json',
'2023-01-05-18.json',	'2023-01-10-07.json',	'2023-01-17-07.json',
'2023-01-06-01.json',	'2023-01-10-14.json',	'2023-01-17-14.json',
'2023-01-06-08.json',	'2023-01-13-07.json',	'2023-01-18-01.json',
'2023-01-06-15.json',	'2023-01-16-01.json', '2023-01-09-08.json',	'2023-01-16-08.json']}
atanas_link_prefix = 'https://storage.googleapis.com/www-deploy-bucket/activity/atanas_kim_2023/'

neuropeptide_atlas_links = ['01022024_neuropeptide_connectome_long_range_model.csv']

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
    'cook_connectome' : [
        'https://doi.org/10.1038/s41586-019-1352-7',  # Cook et al. 2019 Nature — main reconstruction
        'https://doi.org/10.1098/rstb.1986.0056',     # White et al. 1986 — upstream somatic EM
        'https://doi.org/10.1098/rstb.1976.0085',     # Albertson & Thomson 1976 — upstream pharyngeal EM
        'https://doi.org/10.1002/cne.24932',          # Cook et al. 2020 JCN — pharynx re-analysis
    ],
    'witvliet_connectome': ['https://doi.org/10.1038/s41586-021-03778-8'],
    'sig_prop_atlas': ['https://doi.org/10.1038/s41586-023-06683-4'],
    'neuropeptide_atlas': ['https://doi.org/10.1016/j.neuron.2023.09.043'],
    'neurotransmitter_atlas': ['https://doi.org/10.7554/eLife.95402.3'],
    'atanas_whole_brain': ['https://doi.org/10.1016/j.cell.2023.07.035'],
    'neuropal_positions': [
        'https://doi.org/10.1186/s12859-022-04738-3',
        'https://doi.org/10.1016/j.cell.2020.12.012',
    ],
    'altun_neurotrasmitters_receptors': ['https://doi.org/10.3908/wormatlas.5.202'],
    'worm_lineage': ['https://doi.org/10.1016/0012-1606(77)90158-0'],
    'winding_connectome': ['https://doi.org/10.1126/science.add9330'], 
    'ciona_connectome': ['https://doi.org/10.7554/eLife.16962'],
    'bumbarger_pharynx': ['https://doi.org/10.1016/j.cell.2012.12.013'],
    'white_connectome': ['https://doi.org/10.1098/rstb.1986.0056'],
    'veraszto_connectome': ['https://doi.org/10.7554/eLife.97964'],
    'brittin_contactome': ['https://doi.org/10.1038/s41586-021-03284-x'],
    }

# Per-dataset capability matrix. A "dataset" is a specific
# (species, sex, stage, style) tuple — e.g. C. elegans Cook hermaphrodite-adult
# and Cook male-adult are separate datasets because the layered data available
# for each is different.
#
# `capabilities` declares which CeDNe data layers exist for the dataset in
# principle (regardless of whether a loader is wired up yet). The web UI uses
# this to gray out unsupported controls; programmatic users can enumerate it
# to discover what's loadable for a given organism.
#
# Capability keys:
#   connectome        — synaptic wiring (the make_* / style itself)
#   neurotransmitters — per-neuron transmitter assignments
#   neuropeptides     — per-neuron neuropeptide expression (Ripoll-Sanchez 2023)
#   transcriptome     — per-neuron transcript expression (CeNGEN)
#   contactome        — physical adjacency (Brittin 2018)
#   lineage           — developmental lineage (Altun, via load_lineage)
#   position          — anatomical coordinates per neuron (drives Anatomical /
#                       2D Map / 3D Map layouts in the web UI)
#
# Use None for cells whose status has not been verified — please confirm before
# relying on them.

organism_datasets = [
    # === Caenorhabditis elegans ===
    {'species': 'Caenorhabditis elegans', 'common_name': 'worm',
     'sex': 'hermaphrodite', 'stage': 'adult', 'style': 'cook',
     'capabilities': {
         'connectome': True, 'neurotransmitters': True, 'neuropeptides': True,
         'transcriptome': True, 'contactome': True, 'lineage': True,
         'position': True}},
    # Witvliet 2020 — 8 specimens across 4 stages (L1, L2, L3, adult).
    # Replicates are surfaced via the loader's `dataset_ind` argument and are
    # not enumerated as separate matrix rows since their capabilities are
    # identical within a stage.
    {'species': 'Caenorhabditis elegans', 'common_name': 'worm',
     'sex': 'hermaphrodite', 'stage': 'L1', 'style': 'witvliet',
     'capabilities': {
         'connectome': True, 'neurotransmitters': True, 'neuropeptides': True,
         'transcriptome': True,
         'contactome': False,   # Brittin 2018 only released adult + L4 contactomes
         'lineage': True,
         'position': False}},   # witvliet loader does not populate node.position
    {'species': 'Caenorhabditis elegans', 'common_name': 'worm',
     'sex': 'hermaphrodite', 'stage': 'L2', 'style': 'witvliet',
     'capabilities': {
         'connectome': True, 'neurotransmitters': True, 'neuropeptides': True,
         'transcriptome': True, 'contactome': False, 'lineage': True,
         'position': False}},
    {'species': 'Caenorhabditis elegans', 'common_name': 'worm',
     'sex': 'hermaphrodite', 'stage': 'L3', 'style': 'witvliet',
     'capabilities': {
         'connectome': True, 'neurotransmitters': True, 'neuropeptides': True,
         'transcriptome': True, 'contactome': False, 'lineage': True,
         'position': False}},
    {'species': 'Caenorhabditis elegans', 'common_name': 'worm',
     'sex': 'hermaphrodite', 'stage': 'adult', 'style': 'witvliet',
     'capabilities': {
         'connectome': True, 'neurotransmitters': True, 'neuropeptides': True,
         'transcriptome': True, 'contactome': True, 'lineage': True,
         'position': False}},
    {'species': 'Caenorhabditis elegans', 'common_name': 'worm',
     'sex': 'male', 'stage': 'adult', 'style': 'cook',
     'capabilities': {
         'connectome': True,
         'neurotransmitters': True,   # Wang 2024 ligand-table.xlsx has a Male sheet
         'neuropeptides': False,      # Ripoll-Sanchez 2023 is hermaphrodite-only
         'transcriptome': False,      # CeNGEN is hermaphrodite-only here, but new version has been updated with other datasets.
         'contactome': False,         # Brittin 2018 sampled hermaphrodites only
         'lineage': None,             # Altun workbook may have a Male sheet — please verify
         'position': False}},         # cook loader only populates position for hermaphrodite

    # === Drosophila melanogaster ===
    {'species': 'Drosophila melanogaster', 'common_name': 'fruit fly',
     'sex': 'female', 'stage': 'adult', 'style': 'fly_wire',
     'capabilities': {
         'connectome': True,
         'neurotransmitters': True,   # EM-inferred (Eckstein et al.) and shipped via FlyWire
         'neuropeptides': False, 'transcriptome': False,
         'contactome': False, 'lineage': False,
         'position': None}},          # FlyWire ships 3D coords; loader wiring not yet verified
    {'species': 'Drosophila melanogaster', 'common_name': 'fruit fly larva',
     'sex': '', 'stage': 'L1', 'style': 'Winding_2023',
     'capabilities': {
         'connectome': True,
         'neurotransmitters': None,   # please verify whether Winding 2023 ships NT calls
         'neuropeptides': False, 'transcriptome': False,
         'contactome': False, 'lineage': False,
         'position': False}},

    # === Other species ===
    {'species': 'Ciona intestinalis', 'common_name': 'sea squirt',
     'sex': '', 'stage': 'larva', 'style': 'ryan',
     'capabilities': {
         'connectome': True, 'neurotransmitters': False, 'neuropeptides': False,
         'transcriptome': False, 'contactome': False, 'lineage': False,
         'position': True}},          # ~57% of neurons carry numpy-array coords
    {'species': 'Pristionchus pacificus', 'common_name': 'pristionchus (pharynx only)',
     'sex': 'hermaphrodite', 'stage': 'adult', 'style': 'bumbarger',
     'capabilities': {
         'connectome': True, 'neurotransmitters': False, 'neuropeptides': False,
         'transcriptome': False, 'contactome': False, 'lineage': False,
         'position': False}},
    {'species': 'Platynereis dumerilii', 'common_name': 'ragworm',
     'sex': '', 'stage': '3-day larva', 'style': 'veraszto',
     'capabilities': {
         'connectome': True, 'neurotransmitters': False, 'neuropeptides': False,
         'transcriptome': False, 'contactome': False, 'lineage': False,
         'position': False}},
]
