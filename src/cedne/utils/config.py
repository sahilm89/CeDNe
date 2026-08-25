"""
Configuration file for CeDNe utils module"""

__author__ = "Sahil Moza"
__date__ = "2025-04-06"
__license__ = "MIT"

import datetime
import cedne as ced
from cedne.core.source import Citation
from pathlib import Path

root_path = Path(ced.__path__[0])

TOPDIR = root_path.parents[1]
DATADIR = TOPDIR / "data_sources"
# Historically ``DOWNLOAD_DIR`` sat one level below ``DATADIR`` at
# ``data_sources/downloads/``. The intermediate directory was removed so
# raw fetches (Cook_2019/, FlyWire/, ...) and derived bakes
# (FlyWire/baked/, etc.) can share one persistent-volume mount cleanly.
# The variable name is preserved as an alias so every existing
# ``DOWNLOAD_DIR / "X"`` reference downstream keeps working unchanged.
DOWNLOAD_DIR = DATADIR
OUTPUT_DIR = TOPDIR / "Output" / f'{str(datetime.datetime.now()).split(" ")[0]}'

prefix_NT = "Wang_2024/"
prefix_CENGEN = "CENGEN/"
prefix_NP = "Ripoll-Sanchez_2023/"
prefix_synaptic_weights = "Randi_2023/"  # Signal propagation atlas

# Dataset directories first — order matters because the per-file paths
# below resolve through ``cook_connectome`` etc.
cook_connectome = DOWNLOAD_DIR / "Cook_2019"

## Loading and building functions
# Pre-processed C. elegans default-worm bundle. Lives inside Cook_2019/
# (or Worm_Atlas/ for positions, which are NeuroPAL-derived per
# Yemini 2021). Previously these sat at the data_sources root as
# "stray" files; co-locating with their source dataset makes the
# provenance chain explicit.
cell_list = cook_connectome / "Cell_list.pkl"
chemsyns = cook_connectome / "chem_adj.pkl"
elecsyns = cook_connectome / "gapjn_symm_adj.pkl"
neuronPositions = DOWNLOAD_DIR / "Worm_Atlas" / "neuronPosition.pkl"

# One-shot cache for the Cook male loader. The raw supplementary workbook is
# slow to parse (~5s via openpyxl); we preprocess it into a single pickle
# (labels, type dict, category dict, sparse chem adj, sparse gap-jn adj) the
# first time makeWorm(style='cook', sex='male') runs, and read from that
# cache on every subsequent call.
cook_male_cache = cook_connectome / "cook_male_data.pkl"
witvliet_connectome = DOWNLOAD_DIR / "witvliet_2020"
white_connectome = DOWNLOAD_DIR / "white_1986"

lineage = DOWNLOAD_DIR / "Worm_Atlas" / "Altun_lineage_corrected.xlsx"

## FlyWire
fly_wire = DOWNLOAD_DIR / "FlyWire"

## Atanas whole brain
atanas_whole_brain = {
    "Control": DOWNLOAD_DIR / "Atanas_2023" / "Control",
    "Heat": DOWNLOAD_DIR / "Atanas_2023" / "Heat",
}

# Winding connectome - Fly Larva Instar-1
winding_connectome = DOWNLOAD_DIR / "Winding_2023"

# Ciona connectome
ciona_connectome = DOWNLOAD_DIR / "Ryan_2016"

# Pristionchus pacificus pharyngeal connectome
pristionchus_pharynx = DOWNLOAD_DIR / "Bumbarger_2013"

# Platynereis dumerilii whole-body synaptic connectome (3-day larva)
veraszto_connectome = DOWNLOAD_DIR / "Veraszto_2025"

# Brittin et al. 2018 C. elegans nerve-ring contactome (adult + L4)
brittin_contactome = DOWNLOAD_DIR / "Brittin_2018"

# Skuhersky et al. 2022 NeuroPAL-derived C. elegans 3D anatomical atlas
skuhersky_neuropal = DOWNLOAD_DIR / "Skuhersky_2022"

# Download Links
# CENGEN
cengen_links = [
    "https://cengen.org/storage/021821_liberal_threshold1.csv",
    "https://cengen.org/storage/021821_medium_threshold2.csv",
    "https://cengen.org/storage/021821_conservative_threshold3.csv",
    "https://cengen.org/storage/021821_stringent_threshold4.csv",
]

# FlyWire
flywire_links = [
    "https://codex.flywire.ai/api/download?data_product=consolidated_cell_types&data_version=783&dataset=fafb",
    "https://codex.flywire.ai/api/download?data_product=classification&data_version=783&dataset=fafb",
    "https://codex.flywire.ai/api/download?data_product=cell_stats&data_version=783&dataset=fafb",
    "https://codex.flywire.ai/api/download?data_product=connections_princeton_no_threshold&data_version=783&dataset=fafb",
    "https://codex.flywire.ai/api/download?data_product=coordinates&data_version=783&dataset=fafb",
    "https://codex.flywire.ai/api/download?data_product=names&data_version=783&dataset=fafb"
    "https://codex.flywire.ai/api/download?data_product=neurons&data_version=783&dataset=fafb",
]

# Atanas et al
atanas_links = {
    "Control": [
        "2022-06-14-01.json",
        "2022-07-20-01.json",
        "2023-01-19-22.json",
        "2022-06-14-07.json",
        "2022-07-26-01.json",
        "2023-01-23-01.json",
        "2022-06-14-13.json",
        "2022-08-02-01.json",
        "2023-01-23-08.json",
        "2022-06-28-01.json",
        "2023-01-09-28.json",
        "2023-01-23-15.json",
        "2022-06-28-07.json",
        "2023-01-17-01.json",
        "2023-01-23-21.json",
        "2022-07-15-06.json",
        "2023-01-19-08.json",
        "2023-03-07-01.json",
        "2022-07-15-12.json",
        "2023-01-19-15.json",
    ],
    "Heat": [
        "2022-12-21-06.json",
        "2023-01-09-15.json",
        "2023-01-16-15.json",
        "2023-01-05-01.json",
        "2023-01-09-22.json",
        "2023-01-16-22.json",
        "2023-01-05-18.json",
        "2023-01-10-07.json",
        "2023-01-17-07.json",
        "2023-01-06-01.json",
        "2023-01-10-14.json",
        "2023-01-17-14.json",
        "2023-01-06-08.json",
        "2023-01-13-07.json",
        "2023-01-18-01.json",
        "2023-01-06-15.json",
        "2023-01-16-01.json",
        "2023-01-09-08.json",
        "2023-01-16-08.json",
    ],
}
atanas_link_prefix = (
    "https://storage.googleapis.com/www-deploy-bucket/activity/atanas_kim_2023/"
)

neuropeptide_atlas_links = ["01022024_neuropeptide_connectome_long_range_model.csv"]

## Citation Tables
# Where titles are known with confidence, citations are stored as structured
# ``Citation`` objects so the web UI renders them as "Title (Year)". Bare URL
# strings fall back to "doi:..." rendering in the UI — promote them to
# Citation(...) when a confident title is available.
_white1986 = Citation(
    key="white1986",
    title="The structure of the nervous system of the nematode Caenorhabditis elegans",
    authors=["White JG", "Southgate E", "Thomson JN", "Brenner S"],
    year=1986,
    doi="10.1098/rstb.1986.0056",
    url="https://doi.org/10.1098/rstb.1986.0056",
)
_yemini2021_neuropal = Citation(
    key="yemini2021_neuropal",
    title="NeuroPAL: A Multicolor Atlas for Whole-Brain Neuronal Identification in C. elegans",
    authors=["Yemini E", "Lin A", "Nejatbakhsh A", "et al."],
    year=2021,
    doi="10.1016/j.cell.2020.12.012",
    url="https://doi.org/10.1016/j.cell.2020.12.012",
)

citations = {
    "cengen": [
        Citation(
            key="hammarlund2018_cengen",
            title="The CeNGEN Project (Caenorhabditis elegans Neuronal Gene Expression Network)",
            authors=["Hammarlund M", "Hobert O", "Miller DM III", "Sestan N"],
            year=2018,
            doi="10.1016/j.neuron.2018.07.042",
            url="https://doi.org/10.1016/j.neuron.2018.07.042",
        ),
        Citation(
            key="taylor2021_cengen",
            title="Molecular topography of an entire nervous system",
            authors=["Taylor SR", "Santpere G", "Weinreb A", "et al."],
            year=2021,
            doi="10.1016/j.cell.2021.06.023",
            url="https://doi.org/10.1016/j.cell.2021.06.023",
        ),
        Citation(
            key="taylor2025_cengen",
            title="A gene expression atlas of a juvenile nervous system",
            authors=["Taylor SR", "Olson Claire", "Ripoll-Sanchez Lidia", "et al."],
            year=2025,
            doi="10.1101/2025.11.21.689793",
            url="https://doi.org/10.1101/2025.11.21.689793",
        ),
    ],
    "fly_wire": [
        "https://doi.org/10.1038/s41586-024-07558-y",
        "https://doi.org/10.1038/s41586-024-07686-5",
        "https://doi.org/10.1016/j.cell.2018.06.019",
        "https://doi.org/10.1038/s41592-021-01183-7",
        "https://doi.org/10.1007/978-3-030-00934-2_36",
        "https://doi.org/10.1016/j.cell.2024.03.016",
        "https://doi.org/10.1038/s41586-024-07981-1",
        "https://doi.org/10.1038/s41586-024-07968-y",
        "https://www.biorxiv.org/content/10.1101/2025.06.10.658788",
    ],
    "cook_connectome": [
        Citation(
            key="cook2019",
            title="Whole-animal connectomes of both Caenorhabditis elegans sexes",
            authors=["Cook SJ", "Jarrell TA", "Brittin CA", "et al."],
            year=2019,
            doi="10.1038/s41586-019-1352-7",
            url="https://doi.org/10.1038/s41586-019-1352-7",
            notes="Main reconstruction",
        ),
        _white1986,
        Citation(
            key="albertson1976",
            title="The pharynx of Caenorhabditis elegans",
            authors=["Albertson DG", "Thomson JN"],
            year=1976,
            doi="10.1098/rstb.1976.0085",
            url="https://doi.org/10.1098/rstb.1976.0085",
            notes="Upstream pharyngeal EM",
        ),
        Citation(
            key="cook2020_pharynx",
            title="The connectome of the Caenorhabditis elegans pharynx",
            authors=[
                "Cook SJ",
                "Crouse CM",
                "Yemini E",
                "Hall DH",
                "Emmons SW",
                "Hobert O",
            ],
            year=2020,
            doi="10.1002/cne.24932",
            url="https://doi.org/10.1002/cne.24932",
            notes="Pharynx re-analysis",
        ),
    ],
    "witvliet_connectome": [
        Citation(
            key="witvliet2021",
            title="Connectomes across development reveal principles of brain maturation",
            authors=["Witvliet D", "Mulcahy B", "Mitchell JK", "et al."],
            year=2021,
            doi="10.1038/s41586-021-03778-8",
            url="https://doi.org/10.1038/s41586-021-03778-8",
        ),
    ],
    "sig_prop_atlas": [
        Citation(
            key="randi2023",
            title="Neural signal propagation atlas of Caenorhabditis elegans",
            authors=["Randi F", "Sharma AK", "Dvali S", "Leifer AM"],
            year=2023,
            doi="10.1038/s41586-023-06683-4",
            url="https://doi.org/10.1038/s41586-023-06683-4",
        ),
    ],
    "neuropeptide_atlas": [
        Citation(
            key="ripollsanchez2023",
            title="The neuropeptidergic connectome of C. elegans",
            authors=["Ripoll-Sanchez L", "Watteyne J", "Sun H", "et al."],
            year=2023,
            doi="10.1016/j.neuron.2023.09.043",
            url="https://doi.org/10.1016/j.neuron.2023.09.043",
        ),
    ],
    "neurotransmitter_atlas": [
        Citation(
            key="wang2024_nt_atlas",
            title="A neurotransmitter atlas of C. elegans males and hermaphrodites",
            authors=["Wang C", "Vidal B", "Sural S", "et al."],
            year=2023,
            doi="10.7554/eLife.95402.3",
            url="https://doi.org/10.7554/eLife.95402.3",
        ),
    ],
    "atanas_whole_brain": [
        Citation(
            key="atanas2023",
            title="Brain-wide representations of behavior spanning multiple timescales and states in C. elegans",
            authors=["Atanas AA", "Kim J", "Wang Z", "et al."],
            year=2023,
            doi="10.1016/j.cell.2023.07.035",
            url="https://doi.org/10.1016/j.cell.2023.07.035",
        ),
    ],
    "neuropal_positions": [
        # Two papers in this entry; the canonical NeuroPAL paper is Yemini 2021.
        "https://doi.org/10.1186/s12859-022-04738-3",
        _yemini2021_neuropal,
    ],
    "altun_neurotransmitters_receptors": [
        Citation(
            key="altun_wormatlas_nt_receptors",
            title="Nervous System, Neurotransmitter Receptors",
            authors=["Altun ZF", "Hall DH"],
            year=2011,
            doi="10.3908/wormatlas.5.202",
            url="https://doi.org/10.3908/wormatlas.5.202",
            notes="WormAtlas chapter",
        ),
    ],
    # Back-compat alias for the historical misspelling.
    "altun_neurotrasmitters_receptors": [
        Citation(
            key="altun_wormatlas_nt_receptors",
            title="Nervous System, Neurotransmitter Receptors",
            authors=["Altun ZF", "Hall DH"],
            year=2011,
            doi="10.3908/wormatlas.5.202",
            url="https://doi.org/10.3908/wormatlas.5.202",
            notes="WormAtlas chapter",
        ),
    ],
    "worm_lineage": [
        Citation(
            key="sulston1977",
            title="Post-embryonic cell lineages of the nematode, Caenorhabditis elegans",
            authors=["Sulston JE", "Horvitz HR"],
            year=1977,
            doi="10.1016/0012-1606(77)90158-0",
            url="https://doi.org/10.1016/0012-1606(77)90158-0",
        ),
    ],
    "winding_connectome": [
        Citation(
            key="winding2023",
            title="The connectome of an insect brain",
            authors=["Winding M", "Pedigo BD", "Barnes CL", "et al."],
            year=2023,
            doi="10.1126/science.add9330",
            url="https://doi.org/10.1126/science.add9330",
        ),
    ],
    "ciona_connectome": [
        Citation(
            key="ryan2016_ciona",
            title="The CNS connectome of a tadpole larva of Ciona intestinalis (L.) highlights sidedness in the brain of a chordate sibling",
            authors=["Ryan K", "Lu Z", "Meinertzhagen IA"],
            year=2016,
            doi="10.7554/eLife.16962",
            url="https://doi.org/10.7554/eLife.16962",
        ),
    ],
    "bumbarger_pharynx": [
        Citation(
            key="bumbarger2013",
            title="System-wide Rewiring Underlies Behavioral Differences in Predatory and Bacterial-Feeding Nematodes",
            authors=["Bumbarger DJ", "Riebesell M", "Rödelsperger C", "Sommer RJ"],
            year=2013,
            doi="10.1016/j.cell.2012.12.013",
            url="https://doi.org/10.1016/j.cell.2012.12.013",
        ),
    ],
    "white_connectome": [_white1986],
    "veraszto_connectome": Citation(
        key="veraszto2025",
        title="Whole-body connectome of a segmented annelid larva",
        authors=["Verasztó C", "Jasek S", "Gühmann M", "et al."],
        year=2025,
        doi="10.7554/eLife.97964",
        url="https://doi.org/10.7554/eLife.97964",
    ),
    "brittin_contactome": [
        Citation(
            key="brittin2021",
            title="A multi-scale brain map derived from whole-brain volumetric reconstructions",
            authors=["Brittin CA", "Cook SJ", "Hall DH", "Emmons SW", "Cohen N"],
            year=2021,
            doi="10.1038/s41586-021-03284-x",
            url="https://doi.org/10.1038/s41586-021-03284-x",
        ),
    ],
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
#   fold_lr           — Left/Right symmetry fold. graphtools.foldLeftRight
#                       relies on the C. elegans naming convention (trailing
#                       L/R suffix), so this is worm-only even though other
#                       organisms have bilateral symmetry biologically.
#   fold_dv           — Dorso/Ventral fold (same naming-convention caveat as
#                       fold_lr).
#   fold_category     — Fold neurons by their ``neuron.category`` attribute
#                       (anatomical body region for C. elegans: HEAD,
#                       MIDBODY, TAIL, PHARYNX, SEX SPECIFIC). Populated only
#                       by the C. elegans loaders today, so this is worm-only.
#   atanas_recordings — Calcium recordings from Atanas 2023; the released
#                       dataset is C. elegans hermaphrodite.
#
# Use None for cells whose status has not been verified — please confirm before
# relying on them.

organism_datasets = [
    # === Caenorhabditis elegans ===
    {
        "species": "Caenorhabditis elegans",
        "common_name": "worm",
        "sex": "hermaphrodite",
        "stage": "adult",
        "style": "cook",
        "capabilities": {
            "connectome": True,
            "neurotransmitters": True,
            "neuropeptides": True,
            "transcriptome": True,
            "contactome": True,
            "lineage": True,
            "position": True,
            "fold_lr": True,
            "fold_dv": True,
            "fold_category": True,
            "atanas_recordings": True,
        },
    },
    # Witvliet 2020 — 8 specimens across 4 stages (L1, L2, L3, adult).
    # Replicates are surfaced via the loader's `dataset_ind` argument and are
    # not enumerated as separate matrix rows since their capabilities are
    # identical within a stage.
    {
        "species": "Caenorhabditis elegans",
        "common_name": "worm",
        "sex": "hermaphrodite",
        "stage": "L1",
        "style": "witvliet",
        "capabilities": {
            "connectome": True,
            "neurotransmitters": True,
            "neuropeptides": True,
            "transcriptome": True,
            "contactome": False,  # Brittin 2018 only released adult + L4 contactomes
            "lineage": True,
            "position": False,
            "fold_lr": True,
            "fold_dv": True,
            "fold_category": True,
            "atanas_recordings": True,
        },
    },  # witvliet loader does not populate node.position
    {
        "species": "Caenorhabditis elegans",
        "common_name": "worm",
        "sex": "hermaphrodite",
        "stage": "L2",
        "style": "witvliet",
        "capabilities": {
            "connectome": True,
            "neurotransmitters": True,
            "neuropeptides": True,
            "transcriptome": True,
            "contactome": False,
            "lineage": True,
            "position": False,
            "fold_lr": True,
            "fold_dv": True,
            "fold_category": True,
            "atanas_recordings": True,
        },
    },
    {
        "species": "Caenorhabditis elegans",
        "common_name": "worm",
        "sex": "hermaphrodite",
        "stage": "L3",
        "style": "witvliet",
        "capabilities": {
            "connectome": True,
            "neurotransmitters": True,
            "neuropeptides": True,
            "transcriptome": True,
            "contactome": False,
            "lineage": True,
            "position": False,
            "fold_lr": True,
            "fold_dv": True,
            "fold_category": True,
            "atanas_recordings": True,
        },
    },
    {
        "species": "Caenorhabditis elegans",
        "common_name": "worm",
        "sex": "hermaphrodite",
        "stage": "adult",
        "style": "witvliet",
        "capabilities": {
            "connectome": True,
            "neurotransmitters": True,
            "neuropeptides": True,
            "transcriptome": True,
            "contactome": True,
            "lineage": True,
            "position": False,
            "fold_lr": True,
            "fold_dv": True,
            "fold_category": True,
            "atanas_recordings": True,
        },
    },
    {
        "species": "Caenorhabditis elegans",
        "common_name": "worm",
        "sex": "male",
        "stage": "adult",
        "style": "cook",
        "capabilities": {
            "connectome": True,
            "neurotransmitters": True,  # Wang 2024 ligand-table.xlsx has a Male sheet
            "neuropeptides": False,  # Ripoll-Sanchez 2023 is hermaphrodite-only
            "transcriptome": False,  # CeNGEN is hermaphrodite-only here, but new version has been updated with other datasets.
            "contactome": False,  # Brittin 2018 sampled hermaphrodites only
            "lineage": None,  # Altun workbook may have a Male sheet — please verify
            "position": False,
            # Male loader uses the same L/R/D/V naming as hermaphrodite,
            # so the symmetry folds should work; but the male connectome
            # is asymmetric (sex-specific neurons on one side only) and
            # has not been spot-checked. Mark None until verified.
            "fold_lr": None,
            "fold_dv": None,
            # Male cook loader populates ``category`` (from list_2 / l1)
            # the same way as the hermaphrodite path, so category fold
            # is structurally supported. The categories ("HEAD" /
            # "MIDBODY" / "TAIL" / "SEX SPECIFIC") are male-specific in
            # composition but the fold itself is well-defined.
            "fold_category": True,
            "atanas_recordings": False,  # Atanas 2023 is hermaphrodite-only
        },
    },  # cook loader only populates position for hermaphrodite
    # === Drosophila melanogaster ===
    {
        "species": "Drosophila melanogaster",
        "common_name": "fruit fly",
        "sex": "female",
        "stage": "adult",
        "style": "fly_wire",
        "capabilities": {
            "connectome": True,
            # The `neurotransmitters` capability gates the web UI's "Load
            # Putative Neurotransmitters" button, which calls the worm-
            # specific Wang 2024 + CENGEN pipeline. FlyWire ships EM-inferred
            # NT calls (Eckstein et al.) — those are a different pipeline,
            # not exposed via /load/neurotransmitters today, so the button
            # would 500 here. Wire them up in a separate capability key if
            # we ever surface them in the UI.
            "neurotransmitters": False,
            "neuropeptides": False,
            "transcriptome": False,
            "contactome": False,
            "lineage": False,
            "position": None,
            "fold_lr": False,  # FlyWire neuron IDs don't follow the L/R suffix convention
            "fold_dv": False,
            "fold_category": False,  # FlyWire loader does not populate ``category``
            "atanas_recordings": False,
        },
    },  # FlyWire ships 3D coords; loader wiring not yet verified
    {
        "species": "Drosophila melanogaster",
        "common_name": "fruit fly larva",
        "sex": "",
        "stage": "L1",
        "style": "Winding_2023",
        "capabilities": {
            "connectome": True,
            # /load/neurotransmitters runs the Wang 2024 + CENGEN pipeline,
            # which is C. elegans-only. Even if Winding 2023 carries its own
            # NT calls, they're not loadable through this endpoint, so the
            # button must stay disabled here.
            "neurotransmitters": False,
            "neuropeptides": False,
            "transcriptome": False,
            "contactome": False,
            "lineage": False,
            "position": False,
            "fold_lr": False,
            "fold_dv": False,
            "fold_category": False,
            "atanas_recordings": False,
        },
    },
    # === Other species ===
    {
        "species": "Ciona intestinalis",
        "common_name": "sea squirt",
        "sex": "",
        "stage": "larva",
        "style": "ryan",
        "capabilities": {
            "connectome": True,
            "neurotransmitters": False,
            "neuropeptides": False,
            "transcriptome": False,
            "contactome": False,
            "lineage": False,
            "position": True,
            "fold_lr": False,
            "fold_dv": False,
            "fold_category": False,
            "atanas_recordings": False,
        },
    },  # ~57% of neurons carry numpy-array coords
    {
        "species": "Pristionchus pacificus",
        "common_name": "pristionchus (pharynx only)",
        "sex": "hermaphrodite",
        "stage": "adult",
        "style": "bumbarger",
        "capabilities": {
            "connectome": True,
            "neurotransmitters": False,
            "neuropeptides": False,
            "transcriptome": False,
            "contactome": False,
            "lineage": False,
            "position": False,
            "fold_lr": False,
            "fold_dv": False,
            "fold_category": False,
            "atanas_recordings": False,
        },
    },
    {
        "species": "Platynereis dumerilii",
        "common_name": "ragworm",
        "sex": "",
        "stage": "3-day larva",
        "style": "veraszto",
        "capabilities": {
            "connectome": True,
            "neurotransmitters": False,
            "neuropeptides": False,
            "transcriptome": False,
            "contactome": False,
            "lineage": False,
            "position": False,
            "fold_lr": False,
            "fold_dv": False,
            "fold_category": False,
            "atanas_recordings": False,
        },
    },
]
