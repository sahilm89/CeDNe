import pickle
import json
from types import SimpleNamespace

import networkx as nx
import numpy as np
import pandas as pd
import pytest

from cedne.core.animal import Worm
from cedne.core.network import NervousSystem
from cedne.utils import graphtools, loader


class FoldSpyNetwork:
    def __init__(self, neuron_names):
        self.neurons = {name: object() for name in neuron_names}

    def fold_network(self, fold_by, **kwargs):
        return {"fold_by": fold_by, **kwargs}


def write_pickle(path, payload):
    with open(path, "wb") as handle:
        pickle.dump(payload, handle)


def find_connection(network, pre, post, connection_type):
    for connection in network.connections.values():
        if (
            connection.pre.name == pre
            and connection.post.name == post
            and connection.connection_type == connection_type
        ):
            return connection
    raise AssertionError(f"Connection {pre}->{post} ({connection_type}) not found")


@pytest.mark.parametrize(
    ("sex", "sheet_name"),
    [
        ("Hermaphrodite", "Hermaphrodite, sorted by neuron"),
        ("male", "Male neurons, sorted by neuron"),
    ],
)
def test_getLigands_reads_expected_sheet_and_canonicalizes(
    monkeypatch, sex, sheet_name
):
    calls = []

    def fake_read_excel(path, sheet_name=None, skiprows=None, engine=None):
        calls.append(
            {
                "path": path,
                "sheet_name": sheet_name,
                "skiprows": skiprows,
                "engine": engine,
            }
        )
        return pd.DataFrame(
            {
                "Neuron": ["AVAL"],
                "Neurotransmitter 1": [" ACh "],
                "Neurotransmitter 2": ["Glu"],
            }
        )

    monkeypatch.setattr(loader.pd, "read_excel", fake_read_excel)

    assert loader.getLigands("AVAL", sex=sex) == ["Acetylcholine", "Glutamate"]
    assert calls == [
        {
            "path": loader.DOWNLOAD_DIR / loader.prefix_NT / "ligand-table.xlsx",
            "sheet_name": sheet_name,
            "skiprows": 7,
            "engine": "openpyxl",
        }
    ]


def test_getLigands_ignores_blank_and_non_string_entries(monkeypatch):
    def fake_read_excel(*args, **kwargs):
        return pd.DataFrame(
            {
                "Neuron": ["AVAL"],
                "Neurotransmitter 1": [np.nan],
                "Neurotransmitter 2": [" "],
            }
        )

    monkeypatch.setattr(loader.pd, "read_excel", fake_read_excel)

    assert loader.getLigands("AVAL") == []


def test_getLigands_rejects_invalid_sex():
    with pytest.raises(ValueError, match="Sex must be 'Hermaphrodite' or 'Male'"):
        loader.getLigands("AVAL", sex="Unknown")


def test_getLigandsAndReceptors_canonicalizes_and_handles_missing_ligands():
    npr = pd.DataFrame(
        {
            "gene_name": ["acr-1", "orphan-rec"],
            "AVA": [True, True],
        }
    )
    ligmap = pd.DataFrame(
        {
            "gene": ["acr-1"],
            "ligand": ["ACh"],
        }
    )

    assert loader.getLigandsAndReceptors(npr, ligmap, "AVA") == {
        "acr-1": "Acetylcholine",
        "orphan-rec": "",
    }


@pytest.mark.parametrize(
    ("chem_only", "gapjn_only", "expected_types"),
    [
        (False, False, {"chemical-synapse", "gap-junction"}),
        (True, False, {"chemical-synapse"}),
        (False, True, {"gap-junction"}),
    ],
)
def test_build_nervous_system_respects_connection_flags(
    tmp_path, chem_only, gapjn_only, expected_types
):
    neuron_data = tmp_path / "neurons.pkl"
    chem_synapses = tmp_path / "chemical.pkl"
    elec_synapses = tmp_path / "gap.pkl"
    positions = tmp_path / "positions.pkl"

    write_pickle(
        neuron_data,
        pd.DataFrame(
            [
                ["AVAL", "sensory", "head", "chemosensory"],
                ["AVAR", "interneuron", "head", "integrative"],
            ]
        ),
    )
    write_pickle(chem_synapses, {"AVAL": {"AVAR": {"weight": 2}}, "AVAR": {}})
    write_pickle(elec_synapses, {"AVAR": {"AVAL": {"weight": 3}}, "AVAL": {}})
    write_pickle(
        positions,
        {
            "AVAL": np.array([0.0, 0.0, 0.0]),
            "AVAR": np.array([1.0, 0.0, 0.0]),
        },
    )

    nn = NervousSystem(Worm("test-worm"), network="TestNetwork")
    loader.build_nervous_system(
        nn,
        neuron_data=neuron_data,
        chem_synapses=chem_synapses,
        elec_synapses=elec_synapses,
        positions=positions,
        chem_only=chem_only,
        gapjn_only=gapjn_only,
    )

    assert set(nn.neurons) == {"AVAL", "AVAR"}
    assert np.array_equal(nn.neurons["AVAL"].position, np.array([0.0, 0.0, 0.0]))
    assert {conn.connection_type for conn in nn.connections.values()} == expected_types


def test_build_nervous_system_cites_default_neuropal_positions(tmp_path, monkeypatch):
    neuron_data = tmp_path / "neurons.pkl"
    chem_synapses = tmp_path / "chemical.pkl"
    elec_synapses = tmp_path / "gap.pkl"
    positions = tmp_path / "positions.pkl"

    write_pickle(
        neuron_data,
        pd.DataFrame(
            [
                ["AVAL", "sensory", "head", "chemosensory"],
                ["AVBL", "interneuron", "head", "integrative"],
            ]
        ),
    )
    write_pickle(chem_synapses, {})
    write_pickle(elec_synapses, {})
    write_pickle(positions, {"AVAL": np.array([0.0, 0.0, 0.0])})
    monkeypatch.setattr(loader, "neuronPositions", positions)

    nn = NervousSystem(Worm("test-worm"), network="TestNetwork")
    loader.build_nervous_system(
        nn,
        neuron_data=neuron_data,
        chem_synapses=chem_synapses,
        elec_synapses=elec_synapses,
        positions=positions,
    )

    assert set(nn.neurons["AVAL"].citations) == {"Skuhersky2022", "Yemini2020"}
    assert (
        nn.neurons["AVAL"].citations["Skuhersky2022"].doi
        == "10.1186/s12859-022-04738-3"
    )
    assert (
        nn.neurons["AVAL"].citations["Yemini2020"].doi == "10.1016/j.cell.2020.12.012"
    )
    assert nn.neurons["AVBL"].citations == {}


def test_build_nervous_system_rejects_conflicting_connection_modes(tmp_path):
    neuron_data = tmp_path / "neurons.pkl"
    chem_synapses = tmp_path / "chemical.pkl"
    elec_synapses = tmp_path / "gap.pkl"
    positions = tmp_path / "positions.pkl"

    write_pickle(
        neuron_data,
        pd.DataFrame([["AVAL", "sensory", "head", "chemosensory"]]),
    )
    write_pickle(chem_synapses, {"AVAL": {}})
    write_pickle(elec_synapses, {"AVAL": {}})
    write_pickle(positions, {"AVAL": np.array([0.0, 0.0, 0.0])})

    nn = NervousSystem(Worm("test-worm"), network="TestNetwork")
    with pytest.raises(
        AssertionError,
        match="Select at most one of gapjn_only or chem_only attributes to be True",
    ):
        loader.build_nervous_system(
            nn,
            neuron_data=neuron_data,
            chem_synapses=chem_synapses,
            elec_synapses=elec_synapses,
            positions=positions,
            chem_only=True,
            gapjn_only=True,
        )


def test_makeWorm_white_1986_loads_and_normalizes_edges(monkeypatch):
    neuron_types = pd.DataFrame(
        {
            " Neuron ": ["AVFL", "AVFR", "PVCL"],
            " Soma Position ": [1.0, 2.0, 3.0],
            " Soma Region ": ["head", "head", "tail"],
        }
    )
    edges = pd.DataFrame(
        [
            {"Neuron 1": "avfl", "Neuron 2": "PVCL", "Type": "S", "Nbr": 2},
            {"Neuron 1": "AVFR", "Neuron 2": "PVCL", "Type": "Sp", "Nbr": 1},
            {"Neuron 1": "PVCL", "Neuron 2": "avfl", "Type": "R", "Nbr": 9},
            {"Neuron 1": "avfl", "Neuron 2": "avfr", "Type": "EJ", "Nbr": 4},
            {"Neuron 1": "AVFL", "Neuron 2": "bodywall", "Type": "NMJ", "Nbr": 7},
        ]
    )

    def fake_read_excel(path, *args, **kwargs):
        if path.name == "NeuronType.xls":
            return neuron_types
        if path.name == "NeuronConnect.xls":
            return edges
        raise AssertionError(f"Unexpected path {path}")

    monkeypatch.setattr(loader.pd, "read_excel", fake_read_excel)

    worm = loader.makeWorm("white-test", import_parameters={"style": "white_1986"})
    nn = worm.networks["white_1986"]

    assert worm.citations["white_connectome"] == loader.citations["white_connectome"]
    assert nn.neurons["AVFL"].soma_position == 1.0
    assert nn.neurons["PVCL"].soma_region == "tail"
    assert find_connection(nn, "AVFL", "PVCL", "chemical-synapse").weight == 2
    assert find_connection(nn, "AVFR", "PVCL", "chemical-synapse").weight == 1
    assert find_connection(nn, "AVFL", "AVFR", "gap-junction").weight == 4
    assert find_connection(nn, "AVFR", "AVFL", "gap-junction").weight == 4
    assert (
        len(
            [
                c
                for c in nn.connections.values()
                if c.connection_type == "chemical-synapse"
            ]
        )
        == 2
    )


@pytest.mark.parametrize(
    ("dataset_ind", "expected_specimen", "expected_weights"),
    [
        (1, "107", {("A", "B"): 2, ("A", "C"): 1, ("B", "C"): 1}),
        (2, "148", {("X", "Y"): 2, ("X", "Z"): 1}),
    ],
)
def test_make_pristionchus_selects_requested_specimen(
    monkeypatch, dataset_ind, expected_specimen, expected_weights
):
    raw = pd.DataFrame(
        [
            [1, "A", "B", 10, "X", "Y"],
            [1, "A", "B", 10, "X", "Y"],
            [2, "A", "C", 11, "X", "Z"],
            [3, "B", "C", np.nan, np.nan, np.nan],
        ]
    )

    monkeypatch.setattr(loader.pd, "read_excel", lambda *args, **kwargs: raw)

    animal = loader.make_pristionchus("pristi", dataset_ind=dataset_ind)
    nn = animal.networks["pharynx"]

    assert animal.species == "Pristionchus pacificus"
    assert animal.specimen == expected_specimen
    assert (
        animal.citations["bumbarger_pharynx"] == loader.citations["bumbarger_pharynx"]
    )
    assert set(nn.neurons) == {name for edge in expected_weights for name in edge}
    for (pre, post), weight in expected_weights.items():
        assert find_connection(nn, pre, post, "chemical-synapse").weight == weight


def test_make_pristionchus_rejects_unknown_dataset():
    with pytest.raises(AssertionError, match="dataset_ind must be one of"):
        loader.make_pristionchus(dataset_ind=3)


def test_make_pristionchus_filters_repeat_header_rows(monkeypatch):
    """Repeat header rows mid-sheet must not leak into the cell list as
    pseudo-neurons. The Bumbarger 2013 mmc2.xlsx workbook has a
    multi-block layout where header literals reappear deeper in the file;
    without the guard they showed up as neurons named 'presynaptic' /
    'postsynaptic' in the loaded NervousSystem."""
    raw = pd.DataFrame(
        {
            "presynaptic": ["A", "presynaptic", "A", "B"],
            "postsynaptic": ["B", "postsynaptic", "C", "C"],
            "weight 107": [3, 0, 1, 1],
            "weight 148": [2, 0, 1, 0],
        }
    )
    monkeypatch.setattr(loader.pd, "read_excel", lambda *args, **kwargs: raw)

    animal = loader.make_pristionchus("pristi", dataset_ind=1)
    nn = animal.networks["pharynx"]
    assert "presynaptic" not in nn.neurons
    assert "postsynaptic" not in nn.neurons
    assert set(nn.neurons) == {"A", "B", "C"}


def _fake_flywire_read_csv(annotations_df=None):
    """Build a monkeypatch replacement for `pd.read_csv` covering every file
    the FlyWire loader touches. Pass an `annotations_df` to inject Schlegel
    2024 v783 annotations; pass `None` to simulate the file being absent.
    """

    def fake_read_csv(path, *args, **kwargs):
        if path.name == "names.csv":
            return pd.DataFrame(
                {
                    "name": ["NeuronB", "NeuronA"],
                    "group": ["TypeB", "TypeA"],
                    "root_id": [10, 2],
                }
            )
        if path.name == "coordinates.csv":
            return pd.DataFrame(
                {
                    "root_id": [2, 10],
                    "position": ["[1 2 3]", "[4 5 6]"],
                }
            )
        if path.name == "cell_stats.csv":
            return pd.DataFrame(
                {
                    "root_id": [2, 10],
                    "length_nm": [11, 22],
                    "area_nm": [33, 44],
                    "size_nm": [55, 66],
                }
            )
        if path.name == "Supplemental_file1_neuron_annotations.tsv":
            if annotations_df is None:
                raise FileNotFoundError(path)
            return annotations_df
        if path.name == "connections_no_threshold.csv":
            return pd.DataFrame(
                {
                    "pre_root_id": [2],
                    "post_root_id": [10],
                    "syn_count": [5],
                    "nt_type": ["GABA"],
                }
            )
        raise AssertionError(f"Unexpected csv path {path}")

    return fake_read_csv


def test_makeFly_fly_wire_loads_metadata_and_connections(monkeypatch):
    monkeypatch.setattr(loader.pd, "read_csv", _fake_flywire_read_csv())

    with pytest.warns(RuntimeWarning, match="annotations TSV not found"):
        fly = loader.makeFly("fly-wire", import_parameters={"style": "fly_wire"})
    nn = fly.networks["Neutral"]

    assert list(nn.neurons) == ["NeuronA", "NeuronB"]
    assert nn.neurons["NeuronA"].type == "TypeA"
    np.testing.assert_array_equal(nn.neurons["NeuronA"].position, np.array([1, 2, 3]))
    assert nn.neurons["NeuronA"].length == 11
    assert nn.neurons["NeuronB"].area == 44
    assert nn.neurons["NeuronB"].volume == 66
    assert fly.citations["fly_wire"] == loader.citations["fly_wire"]
    assert find_connection(nn, "NeuronA", "NeuronB", "chemical-synapse").weight == 5


def test_makeFly_fly_wire_attaches_schlegel_annotations(monkeypatch):
    """Schlegel 2024 v783 annotations should appear as per-neuron properties.

    Covers: NaN -> None handling, root_ids missing from names.csv are skipped,
    `cell_type` is exposed as `consolidated_type` (not `type`), and partial
    coverage (only one of two neurons annotated) is tolerated.
    """
    annot = pd.DataFrame(
        {
            # root_id 2 -> NeuronA (fully annotated); root_id 10 -> NeuronB
            # (only side/flow); root_id 999 has no matching neuron -> ignored.
            "root_id": [2, 10, 999],
            "flow": ["intrinsic", "afferent", "intrinsic"],
            "super_class": ["central", np.nan, "optic"],
            "cell_class": ["ALPN", np.nan, "T1"],
            "cell_sub_class": ["uPN", np.nan, np.nan],
            "supertype": ["VC5", np.nan, np.nan],
            "cell_type": ["DA1_lPN", np.nan, "T1_R"],
            "hemibrain_type": ["DA1_lPN", np.nan, np.nan],
            "ito_lee_hemilineage": ["ALl1", np.nan, np.nan],
            "hartenstein_hemilineage": ["BAlc", np.nan, np.nan],
            "side": ["right", "left", "right"],
            "nerve": [np.nan, np.nan, np.nan],
            "top_nt": ["acetylcholine", np.nan, "histamine"],
            "top_nt_conf": [0.91, np.nan, 0.77],
            "known_nt": ["acetylcholine", np.nan, np.nan],
            "vfb_id": ["VFB_00100000", np.nan, np.nan],
            "fbbt_id": ["FBbt_00067082", np.nan, np.nan],
            "dimorphism": [np.nan, np.nan, np.nan],
        }
    )
    monkeypatch.setattr(loader.pd, "read_csv", _fake_flywire_read_csv(annot))

    fly = loader.makeFly("fly-wire", import_parameters={"style": "fly_wire"})
    nn = fly.networks["Neutral"]
    a = nn.neurons["NeuronA"]
    b = nn.neurons["NeuronB"]

    # Existing `type` from names.csv preserved; Schlegel's verified cell type
    # is surfaced separately as `consolidated_type`.
    assert a.type == "TypeA"
    assert a.consolidated_type == "DA1_lPN"
    assert a.flow == "intrinsic"
    assert a.super_class == "central"
    assert a.cell_class == "ALPN"
    assert a.cell_sub_class == "uPN"
    assert a.ito_lee_hemilineage == "ALl1"
    assert a.hartenstein_hemilineage == "BAlc"
    assert a.side == "right"
    assert a.top_nt == "acetylcholine"
    assert a.top_nt_conf == pytest.approx(0.91)
    assert a.known_nt == "acetylcholine"
    assert a.vfb_id == "VFB_00100000"
    assert a.fbbt_id == "FBbt_00067082"

    # NeuronB only has side + flow + top_nt populated upstream; the rest must
    # land as None (not NaN) so downstream code can rely on `is None` checks.
    assert b.side == "left"
    assert b.flow == "afferent"
    assert b.consolidated_type is None
    assert b.cell_class is None
    assert b.ito_lee_hemilineage is None
    assert b.known_nt is None
    # Columns that are NaN across every annotated row still attach (as None)
    # because the property is registered when at least one neuron has data.
    assert a.nerve is None and b.nerve is None
    assert a.dimorphism is None and b.dimorphism is None


def test_makeFly_winding_2023_marks_unannotated_neurons(monkeypatch):
    def fake_read_csv(path, *args, **kwargs):
        if path.name == "annotations.csv":
            return pd.DataFrame(
                {
                    "left_id": ["L1", "no pair"],
                    "right_id": ["R1", "R2"],
                    "celltype": ["sensory", "motor"],
                }
            )
        if path.name == "all-all_connectivity_matrix.csv":
            return pd.DataFrame(
                [[0, 2, 0], [0, 0, 1], [0, 0, 0]],
                index=["L1", "R1", "X"],
                columns=["L1", "R1", "X"],
            )
        raise AssertionError(f"Unexpected csv path {path}")

    monkeypatch.setattr(loader.pd, "read_csv", fake_read_csv)

    fly = loader.makeFly("winding", import_parameters={"style": "Winding_2023"})
    nn = fly.networks["Neutral"]

    assert fly.stage == "Larva Instar-1"
    assert fly.citations["winding_connectome"] == loader.citations["winding_connectome"]
    assert nn.neurons["L1"].neuron_type == "sensory"
    assert nn.neurons["X"].neuron_type == "unannotated"
    assert find_connection(nn, "L1", "R1", "chemical-synapse").weight == 2.0
    assert find_connection(nn, "R1", "X", "chemical-synapse").weight == 1.0


def test_make_platynereis_builds_fine_and_grouped_networks(monkeypatch):
    def fake_read_csv(path, *args, **kwargs):
        if path.name == "neuronal_celltypes_table.csv":
            return pd.DataFrame(
                {
                    "name of cell type": ["ctA", "eff1"],
                    "soma position": ["head", "body"],
                    "region": ["brain", "trunk"],
                    "transmitter phenotype": ["GABA", None],
                    "number of cells": [2, 1],
                    "Sensory/inter/motor neuron": ["sensory", "effector"],
                }
            )
        if path.name == "elife-97964-fig1-data1.txt":
            return pd.DataFrame({"neuron": ["orphan_1"], "neuron_type": ["SN"]})
        if path.name == "elife-97964-fig3-data1.txt":
            return pd.DataFrame({"celltype": ["ctA"], "SN_MN_IN": ["Sensory neuron"]})
        if path.name == "elife-97964-fig3-data2-v1.txt":
            return pd.DataFrame({"celltype": ["eff1"], "ANNOT": ["muscle"]})
        if path.name == "elife-97964-fig3-figsupp2-data1.txt":
            return pd.DataFrame(
                {
                    "ctA": [1, 1, 0, 1, 0],
                    "eff1": [0, 0, 1, 0, 1],
                },
                index=[
                    "segment_0",
                    "glutamatergic",
                    "mushroom body",
                    "descending",
                    "dense cored vesicles",
                ],
            )
        if path.name == "full_connectome_adjacency_matrix.csv":
            return pd.DataFrame(
                [[0, 3, 0], [1, 0, 0], [0, 0, 0]],
                index=["ctA_1", "orphan_1", "fragment"],
                columns=["ctA_1", "orphan_1", "fragment"],
            )
        if path.name == "all_celltypes_synapse_matrix.csv":
            return pd.DataFrame(
                [[0, 2], [0, 0]],
                index=["ctA", "eff1"],
                columns=["ctA", "eff1"],
            )
        raise AssertionError(f"Unexpected csv path {path}")

    monkeypatch.setattr(loader.pd, "read_csv", fake_read_csv)

    animal = loader.make_platynereis("platy")
    fine = animal.networks["neurons"]
    grouped = animal.networks["celltypes"]

    assert (
        animal.citations["veraszto_connectome"]
        == loader.citations["veraszto_connectome"]
    )
    assert "fragment" not in fine.neurons
    assert fine.neurons["ctA_1"].type == "ctA"
    assert fine.neurons["ctA_1"].category == "sensory"
    assert fine.neurons["ctA_1"].body_segments == ["segment_0"]
    assert fine.neurons["ctA_1"].nt_tags == ["glutamatergic"]
    assert fine.neurons["ctA_1"].projection_tags == ["descending"]
    assert fine.neurons["orphan_1"].type is None
    assert fine.neurons["orphan_1"].category == "sensory"
    assert grouped.neurons["eff1"].is_neuron is False
    assert grouped.neurons["eff1"].cell_kind == "muscle"
    assert find_connection(grouped, "ctA", "eff1", "chemical-synapse").weight == 2.0


def test_load_contactome_mirrors_weights_and_reports_skips(tmp_path):
    workbook = tmp_path / "contactome.xlsx"
    adult = pd.DataFrame(
        [[0, 5, 2], [np.nan, 0, 0], [1, 0, 0]],
        index=["A", "B", "C"],
        columns=["A", "B", "C"],
    )
    l4 = pd.DataFrame([[0]], index=["A"], columns=["A"])
    with pd.ExcelWriter(workbook) as writer:
        adult.to_excel(writer, sheet_name="adult nerve ring neighbors")
        l4.to_excel(writer, sheet_name="L4 nerve ring neighbors")

    nn = NervousSystem(Worm("contactome"), network="Neutral")
    nn.create_neurons(["A", "B"])

    added, skipped = loader.load_contactome(nn, stage="adult", matrix_path=workbook)

    assert added == 2
    assert skipped == 1
    assert nn.weight_units == "nm^2"
    assert (
        nn.worm.citations["brittin_contactome"]
        == loader.citations["brittin_contactome"]
    )
    assert find_connection(nn, "A", "B", "contact").weight == 5.0
    assert find_connection(nn, "B", "A", "contact").weight == 5.0


def test_load_lineage_reads_requested_sheet(monkeypatch):
    calls = []
    lineage_df = pd.DataFrame({"cell": ["AB"], "meaning": ["founder"]})

    def fake_read_excel(path, sheet_name=None, engine=None):
        calls.append((path, sheet_name, engine))
        return lineage_df

    monkeypatch.setattr(loader.pd, "read_excel", fake_read_excel)

    result = loader.load_lineage(None, sex="Male")

    assert result.equals(lineage_df)
    assert calls == [(loader.lineage, "Male", "openpyxl")]


def test_loadNeuropeptides_builds_requested_worm_networks(monkeypatch):
    model = pd.DataFrame([[0, 1], [2, 0], [0, 0], [3, 0]])
    neuron_ids = pd.DataFrame({"Neuron": ["A", "B"]}, index=[1, 2])
    neuropep_rec = pd.DataFrame({"pair_names_NPP": ["NP1", "NP2"]})
    calls = []

    def fake_read_csv(path, *args, **kwargs):
        if path.name == "NPP_GPCR_networks_long_range_model_2.csv":
            return model
        if path.name == "26012022_num_neuronID.txt":
            return neuron_ids
        if path.name == "91-NPPGPCR networks":
            return neuropep_rec
        raise AssertionError(f"Unexpected csv path {path}")

    def fake_build_network(self, neuron_data, adj, label):
        calls.append(
            (
                self.name,
                neuron_data,
                label,
                adj["A"]["B"]["weight"],
                adj["B"]["A"]["weight"],
            )
        )

    monkeypatch.setattr(loader.pd, "read_csv", fake_read_csv)
    monkeypatch.setattr(NervousSystem, "build_network", fake_build_network)

    worm = Worm("np-worm")
    loader.loadNeuropeptides(worm, neuropeps=["NP1"])

    assert calls == [("NP1", loader.cell_list, "NP1", 1, 2)]
    assert "NP1" in worm.networks
    assert (
        worm.citations["neuropeptide_atlas"] == loader.citations["neuropeptide_atlas"]
    )


def test_loadNeuropeptides_adds_requested_connections_to_network(monkeypatch):
    model = pd.DataFrame([[0, 1], [2, 0], [0, 0], [3, 0]])
    neuron_ids = pd.DataFrame({"Neuron": ["A", "B"]}, index=[1, 2])
    neuropep_rec = pd.DataFrame({"pair_names_NPP": ["NP1", "NP2"]})

    def fake_read_csv(path, *args, **kwargs):
        if path.name == "NPP_GPCR_networks_long_range_model_2.csv":
            return model
        if path.name == "26012022_num_neuronID.txt":
            return neuron_ids
        if path.name == "91-NPPGPCR networks":
            return neuropep_rec
        raise AssertionError(f"Unexpected csv path {path}")

    monkeypatch.setattr(loader.pd, "read_csv", fake_read_csv)

    nn = NervousSystem(Worm("np-network"), network="Neutral")
    nn.create_neurons(["A", "B"])

    loader.loadNeuropeptides(nn, neuropeps=["NP1"])

    assert find_connection(nn, "A", "B", "NP1").weight == 1
    assert find_connection(nn, "B", "A", "NP1").weight == 2
    assert not any(conn.connection_type == "NP2" for conn in nn.connections.values())
    assert (
        nn.worm.citations["neuropeptide_atlas"]
        == loader.citations["neuropeptide_atlas"]
    )


def test_getNeuropeptideList_reads_pairs(monkeypatch):
    monkeypatch.setattr(
        loader.pd,
        "read_csv",
        lambda *args, **kwargs: pd.DataFrame({"pair_names_NPP": ["NP1", "NP2"]}),
    )

    assert loader.getNeuropeptideList() == ["NP1", "NP2"]


def test_loadTranscripts_maps_grouped_and_special_case_neurons(monkeypatch):
    transcript_table = pd.DataFrame(
        {
            "Unnamed: 0": [0, 1],
            "Wormbase_ID": ["WBGene1", "WBGene2"],
            "AVA": [1, 0],
            "AWC_OFF": [0, 1],
            "AWC_ON": [1, 0],
            "VD_DD": [1, 1],
            "RME_LR": [0, 1],
            "RME_DV": [1, 0],
            "DA9": [0, 1],
            "VC_4_5": [1, 0],
        },
        index=["gene0", "gene1"],
    )

    monkeypatch.setattr(
        loader.pd, "read_csv", lambda *args, **kwargs: transcript_table.copy()
    )

    nn = NervousSystem(Worm("transcripts"), network="Neutral")
    nn.create_neurons(
        ["AVAL", "AVAR", "AWCL", "AWCR", "VD1", "RMEL", "RMED", "DA09", "VC04"]
    )

    loader.loadTranscripts(nn, threshold=1)

    assert bool(nn.neurons["AVAL"].transcript["gene0"]) is True
    assert bool(nn.neurons["AWCL"].transcript["gene1"]) is True
    assert bool(nn.neurons["AWCR"].transcript["gene0"]) is True
    assert bool(nn.neurons["VD1"].transcript["gene0"]) is True
    assert bool(nn.neurons["RMEL"].transcript["gene1"]) is True
    assert bool(nn.neurons["RMED"].transcript["gene0"]) is True
    assert bool(nn.neurons["DA09"].transcript["gene1"]) is True
    assert bool(nn.neurons["VC04"].transcript["gene0"]) is True
    assert nn.worm.citations["cengen"] == loader.citations["cengen"]


def test_get_enriched_neurons_filters_excluded_targets():
    nn = NervousSystem(Worm("enriched"), network="Neutral")
    nn.create_neurons(["A", "B", "C"])
    nn.neurons["A"].transcript = pd.Series([0, 0, 0, 0, 1])
    nn.neurons["B"].transcript = pd.Series([0, 0, 0, 0, 0])
    nn.neurons["C"].transcript = pd.Series([0, 0, 0, 0, 1])

    enriched = loader.get_enriched_neurons(
        nn,
        ["A", "B", "C"],
        excluded_neurons=["C"],
        threshold=4,
    )

    assert enriched == ["A"]


def test_loadGapJunctions_uses_transcripts_to_annotate_pairs(monkeypatch):
    nn = NervousSystem(Worm("gap"), network="Neutral")
    nn.create_neurons(["A", "B"])
    nn.setup_gap_junctions({"A": {"B": {"weight": 1}}})

    def fake_load_transcripts(network, threshold):
        network.neurons["A"].set_property(
            "transcript",
            pd.Series([True, True, False], index=["inx1", "unc-7", "foo"]),
        )
        network.neurons["B"].set_property(
            "transcript",
            pd.Series([True, False, True], index=["inx1", "unc-7", "bar"]),
        )

    monkeypatch.setattr(loader, "loadTranscripts", fake_load_transcripts)

    loader.loadGapJunctions(nn)

    conn = find_connection(nn, "A", "B", "gap-junction")
    assert conn.putative_gapjn_subunits == {("inx1", "inx1"), ("unc-7", "inx1")}
    assert nn.worm.citations["cengen"] == loader.citations["cengen"]


def test_loadSynapticWeights_updates_known_and_missing_edges(monkeypatch):
    monkeypatch.setattr(
        loader.pd,
        "read_excel",
        lambda *args, **kwargs: pd.DataFrame([[7.5]], index=["A"], columns=["B"]),
    )

    nn = NervousSystem(Worm("weights"), network="Neutral")
    nn.create_neurons(["A", "B", "C"])
    nn.setup_connections(
        {"pre": "A", "post": "B", "weight": 1}, "chemical-synapse", input_type="edge"
    )
    nn.setup_connections(
        {"pre": "C", "post": "A", "weight": 2}, "chemical-synapse", input_type="edge"
    )

    # The new RuntimeWarning surfaces partial atlas coverage to the caller;
    # the test must allow it (the warning is the contract).
    with pytest.warns(RuntimeWarning, match="weight_inferred"):
        wt_matrix = loader.loadSynapticWeights(nn)

    assert wt_matrix.loc["B", "A"] == 7.5
    # Atlas-covered edge: weight updated to the predicted value, flagged inferred.
    ab = find_connection(nn, "A", "B", "chemical-synapse")
    assert ab.weight == 7.5
    assert ab.weight_inferred is True
    # Missing-from-atlas edge: structural weight preserved (was 2), flagged
    # not-inferred. Critically NOT NaN — NaN here used to poison the simulator's
    # weight matrix and produce silently-wrong rate trajectories.
    ca = find_connection(nn, "C", "A", "chemical-synapse")
    assert ca.weight == 2
    assert ca.weight_inferred is False
    assert nn.worm.citations["sig_prop_atlas"] == loader.citations["sig_prop_atlas"]


def test_load_atanas_populates_trials_and_behavior_with_existing_network(
    tmp_path, monkeypatch
):
    control_dir = tmp_path / "Control"
    heat_dir = tmp_path / "Heat"
    control_dir.mkdir()
    heat_dir.mkdir()

    payload = {
        "trace_array": {"aval": [1.0, 2.0], "AVBL": [0.5, 0.25]},
        "avg_timestep": 0.5,
        "velocity": [0.1, 0.2],
        "pumping": [1.0, 2.0],
    }
    (control_dir / "recording.json").write_text(json.dumps(payload))

    monkeypatch.setattr(
        loader,
        "atanas_whole_brain",
        {"Control": control_dir, "Heat": heat_dir},
    )

    nn = NervousSystem(Worm("atanas"), network="Neutral")
    nn.create_neurons(["AVAL", "AVBL", "AVCL"])

    result = loader.load_atanas(condition="Control", max_files=1, network=nn)

    assert result["network"] is nn
    assert result["neurons_loaded"] == 2
    assert result["num_timepoints"] == 2
    assert result["condition"] == "Control"
    assert len(result["sessions"]) == 1

    session = result["sessions"][0]
    assert (
        session.context.experimental.experimental_conditions["source_file"]
        == "recording.json"
    )
    assert session.behavior is not None
    assert np.allclose(session.behavior.timestamps, np.array([0.0, 30.0]))
    assert np.allclose(session.behavior.get_variable("velocity"), np.array([0.1, 0.2]))

    aval_trial = nn.neurons["AVAL"].trial[0]
    assert np.array_equal(aval_trial.recording, np.array([1.0, 2.0]))
    assert aval_trial.behavior is session.behavior
    assert aval_trial.metadata["source_file"] == "recording.json"
    assert set(nn.neurons["AVAL"].citations) == {"atanas2023"}
    assert set(nn.neurons["AVBL"].citations) == {"atanas2023"}
    assert (
        nn.neurons["AVAL"].citations["atanas2023"].doi == "10.1016/j.cell.2023.07.035"
    )
    assert nn.neurons["AVCL"].citations == {}


def test_loadNeurotransmitters_populates_edges_and_preserves_gap_junctions(
    monkeypatch, tmp_path
):
    worm = Worm("test-worm")
    nn = NervousSystem(worm, network="Neutral")
    nn.create_neurons(["AVAL", "AVBL", "AVCL"])
    nn.neurons["AVBL"]._postSynapse = {"legacy": "Legacy"}

    nn.setup_connections(
        {"pre": "AVAL", "post": "AVBL", "weight": 1},
        connection_type="chemical-synapse",
        input_type="edge",
    )
    nn.setup_connections(
        {"pre": "AVAL", "post": "AVCL", "weight": 1},
        connection_type="chemical-synapse",
        input_type="edge",
    )
    nn.setup_gap_junctions({"AVBL": {"AVCL": {"weight": 1}}})

    # Stub the CeNGEN CSV under the new canonical layout (hermaphrodite/L4/)
    # so the test is independent of whether the data is staged on disk in CI.
    # require_dataset_file checks Path.exists, so we write a real file instead
    # of monkey-patching the path resolver.
    cengen_dir = tmp_path / "CENGEN" / "hermaphrodite" / "L4"
    cengen_dir.mkdir(parents=True)
    stub_paths = []
    for fname in (
        "liberal_threshold1.csv",
        "medium_threshold2.csv",
        "conservative_threshold3.csv",
        "stringent_threshold4.csv",
    ):
        p = cengen_dir / fname
        p.write_text("gene_name,AVB,AVC\nacr-1,True,False\n")
        stub_paths.append(p)
    monkeypatch.setattr(loader, "_cengen_threshold_paths", lambda **_kw: stub_paths)

    tables = iter(
        [
            pd.DataFrame({"gene_name": ["acr-1"], "AVB": [True], "AVC": [False]}),
            pd.DataFrame({"gene": ["acr-1"], "ligand": ["ACh"]}),
        ]
    )
    monkeypatch.setattr(loader.pd, "read_excel", lambda *args, **kwargs: next(tables))
    monkeypatch.setattr(
        loader,
        "getLigands",
        lambda neuron, sex="Hermaphrodite", ligtable=None: ["Acetylcholine", "GABA"]
        if neuron == "AVAL"
        else [],
    )
    monkeypatch.setattr(
        loader,
        "getLigandsAndReceptors",
        lambda npr, ligmap, col: {"acr-1": "Acetylcholine"} if col == "AVB" else {},
    )

    loader.loadNeurotransmitters(nn, sex="Male")

    assert nn.neurons["AVAL"]._preSynapse == ["Acetylcholine", "GABA"]
    assert nn.neurons["AVBL"]._postSynapse == {
        "legacy": "Legacy",
        "acr-1": "Acetylcholine",
    }
    assert nn.neurons["AVCL"]._postSynapse == {}

    matched_conn = find_connection(nn, "AVAL", "AVBL", "chemical-synapse")
    fallback_conn = find_connection(nn, "AVAL", "AVCL", "chemical-synapse")
    gap_conn = find_connection(nn, "AVBL", "AVCL", "gap-junction")

    assert matched_conn.putative_neurotrasmitter_receptors == [
        ("Acetylcholine", "acr-1")
    ]
    assert matched_conn.neurotransmitters == ["Acetylcholine"]
    assert fallback_conn.putative_neurotrasmitter_receptors == []
    assert fallback_conn.neurotransmitters == []
    assert not hasattr(gap_conn, "neurotransmitters")
    assert (
        nn.worm.citations["neurotransmitter_atlas"]
        == loader.citations["neurotransmitter_atlas"]
    )
    assert (
        nn.worm.citations["altun_neurotransmitters_receptors"]
        == loader.citations["altun_neurotransmitters_receptors"]
    )
    assert nn.worm.citations["cengen"] == loader.citations["cengen"]


def test_foldByNeuronType_groups_suffix_families_and_forwards_options():
    network = FoldSpyNetwork(["AVAL", "AVAR", "VB01", "VB02", "MC01", "MC02", "RID"])

    result = graphtools.foldByNeuronType(
        network,
        exceptions=["RID"],
        self_loops=False,
        data="collect",
    )

    assert result["fold_by"] == {
        "AVA": ["AVAL", "AVAR"],
        "VB": ["VB01", "VB02"],
        "MC": ["MC01", "MC02"],
        "RID": ["RID"],
    }
    assert result["exceptions"] == ["RID"]
    assert result["self_loops"] is False
    assert result["data"] == "collect"


def test_foldLeftRight_builds_expected_pairs_and_skips_exceptions():
    network = FoldSpyNetwork(["AVAL", "AVAR", "AVL", "PVCL", "PVCR", "BAGL", "BAGR"])

    result = graphtools.foldLeftRight(network, exceptions=["BAGL"])

    assert result["fold_by"] == {
        "AVA": ["AVAL", "AVAR"],
        "PVC": ["PVCL", "PVCR"],
    }


def test_foldDorsoVentral_handles_simple_and_lateral_suffix_pairs():
    network = FoldSpyNetwork(["SMDD", "SMDV", "RMEDL", "RMEVL", "RID"])

    result = graphtools.foldDorsoVentral(network)

    assert result["fold_by"] == {
        "SMD": ["SMDD", "SMDV"],
        "RMEL": ["RMEDL", "RMEVL"],
    }


def test_make_hypermotifs_contracts_requested_nodes():
    motif = nx.DiGraph([(1, 2), (2, 3)])

    hypermotif = graphtools.make_hypermotifs(motif, length=3, join_at=[(3, 1)])

    assert sorted(hypermotif.nodes(), key=str) == [
        "1.1",
        "1.2",
        "1.3-2.1",
        "2.2",
        "2.3-3.1",
        "3.2",
        "3.3",
    ]
    assert sorted(
        hypermotif.edges(), key=lambda edge: (str(edge[0]), str(edge[1]))
    ) == [
        ("1.1", "1.2"),
        ("1.2", "1.3-2.1"),
        ("1.3-2.1", "2.2"),
        ("2.2", "2.3-3.1"),
        ("2.3-3.1", "3.2"),
        ("3.2", "3.3"),
    ]


def test_make_hypermotifs_rejects_non_integer_node_names():
    bad_motif = nx.DiGraph([("a", "b")])

    with pytest.raises(AssertionError, match="All nodes must have integer node names"):
        graphtools.make_hypermotifs(bad_motif, length=2, join_at=[])


def test_hierarchical_alignment_scores_feedforward_against_feedback():
    conns = [
        (SimpleNamespace(type="sensory"), SimpleNamespace(type="interneuron")),
        (SimpleNamespace(type="interneuron"), SimpleNamespace(type="motorneuron")),
        (SimpleNamespace(type="motorneuron"), SimpleNamespace(type="sensory")),
        (SimpleNamespace(type="other"), SimpleNamespace(type="motorneuron")),
    ]

    assert graphtools.hierarchical_alignment(conns) == pytest.approx(1 / 3)
