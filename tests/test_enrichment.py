import math
from collections.abc import Mapping, Sequence

import pytest

from cedne.core.connection import Connection, ConnectionGroup
from cedne.core.network import NervousSystem
from cedne.core.neuron import Neuron, NeuronGroup
from cedne.utils.enrichment import group_attribute_enrichment


def assert_json_safe(value):
    if isinstance(value, float):
        assert math.isfinite(value)
    elif isinstance(value, Mapping):
        for child in value.values():
            assert_json_safe(child)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for child in value:
            assert_json_safe(child)


@pytest.fixture()
def neuron_enrichment_network():
    net = NervousSystem()
    neurons = []
    for idx in range(10):
        neuron_type = "sensory" if idx < 5 else "motor"
        neuron = Neuron(f"N{idx}", net, type=neuron_type, score=float(idx))
        neuron.loadings = {"PC1": float(idx)}
        neurons.append(neuron)
    group = NeuronGroup(net, neurons[:4], group_name="sensory_group")
    comparison = NeuronGroup(net, neurons[6:], group_name="high_score_group")
    return net, neurons, group, comparison


def test_categorical_enrichment_against_network(neuron_enrichment_network):
    _, _, group, _ = neuron_enrichment_network

    result = group_attribute_enrichment(group, "type")

    assert result["group"] == "sensory_group"
    assert result["element"] == "node"
    assert result["attribute_type"] == "categorical"
    sensory = next(row for row in result["results"] if row["value"] == "sensory")
    assert sensory["observed_count"] == 4
    assert sensory["reference_count"] == 5
    assert sensory["direction"] == "enriched"
    assert sensory["p_enrichment"] < 0.05
    assert "q_value" in sensory


def test_numeric_enrichment_uses_deterministic_size_matched_null(
    neuron_enrichment_network,
):
    _, _, group, _ = neuron_enrichment_network

    result = group_attribute_enrichment(
        group,
        "score",
        null_model="size_matched",
        n_resamples=200,
        random_state=7,
    )
    repeat = group_attribute_enrichment(
        group,
        "score",
        null_model="size_matched",
        n_resamples=200,
        random_state=7,
    )

    row = result["results"][0]
    assert result == repeat
    assert row["statistic"] == "mean"
    assert row["observed"] == pytest.approx(1.5)
    assert row["reference_mean"] == pytest.approx(4.5)
    assert row["p_depletion"] < 0.1
    assert len(row["null_ci95"]) == 2


def test_explicit_group_numeric_comparison(neuron_enrichment_network):
    _, _, group, comparison = neuron_enrichment_network

    result = group_attribute_enrichment(group, "loadings.PC1", reference=comparison)

    row = result["results"][0]
    assert result["reference"]["name"] == "high_score_group"
    assert result["attribute_type"] == "numeric"
    assert row["observed"] == pytest.approx(1.5)
    assert row["reference_mean"] == pytest.approx(7.5)
    assert row["p_depletion"] < 0.05
    assert row["null_ci95"] == [None, None]


def test_connection_group_attribute_enrichment():
    net = NervousSystem()
    neurons = [Neuron(f"N{idx}", net) for idx in range(5)]
    chem = [
        Connection(
            neurons[0], neurons[1], uid="c01", connection_type="chemical-synapse"
        ),
        Connection(
            neurons[1], neurons[2], uid="c12", connection_type="chemical-synapse"
        ),
        Connection(
            neurons[2], neurons[3], uid="c23", connection_type="chemical-synapse"
        ),
    ]
    gap = [
        Connection(neurons[3], neurons[4], uid="g34", connection_type="gap-junction"),
        Connection(neurons[4], neurons[0], uid="g40", connection_type="gap-junction"),
    ]
    group = ConnectionGroup(net, chem, group_name="chemical_subset")

    result = group_attribute_enrichment(group, "connection_type")

    assert result["element"] == "edge"
    chemical = next(
        row for row in result["results"] if row["value"] == "chemical-synapse"
    )
    gap_row = next(row for row in result["results"] if row["value"] == "gap-junction")
    assert chemical["direction"] == "enriched"
    assert gap_row["direction"] == "depleted"
    assert len(gap) == 2


def test_missing_attribute_raises(neuron_enrichment_network):
    _, _, group, _ = neuron_enrichment_network

    result = group_attribute_enrichment(group, "missing_attr")

    assert result["results"] == []
    assert result["observed"]["valid_size"] == 0
    assert result["observed"]["missing"] == 4
    assert result["missingness"]["direction"] == "unchanged"
    assert "missingness block" in result["warning"]


def test_no_infinite_odds_ratio_leaks_to_json(neuron_enrichment_network):
    _, _, group, _ = neuron_enrichment_network

    result = group_attribute_enrichment(group, "type")
    sensory = next(row for row in result["results"] if row["value"] == "sensory")
    assert sensory["odds_ratio"] is None or math.isfinite(sensory["odds_ratio"])


def test_missing_numeric_activity_is_reported_not_depleted():
    net = NervousSystem()
    neurons = [Neuron(f"N{idx}", net) for idx in range(6)]
    for idx, neuron in enumerate(neurons):
        if idx >= 3:
            neuron.mean_activity = float(idx)
    group = NeuronGroup(net, neurons[:3], group_name="unrecorded_group")

    result = group_attribute_enrichment(group, "mean_activity", mode="numeric")

    assert result["results"] == []
    assert result["observed"]["valid_size"] == 0
    assert result["observed"]["missing"] == 3
    assert result["reference"]["valid_size"] == 3
    assert result["missingness"]["direction"] == "missing_enriched"


def test_set_membership_empty_lists_are_absence_not_missing():
    net = NervousSystem()
    neurons = [Neuron(f"N{idx}", net) for idx in range(6)]
    conns = [
        Connection(neurons[0], neurons[1], uid="c01", neuropeptides=["FLP-18"]),
        Connection(neurons[1], neurons[2], uid="c12", neuropeptides=[]),
        Connection(neurons[2], neurons[3], uid="c23", neuropeptides=[]),
        Connection(neurons[3], neurons[4], uid="c34", neuropeptides=["FLP-18"]),
        Connection(neurons[4], neurons[5], uid="c45", neuropeptides=["NLP-1"]),
    ]
    group = ConnectionGroup(net, conns[1:3], group_name="no_flp18_group")

    result = group_attribute_enrichment(
        group,
        "neuropeptides",
        mode="set_membership",
        value="FLP-18",
        missing_policy="empty_is_absent",
    )

    row = result["results"][0]
    assert result["mode"] == "set_membership"
    assert result["observed"]["missing"] == 0
    assert row["value"] == "FLP-18"
    assert row["observed_count"] == 0
    assert row["direction"] == "depleted"


def test_eligible_filter_limits_denominator():
    net = NervousSystem()
    neurons = [Neuron(f"N{idx}", net) for idx in range(5)]
    conns = [
        Connection(
            neurons[0],
            neurons[1],
            uid="np1",
            connection_type="neuropeptide-receptor",
            neuropeptides=["FLP-18"],
        ),
        Connection(
            neurons[1],
            neurons[2],
            uid="np2",
            connection_type="neuropeptide-receptor",
            neuropeptides=[],
        ),
        Connection(
            neurons[2],
            neurons[3],
            uid="chem1",
            connection_type="chemical-synapse",
            neuropeptides=[],
        ),
    ]
    group = ConnectionGroup(net, conns, group_name="mixed_connections")

    result = group_attribute_enrichment(
        group,
        "neuropeptides",
        mode="set_membership",
        value="FLP-18",
        missing_policy="empty_is_absent",
        eligible_filter={"connection_type": "neuropeptide-receptor"},
    )

    assert result["observed"]["size"] == 3
    assert result["observed"]["eligible_size"] == 2
    assert result["reference"]["size"] == 2
    assert result["results"][0]["observed_count"] == 1


@pytest.fixture()
def stress_network():
    net = NervousSystem()
    neurons = []
    for idx in range(12):
        kwargs = {
            "type": "sensory" if idx < 4 else "motor" if idx < 8 else "interneuron",
            "score": float(idx),
            "is_active": idx % 2 == 0,
        }
        if idx != 11:
            kwargs["annotation"] = "known" if idx < 6 else "other"
        neuron = Neuron(f"S{idx}", net, **kwargs)
        if idx < 10:
            neuron.mean_activity = float(idx) / 10.0
        neurons.append(neuron)
    group = NeuronGroup(net, neurons[:4], group_name="stress_group")
    comparison = NeuronGroup(net, neurons[8:12], group_name="stress_comparison")
    return net, neurons, group, comparison


@pytest.fixture()
def connection_stress_network():
    net = NervousSystem()
    neurons = [Neuron(f"C{idx}", net) for idx in range(8)]
    conns = [
        Connection(
            neurons[0],
            neurons[1],
            uid="c0",
            connection_type="chemical-synapse",
            neurotransmitters=["GABA"],
            neuropeptides={"FLP-18": 1},
            curated=True,
        ),
        Connection(
            neurons[1],
            neurons[2],
            uid="c1",
            connection_type="chemical-synapse",
            neurotransmitters=["ACh"],
            neuropeptides={},
            curated=False,
        ),
        Connection(
            neurons[2],
            neurons[3],
            uid="c2",
            connection_type="chemical-synapse",
            neurotransmitters=[],
            curated=False,
        ),
        Connection(
            neurons[3],
            neurons[4],
            uid="c3",
            connection_type="neuropeptide-receptor",
            neuropeptides=["FLP-18"],
            curated=True,
        ),
        Connection(
            neurons[4],
            neurons[5],
            uid="c4",
            connection_type="neuropeptide-receptor",
            neuropeptides=[],
            curated=False,
        ),
        Connection(
            neurons[5],
            neurons[6],
            uid="c5",
            connection_type="gap-junction",
            curated=True,
        ),
    ]
    group = ConnectionGroup(net, conns[:3], group_name="connection_stress_group")
    comparison = ConnectionGroup(
        net, conns[3:], group_name="connection_stress_comparison"
    )
    return net, conns, group, comparison


@pytest.mark.parametrize("mode", ["auto", "numeric", "categorical", "binary"])
def test_supported_non_set_modes_return_json_safe_results(stress_network, mode):
    _, _, group, _ = stress_network
    attribute = {
        "auto": "score",
        "numeric": "score",
        "categorical": "type",
        "binary": "is_active",
    }[mode]

    result = group_attribute_enrichment(
        group, attribute, mode=mode, n_resamples=25, random_state=1
    )

    assert result["mode"] == ("numeric" if mode == "auto" else mode)
    assert result["results"]
    assert_json_safe(result)


@pytest.mark.parametrize(
    "null_model",
    [
        "network",
        "full_network",
        "full",
        "size_matched",
        "shuffled",
        "permutation",
        "sample",
    ],
)
def test_numeric_network_null_aliases_are_supported(stress_network, null_model):
    _, _, group, _ = stress_network

    result = group_attribute_enrichment(
        group,
        "score",
        null_model=null_model,
        n_resamples=20,
        random_state=2,
    )

    assert result["null_model"] == null_model
    assert result["reference"]["name"] == "network"
    assert result["results"][0]["observed_n"] == 4


@pytest.mark.parametrize("alternative", ["two-sided", "greater", "less"])
def test_alternative_controls_primary_p_value(stress_network, alternative):
    _, _, group, _ = stress_network

    result = group_attribute_enrichment(
        group,
        "type",
        alternative=alternative,
    )

    sensory = next(row for row in result["results"] if row["value"] == "sensory")
    expected = {
        "two-sided": min(
            1.0, 2.0 * min(sensory["p_enrichment"], sensory["p_depletion"])
        ),
        "greater": sensory["p_enrichment"],
        "less": sensory["p_depletion"],
    }[alternative]
    assert sensory["p_value"] == pytest.approx(expected)


def test_reference_group_can_be_object_or_name(stress_network):
    net, _, group, comparison = stress_network

    by_object = group_attribute_enrichment(group, "score", reference=comparison)
    by_name = group_attribute_enrichment(group, "score", reference="stress_comparison")

    assert by_object["reference"]["name"] == "stress_comparison"
    assert by_name["reference"]["name"] == "stress_comparison"
    assert (
        by_object["results"][0]["reference_mean"]
        == by_name["results"][0]["reference_mean"]
    )
    assert net.groups["stress_comparison"] is comparison


def test_complement_reference_excludes_observed_group(stress_network):
    _, _, group, _ = stress_network

    result = group_attribute_enrichment(group, "type", reference="complement")

    assert result["reference"]["name"] == "complement"
    assert result["reference"]["size"] == 8


@pytest.mark.parametrize(
    "missing_policy, expected_missing, expected_unknown",
    [
        ("exclude_and_report", 1, False),
        ("missing_is_unknown", 0, True),
    ],
)
def test_categorical_missing_policy_semantics(
    stress_network, missing_policy, expected_missing, expected_unknown
):
    _, neurons, _, _ = stress_network
    group = NeuronGroup(
        neurons[0].network,
        [neurons[10], neurons[11]],
        group_name=f"missing_{missing_policy}",
    )

    result = group_attribute_enrichment(
        group,
        "annotation",
        mode="categorical",
        missing_policy=missing_policy,
    )

    assert result["observed"]["missing"] == expected_missing
    has_unknown = any(row["value"] == "Unknown" for row in result["results"])
    assert has_unknown is expected_unknown


@pytest.mark.parametrize(
    "missing_policy, expected_missing, expected_valid",
    [
        # In set-membership mode, ["ACh"] is a valid known absence of "GABA".
        # The empty list is missing only when the caller chooses exclude_and_report.
        ("exclude_and_report", 1, 2),
        ("empty_is_absent", 0, 3),
        ("missing_is_absent", 0, 3),
    ],
)
def test_set_membership_missing_policy_semantics(
    connection_stress_network, missing_policy, expected_missing, expected_valid
):
    _, _, group, _ = connection_stress_network

    result = group_attribute_enrichment(
        group,
        "neurotransmitters",
        mode="set_membership",
        value="GABA",
        missing_policy=missing_policy,
    )

    row = result["results"][0]
    assert result["observed"]["missing"] == expected_missing
    assert result["observed"]["valid_size"] == expected_valid
    assert row["observed_count"] == 1


@pytest.mark.parametrize(
    "missing_policy, expected_missing, expected_valid",
    [
        ("exclude_and_report", 1, 2),
        ("empty_is_absent", 0, 3),
        ("missing_is_absent", 0, 3),
    ],
)
def test_binary_missing_policy_semantics(
    connection_stress_network, missing_policy, expected_missing, expected_valid
):
    _, conns, _, _ = connection_stress_network
    delattr(conns[2], "curated")
    group = ConnectionGroup(
        conns[0].network, conns[:3], group_name=f"binary_{missing_policy}"
    )

    result = group_attribute_enrichment(
        group,
        "curated",
        mode="binary",
        missing_policy=missing_policy,
    )

    row = result["results"][0]
    assert result["observed"]["missing"] == expected_missing
    assert result["observed"]["valid_size"] == expected_valid
    assert row["value"] is True


def test_set_membership_supports_dict_list_set_and_scalar_values(
    connection_stress_network,
):
    _, conns, _, _ = connection_stress_network
    conns[1].neuropeptides = {"FLP-18": 0}
    conns[2].neuropeptides = "FLP-18"
    group = ConnectionGroup(
        conns[0].network, conns[:3], group_name="mixed_value_shapes"
    )

    result = group_attribute_enrichment(
        group,
        "neuropeptides",
        mode="set_membership",
        value="FLP-18",
        missing_policy="empty_is_absent",
    )

    assert result["results"][0]["observed_count"] == 2


def test_eligible_filter_supports_expected_lists_and_actual_lists(
    connection_stress_network,
):
    _, conns, group, _ = connection_stress_network
    conns[0].tags = ["curated", "synaptic"]
    conns[1].tags = ["synaptic"]
    conns[2].tags = ["uncurated"]

    result = group_attribute_enrichment(
        group,
        "neurotransmitters",
        mode="set_membership",
        value="GABA",
        missing_policy="empty_is_absent",
        eligible_filter={
            "connection_type": ["chemical-synapse", "gap-junction"],
            "tags": "synaptic",
        },
    )

    assert result["observed"]["eligible_size"] == 2
    assert result["results"][0]["observed_count"] == 1


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"attribute": ""}, "attribute must be"),
        ({"attribute": "type", "mode": "unsupported"}, "mode must be"),
        ({"attribute": "type", "attribute_type": "bad"}, "attribute_type must be"),
        ({"attribute": "type", "missing_policy": "bad"}, "missing_policy must be"),
        ({"attribute": "type", "null_model": "bad"}, "null_model must be"),
        ({"attribute": "type", "alternative": "bad"}, "alternative must be"),
        ({"attribute": "type", "n_resamples": 0}, "n_resamples must be"),
        ({"attribute": "type", "mode": "set_membership"}, "value is required"),
    ],
)
def test_invalid_option_matrix_raises(stress_network, kwargs, message):
    _, _, group, _ = stress_network

    with pytest.raises(ValueError, match=message):
        group_attribute_enrichment(group, **kwargs)


@pytest.mark.parametrize(
    "missing_policy", ["empty_is_absent", "missing_is_absent", "missing_is_unknown"]
)
def test_numeric_rejects_absence_style_missing_policies(stress_network, missing_policy):
    _, _, group, _ = stress_network

    with pytest.raises(ValueError, match="Numeric enrichment requires"):
        group_attribute_enrichment(
            group, "score", mode="numeric", missing_policy=missing_policy
        )


def test_invalid_reference_cases_raise(stress_network):
    net, neurons, group, _ = stress_network
    other_net = NervousSystem()
    other_neuron = Neuron("OTHER", other_net, type="sensory")
    other_group = NeuronGroup(other_net, [other_neuron], group_name="other")
    edge = Connection(neurons[0], neurons[1], uid="bad_type_ref")
    connection_group = ConnectionGroup(net, [edge], group_name="edge_ref")

    with pytest.raises(ValueError, match="Reference group"):
        group_attribute_enrichment(group, "type", reference="missing")
    with pytest.raises(ValueError, match="same network"):
        group_attribute_enrichment(group, "type", reference=other_group)
    with pytest.raises(TypeError, match="same element type"):
        group_attribute_enrichment(group, "type", reference=connection_group)


def test_no_eligible_observed_or_reference_members_raise(connection_stress_network):
    _, _, group, comparison = connection_stress_network

    with pytest.raises(ValueError, match="No observed group members are eligible"):
        group_attribute_enrichment(
            group,
            "neuropeptides",
            mode="set_membership",
            value="FLP-18",
            eligible_filter={"connection_type": "not-real"},
        )

    with pytest.raises(ValueError, match="No reference members are eligible"):
        group_attribute_enrichment(
            group,
            "neuropeptides",
            mode="set_membership",
            value="FLP-18",
            reference=comparison,
            eligible_filter={"connection_type": "chemical-synapse"},
        )


def test_categorical_value_present_only_in_observed_group_is_reported(stress_network):
    _, neurons, group, comparison = stress_network
    for neuron in group.values():
        neuron.annotation = "observed_only"
    for neuron in comparison.values():
        neuron.annotation = "reference_only"

    result = group_attribute_enrichment(group, "annotation", reference=comparison)

    observed_only = next(
        row for row in result["results"] if row["value"] == "observed_only"
    )
    reference_only = next(
        row for row in result["results"] if row["value"] == "reference_only"
    )
    assert observed_only["reference_count"] == 0
    assert observed_only["direction"] == "enriched"
    assert reference_only["observed_count"] == 0
    assert reference_only["direction"] == "depleted"


def test_public_alias_matches_primary_function(stress_network):
    from cedne.utils.enrichment import test_group_attribute_enrichment

    _, _, group, _ = stress_network
    assert test_group_attribute_enrichment(group, "type") == group_attribute_enrichment(
        group, "type"
    )
