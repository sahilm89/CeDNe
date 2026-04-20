from __future__ import annotations

import numpy as np
import pytest

from cedne.simulator import StepInput, RateModel, LDSModel, CTRNNModel, DKBModel, CalciumObservation


class SimpleNode:
    def __init__(self, name: str):
        self.name = name


class SimpleGraph:
    def __init__(self):
        self._nodes = [SimpleNode("A"), SimpleNode("B")]
        self._edges = [(self._nodes[0], self._nodes[1], {"weight": 1.0})]

    @property
    def nodes(self):
        return self._nodes

    def edges(self, data=False):
        if data:
            return list(self._edges)
        return [(u, v) for u, v, _ in self._edges]


@pytest.fixture
def small_graph():
    import networkx as nx

    g = nx.DiGraph()
    a = SimpleNode("A")
    b = SimpleNode("B")
    g.add_node(a)
    g.add_node(b)
    g.add_edge(a, b, weight=1.0)
    return g, a, b


@pytest.fixture
def graph_with_static():
    import networkx as nx

    g = nx.DiGraph()
    a = SimpleNode("A")
    b = SimpleNode("B")
    s = SimpleNode("S")
    g.add_node(a)
    g.add_node(b)
    g.add_node(s)
    g.add_edge(a, b, weight=1.0)
    return g, a, b, s


def _build_model(model_kind, graph, a, b, *, static_neurons=None, time_points=None, inputs=None, input_neurons=None):
    if input_neurons is None:
        input_neurons = [a]
    if model_kind == "rate":
        return RateModel(
            graph,
            input_neurons=input_neurons,
            weights={(a, b): 1.0},
            gains={a: 1.0, b: 1.0},
            time_constants={a: 1.0, b: 1.0},
            baseline={a: 0.0, b: 0.0},
            static_neurons=static_neurons,
            time_points=time_points,
            inputs=inputs,
        )
    if model_kind == "ctrnn":
        return CTRNNModel(
            graph,
            input_neurons=input_neurons,
            weights={(a, b): 1.0},
            tau={a: 0.6, b: 0.8},
            gains={a: 1.0, b: 1.1},
            baseline={a: 0.0, b: 0.0},
            activation="tanh",
            static_neurons=static_neurons,
            time_points=time_points,
            inputs=inputs,
        )
    if model_kind == "lds":
        return LDSModel(
            graph,
            input_neurons=input_neurons,
            weights={(a, b): 1.0},
            baseline={a: 0.0, b: 0.0},
            input_weight={a: 1.0, b: 1.0},
            static_neurons=static_neurons,
            time_points=time_points,
            inputs=inputs,
        )
    if model_kind == "dkb":
        return DKBModel(
            graph,
            input_neurons=input_neurons,
            weights={(a, b): 0.9},
            damping={a: 1.2, b: 1.2},
            stiffness={a: 1.0, b: 1.0},
            baseline={a: 0.0, b: 0.0},
            input_weight={a: 1.0, b: 1.0},
            target={a: 0.0, b: 0.0},
            static_neurons=static_neurons,
            time_points=time_points,
            inputs=inputs,
        )
    raise ValueError(f"Unsupported model_kind: {model_kind}")


def test_step_input_is_active_only_within_window(small_graph):
    _g, a, _b = small_graph
    inp = StepInput([a], tstart=1.0, tend=3.0, value=2.5)

    assert inp.process_input(0.5) == 0
    assert inp.process_input(2.0) == pytest.approx(2.5)
    assert inp.process_input(3.0) == 0


def test_rate_model_requires_time_points(small_graph):
    g, a, b = small_graph
    model = RateModel(
        g,
        input_neurons=[a],
        weights={(a, b): 1.0},
        gains={a: 1.0, b: 1.0},
        time_constants={a: 1.0, b: 1.0},
        baseline={a: 0.0, b: 0.0},
    )

    with pytest.raises(ValueError, match="Time points must be set"):
        model.simulate()


def test_rate_model_step_input_drives_response(small_graph):
    g, a, b = small_graph
    time_points = np.linspace(0, 4, 41)
    model = RateModel(
        g,
        input_neurons=[a],
        weights={(a, b): 1.0},
        gains={a: 1.0, b: 1.0},
        time_constants={a: 1.0, b: 1.0},
        baseline={a: 0.0, b: 0.0},
        time_points=time_points,
        inputs=[StepInput([a], tstart=0.5, tend=2.5, value=1.0)],
    )

    rates = model.simulate()

    assert rates[model.neurons[a]][-1] >= 0.0
    assert np.max(rates[model.neurons[b]]) > 0.0


def test_ctrnn_model_requires_time_points(small_graph):
    g, a, b = small_graph
    model = CTRNNModel(
        g,
        input_neurons=[a],
        weights={(a, b): 1.0},
        tau={a: 1.0, b: 1.0},
        gains={a: 1.0, b: 1.0},
        baseline={a: 0.0, b: 0.0},
    )

    with pytest.raises(ValueError, match="Time points must be set"):
        model.simulate()


def test_lds_model_requires_time_points(small_graph):
    g, a, b = small_graph
    model = LDSModel(
        g,
        input_neurons=[a],
        weights={(a, b): 1.0},
        baseline={a: 0.0, b: 0.0},
        input_weight={a: 1.0, b: 1.0},
    )

    with pytest.raises(ValueError, match="Time points must be set"):
        model.simulate()


def test_lds_model_step_input_drives_recurrent_response(small_graph):
    g, a, b = small_graph
    time_points = np.linspace(0, 3, 61)
    model = LDSModel(
        g,
        input_neurons=[a],
        weights={(a, b): 1.0},
        baseline={a: 0.0, b: 0.0},
        input_weight={a: 1.0, b: 1.0},
        time_points=time_points,
        inputs=[StepInput([a], tstart=0.2, tend=1.8, value=1.0)],
    )

    states = model.simulate()

    assert np.max(states[model.neurons[a]]) > 0.0
    assert np.max(states[model.neurons[b]]) > 0.0


def test_lds_model_baseline_produces_linear_ramp(small_graph):
    g, a, b = small_graph
    time_points = np.linspace(0, 2, 41)
    model = LDSModel(
        g,
        input_neurons=[],
        weights={(a, b): 0.0},
        baseline={a: 1.0, b: 0.0},
        input_weight={a: 1.0, b: 1.0},
        time_points=time_points,
    )

    states = model.simulate()

    assert states[model.neurons[a]][10] > 0.0
    assert states[model.neurons[a]][20] > states[model.neurons[a]][10]


def test_ctrnn_model_step_input_drives_recurrent_response(small_graph):
    g, a, b = small_graph
    time_points = np.linspace(0, 4, 41)
    model = CTRNNModel(
        g,
        input_neurons=[a],
        weights={(a, b): 1.0},
        tau={a: 0.5, b: 0.75},
        gains={a: 1.0, b: 1.2},
        baseline={a: 0.0, b: 0.0},
        activation="tanh",
        time_points=time_points,
        inputs=[StepInput([a], tstart=0.5, tend=2.5, value=1.0)],
    )

    rates = model.simulate()

    assert np.max(rates[model.neurons[a]]) > 0.0
    assert np.max(rates[model.neurons[b]]) > 0.0


def test_ctrnn_tau_changes_response_speed(small_graph):
    g, a, b = small_graph
    time_points = np.linspace(0, 3, 31)
    common_kwargs = dict(
        graph=g,
        input_neurons=[a],
        weights={(a, b): 1.0},
        gains={a: 1.0, b: 1.0},
        baseline={a: 0.0, b: 0.0},
        activation="tanh",
        time_points=time_points,
        inputs=[StepInput([a], tstart=0.2, tend=2.0, value=1.0)],
    )

    fast = CTRNNModel(tau={a: 0.25, b: 0.25}, **common_kwargs)
    slow = CTRNNModel(tau={a: 2.0, b: 2.0}, **common_kwargs)

    fast_rates = fast.simulate()
    slow_rates = slow.simulate()

    assert fast_rates[fast.neurons[b]][10] > slow_rates[slow.neurons[b]][10]


def test_dkb_model_requires_time_points(small_graph):
    g, a, b = small_graph
    model = DKBModel(
        g,
        input_neurons=[a],
        weights={(a, b): 0.5},
        damping={a: 0.5, b: 0.5},
        stiffness={a: 1.0, b: 1.0},
        baseline={a: 0.0, b: 0.0},
    )

    with pytest.raises(ValueError, match="Time points must be set"):
        model.simulate()


def test_dkb_model_damped_state_converges_toward_target(small_graph):
    g, a, b = small_graph
    time_points = np.linspace(0, 6, 121)
    model = DKBModel(
        g,
        input_neurons=[],
        weights={(a, b): 0.0},
        damping={a: 2.5, b: 2.5},
        stiffness={a: 1.5, b: 1.5},
        baseline={a: 0.0, b: 0.0},
        target={a: 0.0, b: 0.0},
        time_points=time_points,
    )

    states = model.simulate(initial_states=np.array([1.0, 0.0], dtype=np.float32))

    assert abs(states[model.neurons[a]][-1]) < 0.2


def test_dkb_model_input_can_drive_recurrent_target(small_graph):
    g, a, b = small_graph
    time_points = np.linspace(0, 4, 81)
    model = DKBModel(
        g,
        input_neurons=[a],
        weights={(a, b): 1.2},
        damping={a: 1.4, b: 1.4},
        stiffness={a: 1.0, b: 1.0},
        baseline={a: 0.0, b: 0.0},
        input_weight={a: 1.0, b: 1.0},
        target={a: 0.0, b: 0.0},
        time_points=time_points,
        inputs=[StepInput([a], tstart=0.2, tend=2.0, value=1.0)],
    )

    states = model.simulate()

    assert np.max(states[model.neurons[a]]) > 0.0
    assert np.max(states[model.neurons[b]]) > 0.0
    assert model.last_velocities is not None


def test_dkb_higher_damping_reduces_peak_displacement(small_graph):
    g, a, b = small_graph
    time_points = np.linspace(0, 5, 101)
    common_kwargs = dict(
        graph=g,
        input_neurons=[a],
        weights={(a, b): 0.8},
        stiffness={a: 1.1, b: 1.1},
        baseline={a: 0.0, b: 0.0},
        input_weight={a: 1.0, b: 1.0},
        target={a: 0.0, b: 0.0},
        time_points=time_points,
        inputs=[StepInput([a], tstart=0.2, tend=1.5, value=1.0)],
    )

    low_damping = DKBModel(damping={a: 0.2, b: 0.2}, **common_kwargs)
    high_damping = DKBModel(damping={a: 2.5, b: 2.5}, **common_kwargs)

    low_states = low_damping.simulate()
    high_states = high_damping.simulate()

    assert np.max(low_states[low_damping.neurons[b]]) > np.max(high_states[high_damping.neurons[b]])


def test_dkb_higher_stiffness_returns_to_target_faster(small_graph):
    g, a, b = small_graph
    time_points = np.linspace(0, 5, 101)
    common_kwargs = dict(
        graph=g,
        input_neurons=[],
        weights={(a, b): 0.0},
        damping={a: 3.0, b: 3.0},
        baseline={a: 0.0, b: 0.0},
        target={a: 0.0, b: 0.0},
        time_points=time_points,
    )

    soft = DKBModel(stiffness={a: 0.4, b: 0.4}, **common_kwargs)
    stiff = DKBModel(stiffness={a: 2.0, b: 2.0}, **common_kwargs)

    soft_states = soft.simulate(initial_states=np.array([1.0, 0.0], dtype=np.float32))
    stiff_states = stiff.simulate(initial_states=np.array([1.0, 0.0], dtype=np.float32))

    assert abs(stiff_states[stiff.neurons[a]][20]) < abs(soft_states[soft.neurons[a]][20])


def test_calcium_observation_zero_activity_stays_flat():
    time_points = np.linspace(0, 2, 41)
    node = SimpleNode("A")
    observation = CalciumObservation(rise_tau=0.15, decay_tau=0.7)

    calcium = observation.transform({node: np.zeros_like(time_points)}, time_points)

    assert np.allclose(calcium[node], 0.0)


def test_calcium_observation_filters_step_like_activity():
    time_points = np.linspace(0, 4, 81)
    node = SimpleNode("A")
    latent = np.zeros_like(time_points)
    latent[time_points >= 1.0] = 1.0
    observation = CalciumObservation(rise_tau=0.2, decay_tau=1.0)

    calcium = observation.transform({node: latent}, time_points)[node]

    onset_idx = int(np.where(time_points >= 1.0)[0][0])
    assert calcium[onset_idx] < latent[onset_idx]
    assert np.max(calcium) <= 1.01
    assert calcium[-1] > 0.0


@pytest.mark.parametrize("model_kind", ["rate", "lds", "ctrnn", "dkb"])
def test_calcium_observation_works_for_all_supported_models(small_graph, model_kind):
    g, a, b = small_graph
    time_points = np.linspace(0, 3, 61)
    inputs = [StepInput([a], tstart=0.2, tend=1.8, value=1.0)]

    model = _build_model(model_kind, g, a, b, time_points=time_points, inputs=inputs)
    trajectories = model.simulate()

    calcium = CalciumObservation(rise_tau=0.15, decay_tau=0.8).transform(
        trajectories,
        time_points,
    )

    assert set(calcium.keys()) == set(trajectories.keys())
    assert all(trace.shape == time_points.shape for trace in calcium.values())
    assert np.max(calcium[model.neurons[b]]) >= 0.0


@pytest.mark.parametrize("model_kind", ["rate", "lds", "ctrnn", "dkb"])
def test_all_models_share_observable_contract(small_graph, model_kind):
    g, a, b = small_graph
    time_points = np.linspace(0, 2, 41)
    model = _build_model(
        model_kind,
        g,
        a,
        b,
        time_points=time_points,
        inputs=[StepInput([a], tstart=0.1, tend=1.2, value=1.0)],
    )

    trajectories = model.simulate()

    assert set(trajectories.keys()) == set(model.dynamic_neurons + model.static_neurons)
    assert all(trace.shape == time_points.shape for trace in trajectories.values())


@pytest.mark.parametrize("model_kind", ["rate", "lds", "ctrnn", "dkb"])
def test_all_models_keep_static_neurons_fixed_to_inputs(graph_with_static, model_kind):
    g, a, b, s = graph_with_static
    time_points = np.linspace(0, 2, 41)
    static_inputs = [StepInput([s], tstart=0.5, tend=1.5, value=2.0)]
    model = _build_model(
        model_kind,
        g,
        a,
        b,
        static_neurons=[s],
        time_points=time_points,
        inputs=static_inputs,
        input_neurons=[a, s],
    )

    trajectories = model.simulate()
    static_trace = trajectories[model.neurons[s]]

    assert np.allclose(static_trace[time_points <= 0.5], 0.0)
    assert np.max(static_trace) == pytest.approx(2.0)


def test_rate_and_ctrnn_share_parameter_update_contract(small_graph):
    g, a, b = small_graph
    time_points = np.linspace(0, 2, 41)

    rate_model = _build_model("rate", g, a, b, time_points=time_points)
    rate_model.set_neuron_parameters(
        {"gain": {rate_model.neurons[b]: 2.5}, "time_constant": {}, "baseline": {}}
    )
    rate_model.set_edge_parameters({"weight": {(rate_model.neurons[a], rate_model.neurons[b], 0): 1.7}})

    assert rate_model.neurons[b].gain == pytest.approx(2.5)
    assert rate_model.edges[(rate_model.neurons[a], rate_model.neurons[b], 0)]["weight"] == pytest.approx(1.7)

    ctrnn_model = _build_model("ctrnn", g, a, b, time_points=time_points)
    ctrnn_model.set_neuron_parameters(
        {
            "gain": {ctrnn_model.neurons[b]: 1.8},
            "time_constant": {ctrnn_model.neurons[b]: 0.3},
            "baseline": {},
            "activation": {},
        }
    )
    ctrnn_model.set_edge_parameters({"weight": {(ctrnn_model.neurons[a], ctrnn_model.neurons[b], 0): 1.4}})

    assert ctrnn_model.neurons[b].gain == pytest.approx(1.8)
    assert ctrnn_model.neurons[b].time_constant == pytest.approx(0.3)
    assert ctrnn_model.edges[(ctrnn_model.neurons[a], ctrnn_model.neurons[b], 0)]["weight"] == pytest.approx(1.4)


def test_lds_parameter_update_contract_supports_sparse_updates(small_graph):
    g, a, b = small_graph
    time_points = np.linspace(0, 2, 41)
    model = _build_model("lds", g, a, b, time_points=time_points)

    model.set_neuron_parameters(
        {
            "input_weight": {model.neurons[a]: 1.6},
            "baseline": {model.neurons[b]: 0.5},
        }
    )
    model.set_edge_parameters({"weight": {(model.neurons[a], model.neurons[b], 0): 1.8}})

    assert model.neurons[a].input_weight == pytest.approx(1.6)
    assert model.neurons[b].baseline == pytest.approx(0.5)
    assert model.edges[(model.neurons[a], model.neurons[b], 0)]["weight"] == pytest.approx(1.8)


def test_dynamic_weight_matrix_matches_edge_direction(small_graph):
    g, a, b = small_graph
    time_points = np.linspace(0, 1, 11)
    model = LDSModel(
        g,
        input_neurons=[],
        weights={(a, b): 2.0},
        baseline={a: 0.0, b: 0.0},
        input_weight={a: 1.0, b: 1.0},
        time_points=time_points,
    )

    matrix = model._dynamic_weight_matrix()
    state = np.array([1.5, 0.0], dtype=np.float32)
    recurrent_drive = matrix.T @ state

    assert matrix.tolist() == [
        [0.0, 2.0],
        [0.0, 0.0],
    ]
    assert recurrent_drive.tolist() == pytest.approx([0.0, 3.0])


def test_dkb_parameter_update_contract_includes_velocity_terms(small_graph):
    g, a, b = small_graph
    time_points = np.linspace(0, 2, 41)
    model = _build_model("dkb", g, a, b, time_points=time_points)

    model.set_neuron_parameters(
        {
            "damping": {model.neurons[b]: 2.2},
            "stiffness": {model.neurons[b]: 1.7},
            "baseline": {},
            "input_weight": {model.neurons[a]: 1.4},
            "target": {},
        }
    )
    model.set_edge_parameters({"weight": {(model.neurons[a], model.neurons[b], 0): 1.3}})

    assert model.neurons[b].damping == pytest.approx(2.2)
    assert model.neurons[b].stiffness == pytest.approx(1.7)
    assert model.neurons[a].input_weight == pytest.approx(1.4)
    assert model.edges[(model.neurons[a], model.neurons[b], 0)]["weight"] == pytest.approx(1.3)
