"""
Graph manipulation, motif analysis, and transformation utilities for CeDNe

Includes neuron folding, triad enumeration, graph randomization,
and hierarchical feedforward/feedback alignment measures.
"""

__author__ = "Sahil Moza"
__date__ = "2025-04-06"
__license__ = "MIT"

import copy
import networkx as nx
import numpy as np


suffixes = [
    "",
    "D",
    "V",
    "L",
    "R",
    "DL",
    "DR",
    "VL",
    "VR",
    "01",
    "02",
    "03",
    "04",
    "05",
    "06",
    "07",
    "08",
    "09",
    "10",
    "11",
    "12",
    "13",
]


def getNeuronClass(name):
    """
    Return the neuron class name for a single neuron, by stripping the longest
    matching positional suffix (L/R/D/V/DL/DR/VL/VR or 01..13).

    Mirrors the suffix-matching used by foldByNeuronType so per-neuron class
    lookups stay consistent with how the network as a whole folds.

    Examples:
        ADAL -> ADA, AVAR -> AVA, RMDDL -> RMD, IL2DR -> IL2,
        DA01 -> DA, M2L -> M2, MCL -> MC, AVL -> AVL (unstripped — unpaired).
    """
    _suffs, _clsname = [], []
    for s in suffixes:
        if not s or not name.endswith(s):
            continue
        n0 = name[: -len(s)]
        if len(n0) > 2:
            _suffs.append(s)
            _clsname.append(n0)
        elif len(n0) > 1:
            if n0[-1] in "0123456789" or s[0] in "0123456789" or n0 in ["MC"]:
                _suffs.append(s)
                _clsname.append(n0)
    if _suffs:
        j = max(range(len(_suffs)), key=lambda i: len(_suffs[i]))
        return _clsname[j]
    return name


## Graph contraction functions
def joinLRNodes(nn_old):
    """
    Joins left and right nodes together in the neural network based on specific conditions.

    Parameters:
    - nn_old: The original neural network to be processed.

    Returns:
    - nn_new: The updated neural network after joining left and right nodes.
    """
    nn_new = nn_old.copy()
    for m in nn_new.neurons:
        if m[-1] == "L" and m not in ["AVL"]:
            nn_new.neurons[m].name = m[:-1]
            n = m[:-1] + "R"
            if n in nn_new.neurons:
                neuronPair = [m, n]
                nn_new.contract_neurons(neuronPair, m[:-1])
    return nn_new


def foldByNeuronType(
    nn_old, exceptions=[], self_loops=True, data="clean", fold_policy=None
):
    """
    Folds neurons in the given neural network based on the neuron type.

    Args:
        nn_old (NeuralNetwork): The original neural network.
        fold_policy: Optional FoldPolicySet. Forwarded to ``fold_network``;
            when ``None`` cedne falls back to its built-in defaults.

    Returns:
        NeuralNetwork: The folded neural network.
    """
    # nn_new = nn_old.copy()
    neuron_class = {}
    argmax = lambda lst: lst.index(max(lst))
    for n in nn_old.neurons:
        _suffs = []
        _clsname = []
        for s in suffixes:
            if n.endswith(s):
                n0 = n[: -len(s)]
                if len(n0) > 2:
                    _suffs.append(s)
                    _clsname.append(n0)
                elif len(n0) > 1:
                    if (
                        (n0[-1] in "0123456789")
                        or (s[0] in "0123456789")
                        or n0 in ["MC"]
                    ):
                        _suffs.append(s)
                        _clsname.append(n0)

        if len(_suffs) > 0:
            _sufflen = [len(s0) for s0 in _suffs]
            j = argmax(_sufflen)
            if _clsname[j] not in neuron_class:
                neuron_class[_clsname[j]] = []
            neuron_class[_clsname[j]].append(n)
        else:
            neuron_class[n] = [n]
    print(neuron_class)
    nn_new = nn_old.fold_network(
        neuron_class,
        exceptions=exceptions,
        self_loops=self_loops,
        data=data,
        fold_policy=fold_policy,
    )
    return nn_new


# def foldByNeuronType(nn_old):
#     """
#     Folds neurons in the given neural network based on the neuron type.

#     Args:
#         nn_old (NeuralNetwork): The original neural network.

#     Returns:
#         NeuralNetwork: The folded neural network.

#     The function iterates over each neuron in the neural network and checks if the neuron type matches a specific pattern. If the neuron type matches the pattern, the function groups the neurons together based on the neuron number. The function then contracts the grouped neurons and updates the neuron dictionary.

#     Note:
#         The function assumes that the neuron type is represented by a sequence of numbers at the end of the neuron name.
#     """
#     nn_new = nn_old.copy()
#     for m in nn_old.neurons:
#         stripNums = m.rstrip('0123456789')
#         if (len(m) - len(stripNums)==1 and m[-1]=='1') or (len(m) - len(stripNums)==2 and m[-1]=='1' and m[-2]=='0'):
#             j=1
#             moreInClass=True
#             classNeurons = []
#             while (moreInClass):
#                 if len(m) - len(stripNums) == 1:
#                     if m[:len(stripNums)] + str(j+1) in nn_new.neurons:
#                         classNeurons.append(m[:len(stripNums)] + str(j+1))
#                         j+=1
#                         moreInClass=True
#                     else:
#                         moreInClass=False
#                 else:
#                     if m[:len(stripNums)] + "{:02d}".format(j+1) in nn_new.neurons:
#                         classNeurons.append(m[:len(stripNums)] + "{:02d}".format(j+1))
#                         j+=1
#                         moreInClass=True
#                     else:
#                         moreInClass=False
#             print(m, classNeurons)
#             for n in classNeurons:
#                 if not m[:len(stripNums)] in nn_new.neurons:
#                     neuronPair = [m,n]
#                     nn_new.contractNeurons(neuronPair)
#                 else:
#                     print(m)
#             nn_new.neurons[m].name = m[:len(stripNums)]
#             nn_new.update_neurons()
#     return nn_new


def foldLeftRight(nn_old, exceptions=[], fold_policy=None):
    """
    This function performs a left-right folding operation on a neural network.
    It takes the old neural network as input and produces a new neural network after the folding operation.
    Parameters:
    - nn_old: The original neural network to be folded dorsoventrally.
    - fold_policy: Optional FoldPolicySet forwarded to ``fold_network``.
    Returns:
    - nn_new: The new neural network after the dorsoventral folding operation.
    """
    exceptions += ["AVL"]
    foldingDict = {}
    for m in nn_old.neurons:
        if m[-1] == "L" and m not in exceptions:
            n = m[:-1] + "R"
            o = m[:-1]
            if n in nn_old.neurons:
                # Skip pairs whose target name already exists as a distinct
                # neuron in the connectome (e.g. fold IL2L+IL2R -> 'IL2' is
                # fine only if 'IL2' is not itself a neuron). fold_network's
                # pre-flight will refuse the collision otherwise.
                if o in nn_old.neurons and o not in (m, n):
                    continue
                if o not in foldingDict:
                    foldingDict[o] = [m, n]
    nn_new = nn_old.fold_network(foldingDict, fold_policy=fold_policy)
    return nn_new


def foldDorsoVentral(nn_old, fold_policy=None):
    """
    This function performs a dorsoventral folding operation on a neural network.
    It takes the old neural network as input and produces a new neural network after the folding operation.
    Parameters:
    - nn_old: The original neural network to be folded dorsoventrally.
    - fold_policy: Optional FoldPolicySet forwarded to ``fold_network``.
    Returns:
    - nn_new: The new neural network after the dorsoventral folding operation.
    """
    foldingDict = {}
    for m in nn_old.neurons:
        if m[-1] == "D" and m not in ["RID"]:
            n = m[:-1] + "V"
            o = m[:-1]
            if n in nn_old.neurons:
                # Skip pairs whose target name already exists as a distinct
                # neuron (e.g. IL2DL+IL2VL -> 'IL2L' would collide with the
                # existing IL2L; fold_network would refuse the collision).
                if o in nn_old.neurons and o not in (m, n):
                    continue
                if o not in foldingDict:
                    foldingDict[o] = [m, n]
        elif m[-2] == "D" and m[-1] in ["L", "R"]:
            n = m[:-2] + "V" + m[-1]
            o = m[:-2] + m[-1]
            if n in nn_old.neurons:
                if o in nn_old.neurons and o not in (m, n):
                    continue
                if o not in foldingDict:
                    foldingDict[o] = [m, n]
    nn_new = nn_old.fold_network(foldingDict, fold_policy=fold_policy)
    # nn_new = nn_old.copy()
    # for m in nn_new.neurons:
    #     if m[-1] == 'D' and not m in ['RID']:
    #         n = m[:-1] + 'V'
    #         o = m[:-1]
    #         if n in nn_new.neurons:
    #             if o in nn_new.neurons:
    #                 neuronPair1 = [m,n] # to contract D and V
    #                 neuronPair2 = [o,m] # to contract - and D
    #                 nn_new.contract_neurons(neuronPair1)
    #                 nn_new.contract_neurons(neuronPair2)
    #             else:
    #                 nn_new.neurons[m].name = m[:-1]
    #                 neuronPair1 = [m,n] # to contract D and V
    #                 nn_new.contract_neurons(neuronPair1)
    # nn_new.update_neurons()
    return nn_new


def foldByCategory(
    nn_old, exceptions=[], self_loops=True, data="clean", fold_policy=None
):
    """
    Fold neurons by their ``category`` attribute.

    Each unique ``neuron.category`` value becomes one merged node, named
    after the category (e.g. ``HEAD``, ``MIDBODY``, ``TAIL``, ``PHARYNX``,
    ``SEX SPECIFIC``). Neurons whose ``category`` is unset (``None`` or
    empty string) are *passed through unchanged* to the folded view —
    same shape as how ``foldLeftRight`` leaves unpaired neurons alone.

    The category attribute is currently populated only by the C. elegans
    loaders (cook / witvliet), so this fold is meaningful for worms; on
    organisms whose loader doesn't set ``category`` the result is an
    empty fold and a ``ValueError`` is raised.

    Args:
        nn_old (NeuralNetwork): The original neural network.
        exceptions (list[str]): Neuron names that should pass through
            under their original name (not folded).
        self_loops (bool): Forwarded to ``fold_network``.
        data (str): Forwarded to ``fold_network`` ('collect' / 'clean').
        fold_policy (FoldPolicySet, optional): Forwarded to
            ``fold_network``; when ``None`` cedne falls back to defaults.

    Returns:
        NeuralNetwork: The folded neural network.
    """
    category_to_neurons: dict[str, list[str]] = {}
    for n, neuron in nn_old.neurons.items():
        cat = getattr(neuron, "category", None)
        if cat is None or (isinstance(cat, str) and not cat.strip()):
            continue
        category_to_neurons.setdefault(str(cat), []).append(n)

    if not category_to_neurons:
        raise ValueError(
            "No neurons carry a 'category' attribute on this network — "
            "nothing to fold. (Category folding is currently meaningful "
            "for C. elegans datasets.)"
        )

    return nn_old.fold_network(
        category_to_neurons,
        exceptions=exceptions,
        self_loops=self_loops,
        data=data,
        fold_policy=fold_policy,
    )


def is_left_neuron(n):
    """Returns if a neuron is a left neuron. This works for Worms only for now."""
    if n[-1] == "L" and n not in ["ADL", "AVL"]:
        return True


def make_hypermotifs(motif, length, join_at):
    """
    Makes hypermotifs from given set of 3-node graph motifs

    Motifs must have integers as node names starting from 1.
    """

    assert all(
        [isinstance(n, int) for n in motif.nodes]
    ), "All nodes must have integer node names"
    assert sorted(motif.nodes) == [
        *range(1, len(motif.nodes) + 1)
    ], "Nodes must be numbered 1 through number of nodes in the motif."
    assert isinstance(motif, nx.classes.digraph.DiGraph)

    motif_set = [motif.copy() for _ in range(length)]
    hypermotif = nx.union_all(motif_set, rename=(f"{j+1}." for j in range(length)))

    join_indices = [
        (f"{l}.{j[0]}", f"{l+1}.{j[1]}") for j in join_at for l in range(1, length)
    ]
    join_indices_copy = join_indices[:]
    copy_join = []
    if len(join_indices):
        _, right_ori = list(zip(*join_indices_copy))
        assert len(set(right_ori)) == len(
            right_ori
        ), "Trying to contract one node to two different nodes"

        while len(copy_join) < len(join_indices):
            left, right = list(zip(*join_indices_copy))
            ind_copy = [j for j, r in enumerate(right) if r not in left]
            ind = [right_ori.index(right[i]) for i in ind_copy]
            copy_join += ind
            for j in ind_copy[::-1]:
                join_indices_copy.pop(j)

        mapping = {}
        for j in copy_join:
            ja = join_indices[j]
            nx.contracted_nodes(hypermotif, ja[0], ja[1], copy=False)
            mapping.update({f"{ja[0]}": f"{ja[0]}-{ja[1]}"})
        nx.relabel_nodes(hypermotif, mapping, copy=False)
    return hypermotif


def return_triads():
    triads = (
        "003",
        "012",
        "102",
        "021D",
        "021U",
        "021C",
        "111D",
        "111U",
        "030T",
        "030C",
        "201",
        "120D",
        "120U",
        "120C",
        "210",
        "300",
    )

    triad_graphs = {t: nx.triad_graph(t) for t in triads}
    for t in triad_graphs:
        triad_graphs[t] = nx.relabel_nodes(
            triad_graphs[t], mapping={"a": 1, "b": 2, "c": 3}, copy=True
        )
    return triad_graphs


def randomize_graph(
    G, seed=None, mode="edge-swap", multiplier=None, edge_subgroups=None, data=True
):
    """Randomizes a directed graph using specified methods. Also randmize within graph subgroups.
    Parameters:
    - G: The directed graph to be randomized.
    - seed: Random seed for reproducibility.
    - mode: Method of randomization. Options are 'edge-swap', 'configuration-model', or 'num-nodes-edges'.
    - multiplier: Multiplier for the number of swaps or edges. If None, it uses log of the number of edges as default.
    - subgroups: Takes a list of subgroups to randomize within. If None, randomizes the entire graph.
    Returns:
    - g_copy: A new directed graph that is a randomized version of G.
    Raises:
    - ValueError: If the multiplier is not an integer when not 'auto'.
    """

    if multiplier == None:
        multiplier = int(np.log(len(G.edges)))
    else:
        if not isinstance(multiplier, int):
            raise ValueError("Multiplier must be an integer")
    # Resolve seed via the package-wide SeedSequence factory so:
    #   * seed is None → deterministic int spawned from cedne's root
    #     SeedSequence (so randomize_graph() with no args is reproducible
    #     across runs, and independent from any other get_rng/get_seed
    #     call in the process).
    #   * seed is int  → use that seed directly.
    # Pass `seed` through to every ``nx.*(seed=seed)`` call below. We
    # deliberately do NOT mutate ``np.random`` global state here (the old
    # ``np.random.seed(seed)`` would have silently affected every later
    # ``np.random.*`` call in the same process).
    from cedne.random import get_seed

    seed = get_seed(seed)

    if edge_subgroups is not None:
        g_copy = G.copy_neurons()
        if not isinstance(edge_subgroups, list):
            raise ValueError("Edge Subgroups must be a list of lists")
        for subgroup in edge_subgroups:
            if not isinstance(subgroup, list):
                raise ValueError("Each subgroup must be a list of edges")
            subgraph = G.subnetwork(connections=subgroup, data=data)
            if len(subgraph.edges) > 0:
                if mode == "edge-swap":
                    if len(subgraph) < 4 or len(subgraph.edges) < 3:
                        # Cannot randomize very small subgraphs while preserving degrees
                        # using the double-edge swap method.
                        continue
                    multiplier = int(np.log(len(subgraph.edges)))
                    nswap = int(multiplier * len(subgraph.edges))
                    try:
                        nx.directed_edge_swap(
                            subgraph, nswap=nswap, max_tries=nswap * 100, seed=seed
                        )
                    except (nx.NetworkXAlgorithmError, nx.NetworkXError):
                        fallback_nswap = int(
                            len(subgraph.edges) * 0.01
                        )  # or some other conservative estimate
                        print(f"Retrying with fallback nswap={fallback_nswap}")
                        try:
                            nx.directed_edge_swap(
                                subgraph,
                                nswap=fallback_nswap,
                                max_tries=fallback_nswap * 1000,
                                seed=seed,
                            )
                        except nx.NetworkXAlgorithmError:
                            print("Still failed, skipping this subgraph.")
                elif mode == "configuration-model":
                    nodes = subgraph.nodes()
                    in_degree = [subgraph.in_degree(n) for n in nodes]
                    out_degree = [subgraph.out_degree(n) for n in nodes]
                    subgraph = nx.directed_configuration_model(
                        in_degree, out_degree, seed=seed
                    )
                elif mode == "num-nodes-edges":
                    subnet = nx.gnm_random_graph(
                        len(subgraph.nodes()),
                        len(subgraph.edges()),
                        seed=seed,
                        directed=True,
                    )
                    nodelist = list(subgraph.nodes())
                    neurons = [nodelist[n] for n in subnet.nodes]
                    edge_dict = {
                        (neurons[e[0]].name, neurons[e[1]].name): {}
                        for e in subnet.edges
                    }
                    subgraph.remove_all_connections()
                    subgraph.create_connections(edge_dict)
                elif mode == "stub-matching":
                    if not subgroup:
                        continue  # skip empty subgroups

                    # Extract all source and target nodes in the subgroup
                    src_nodes = set(e[0] for e in subgroup)
                    tgt_nodes = set(e[1] for e in subgroup)

                    # Infer src_type and tgt_type from node annotations
                    src_type_set = {G.nodes[n]["type"] for n in src_nodes}
                    tgt_type_set = {G.nodes[n]["type"] for n in tgt_nodes}

                    if len(src_type_set) != 1 or len(tgt_type_set) != 1:
                        print(
                            f"Skipping subgroup with mixed neuron types: {src_type_set} → {tgt_type_set}"
                        )
                        continue

                    src_type = next(iter(src_type_set))
                    tgt_type = next(iter(tgt_type_set))
                    num_edges = len(subgroup)

                    # All nodes in the whole graph of the correct type
                    all_src = [
                        n for n in subgraph.nodes if G.neurons[n.name].type == src_type
                    ]
                    all_tgt = [
                        n for n in subgraph.nodes if G.neurons[n.name].type == tgt_type
                    ]

                    # All possible edges between src_type and tgt_type, excluding self-loops
                    possible_edges = [
                        (u, v) for u in all_src for v in all_tgt if u != v
                    ]

                    if len(possible_edges) < num_edges:
                        print(
                            f"Not enough possible edges for {src_type}→{tgt_type} (have {len(possible_edges)}, need {num_edges})"
                        )
                        continue

                    # Randomly sample edges without replacement
                    rng = np.random.default_rng(seed)
                    sampled_edges = rng.choice(
                        possible_edges, size=num_edges, replace=False
                    )

                    # Create edge dict for your custom graph object
                    edge_dict = {(u.name, v.name): {} for u, v in sampled_edges}
                    subgraph.remove_all_connections()
                    subgraph.create_connections(edge_dict)
                else:
                    raise NotImplementedError(
                        f"{mode} not in implemented modes for this method."
                    )
                g_copy.create_connections_from(subgraph, data=data)
    else:
        g_copy = copy.deepcopy(G)
        if mode == "edge-swap":
            nswap = int(multiplier * len(G.edges))
            nx.directed_edge_swap(g_copy, nswap=nswap, max_tries=nswap * 100, seed=seed)
        elif mode == "configuration-model":
            nodes = g_copy.nodes()
            in_degree = [g_copy.in_degree(n) for n in nodes]
            out_degree = [g_copy.out_degree(n) for n in nodes]
            g_copy = nx.directed_configuration_model(in_degree, out_degree, seed=seed)
        elif mode == "num-nodes-edges":
            g_copy = nx.gnm_random_graph(
                len(g_copy.nodes()), len(g_copy.edges()), seed=seed, directed=True
            )
    return g_copy


def addBranch():
    """Add parallel and serial branches to a graph."""
    pass


def hierarchical_alignment(conns):
    ntype = ["sensory", "interneuron", "motorneuron"]
    ntype_pairs = [(n1, n2) for n1 in ntype for n2 in ntype]
    conn_types = {n: 0 for n in ntype_pairs}
    for edge in conns:
        if (edge[0].type in ntype) and (edge[1].type in ntype):
            conn = (edge[0].type, edge[1].type)
            conn_types[conn] += 1
    feedforward = (
        conn_types[("sensory", "interneuron")]
        + conn_types[("sensory", "motorneuron")]
        + conn_types[("interneuron", "motorneuron")]
    )
    feedback = (
        conn_types[("interneuron", "sensory")]
        + conn_types[("motorneuron", "interneuron")]
        + conn_types[("motorneuron", "sensory")]
    )
    lateral = 0  # conn_types[('sensory', 'sensory')] + conn_types[('interneuron', 'interneuron')] + conn_types[('motorneuron', 'motorneuron')]
    print(feedforward, feedback)
    # return (feedforward+lateral)/(feedforward+lateral+feedback) if (feedforward+lateral+feedback) else 0
    return (
        (feedforward - feedback) / (feedforward + feedback + lateral)
        if (feedforward + feedback + lateral)
        else 0
    )


# ---------------------------------------------------------------------------
# Sampled counting of chained-FFL motifs (sequential hierarchies and
# intermediate node-chains) on graphs too large for exhaustive VF2 search.
#
# Semantics match ``NervousSystem.search_motifs`` (VF2 *induced* subgraph
# isomorphism) exactly, including its autapse rule: a node carrying a
# self-loop can never match a motif position, because the induced subgraph
# on the matched nodes would contain the self-loop that the motif lacks.
# Validated against exact VF2 counts on the C. elegans chemical connectome
# (tests/test_chain_sampler.py).
# ---------------------------------------------------------------------------


def remove_autapse_nodes(edges):
    """Drop self-loops and every edge touching a self-looped node.

    Encodes the VF2-induced autapse rule explicitly so that downstream
    enumeration can assume a simple loop-free digraph. Returns
    ``(kept_edges, autapse_nodes)``.
    """
    autapse_nodes = {a for a, b in edges if a == b}
    kept = [
        (a, b)
        for a, b in edges
        if a != b and a not in autapse_nodes and b not in autapse_nodes
    ]
    return kept, autapse_nodes


def enumerate_induced_ffls(edges):
    """Enumerate all induced feed-forward-loop instances in an edge list.

    An FFL instance is an ordered node triple ``(input, intermediate,
    output)`` with edges input->intermediate, input->output,
    intermediate->output and *no other edges* among the three nodes
    (induced semantics; reciprocal edges disqualify the triad).

    Edges must be loop-free (see ``remove_autapse_nodes``). Returns
    ``(ffl_list, edge_set)``.
    """
    from collections import defaultdict

    edge_set = set(edges)
    successors, predecessors = defaultdict(set), defaultdict(set)
    for a, b in edges:
        successors[a].add(b)
        predecessors[b].add(a)
    ffls = []
    for i, k in edges:
        for j in successors[i] & predecessors[k]:
            if j == i or j == k:
                continue
            if (k, i) in edge_set or (j, i) in edge_set or (k, j) in edge_set:
                continue
            ffls.append((i, j, k))
    return ffls, edge_set


def _chain_nodes(ffl_list, join_kind):
    """Ordered node tuple of a chain candidate, or None if nodes collide.

    ``join_kind='seq'`` joins each FFL's output to the next FFL's input
    (a sequential hierarchy, the ``(3, 1)`` hypermotif mapping);
    ``join_kind='int'`` joins the intermediate instead (the ``(2, 1)``
    mapping, an intermediate node-chain).
    """
    nodes = list(ffl_list[0])
    for prev, cur in zip(ffl_list, ffl_list[1:]):
        shared = prev[2] if join_kind == "seq" else prev[1]
        if cur[0] != shared:
            return None
        nodes.extend([cur[1], cur[2]])
    return None if len(set(nodes)) != len(nodes) else tuple(nodes)


def _is_induced_chain(ffl_list, join_kind, edge_set):
    """True iff the FFL tuple forms a valid induced chain match.

    Requires distinct nodes and that the induced edge set among them is
    exactly the union of the constituent FFLs' edges.
    """
    nodes = _chain_nodes(ffl_list, join_kind)
    if nodes is None:
        return False
    required = set()
    for i, j, k in ffl_list:
        required |= {(i, j), (i, k), (j, k)}
    for a in nodes:
        for b in nodes:
            if a != b and (((a, b) in edge_set) != ((a, b) in required)):
                return False
    return True


class ChainSampler:
    """Horvitz-Thompson estimation of chained-FFL counts by uniform
    sampling over the exact candidate space of FFL joins.

    A chain candidate of length L is an ordered tuple of L induced FFLs
    joined at shared nodes; the number of candidates is computed exactly
    from per-node join bookkeeping, candidates are sampled uniformly, and
    the chain count is estimated as ``candidate_total x acceptance_rate``
    where acceptance applies the full induced/distinctness check. Accepted
    candidates form a uniform sample of true chain matches, reusable for
    composition analyses.

    Chain lengths 1-3 are supported. Edges must be loop-free; apply
    ``remove_autapse_nodes`` first to match ``search_motifs`` semantics.
    """

    def __init__(self, edges, seed=0):
        from collections import defaultdict

        if any(a == b for a, b in edges):
            raise ValueError(
                "edge list contains self-loops; apply remove_autapse_nodes first"
            )
        self.rng = np.random.default_rng(seed)
        self.ffls, self.edge_set = enumerate_induced_ffls(edges)
        self.by_input = defaultdict(list)
        self.by_intermediate = defaultdict(list)
        self.by_output = defaultdict(list)
        for idx, (i, j, k) in enumerate(self.ffls):
            self.by_input[i].append(idx)
            self.by_intermediate[j].append(idx)
            self.by_output[k].append(idx)
        self._cache = {}

    def _source(self, join_kind):
        if join_kind not in ("seq", "int"):
            raise ValueError("join_kind must be 'seq' or 'int'")
        return self.by_output if join_kind == "seq" else self.by_intermediate

    def _join_node_fn(self, join_kind):
        return (lambda f: f[2]) if join_kind == "seq" else (lambda f: f[1])

    def candidate_total(self, length, join_kind):
        """Exact number of ordered FFL tuples satisfying the join constraints."""
        if length == 1:
            return len(self.ffls)
        source = self._source(join_kind)
        if length == 2:
            return int(sum(len(source[v]) * len(self.by_input[v]) for v in source))
        if length == 3:
            join_node = self._join_node_fn(join_kind)
            return int(
                sum(
                    len(source.get(f2[0], []))
                    * len(self.by_input.get(join_node(f2), []))
                    for f2 in self.ffls
                )
            )
        raise ValueError("chain length must be 1, 2 or 3")

    def _iter_candidates(self, length, join_kind, n_samples):
        # Uniform candidate draws, batched: the weighted first-stage choice is
        # one cumulative-weight table + searchsorted over all draws (O(log n)
        # per draw), instead of O(n) per draw — required at millions of FFLs.
        source = self._source(join_kind)
        join_node = self._join_node_fn(join_kind)
        if length == 2:
            key = ("pair", join_kind)
            if key not in self._cache:
                join_vs = [v for v in source if self.by_input.get(v)]
                weights = np.array(
                    [len(source[v]) * len(self.by_input[v]) for v in join_vs],
                    dtype=float,
                )
                self._cache[key] = (join_vs, np.cumsum(weights / weights.sum()))
            join_vs, cumweights = self._cache[key]
            first_stage = np.searchsorted(
                cumweights, self.rng.random(n_samples), side="right"
            )
            for v_idx in first_stage:
                v = join_vs[min(v_idx, len(join_vs) - 1)]
                f1 = self.ffls[source[v][self.rng.integers(len(source[v]))]]
                f2 = self.ffls[
                    self.by_input[v][self.rng.integers(len(self.by_input[v]))]
                ]
                yield [f1, f2]
            return
        key = ("triple", join_kind)
        if key not in self._cache:
            weights = np.array(
                [
                    len(source.get(f2[0], []))
                    * len(self.by_input.get(join_node(f2), []))
                    for f2 in self.ffls
                ],
                dtype=float,
            )
            self._cache[key] = np.cumsum(weights / weights.sum())
        cumweights = self._cache[key]
        first_stage = np.searchsorted(
            cumweights, self.rng.random(n_samples), side="right"
        )
        for f2_idx in first_stage:
            f2 = self.ffls[min(f2_idx, len(self.ffls) - 1)]
            pool1 = source[f2[0]]
            pool3 = self.by_input[join_node(f2)]
            f1 = self.ffls[pool1[self.rng.integers(len(pool1))]]
            f3 = self.ffls[pool3[self.rng.integers(len(pool3))]]
            yield [f1, f2, f3]

    def estimate(self, length, join_kind, n_samples):
        """Sampled chain count.

        Returns ``(count_estimate, ci95_half_width, accepted_chains)`` where
        ``accepted_chains`` is a uniform sample of true chain matches as
        ordered node tuples.
        """
        if length == 1:
            return float(len(self.ffls)), 0.0, [tuple(f) for f in self.ffls]
        total = self.candidate_total(length, join_kind)
        if total == 0:
            return 0.0, 0.0, []
        hits, accepted = 0, []
        for candidate in self._iter_candidates(length, join_kind, n_samples):
            if _is_induced_chain(candidate, join_kind, self.edge_set):
                hits += 1
                accepted.append(_chain_nodes(candidate, join_kind))
        p_hat = hits / n_samples
        std_err = np.sqrt(max(p_hat * (1 - p_hat), 1e-12) / n_samples)
        return total * p_hat, total * 1.96 * std_err, accepted

    def exhaustive(self, length, join_kind):
        """Exact chain count by checking every candidate (small graphs only)."""
        if length == 1:
            return len(self.ffls)
        source = self._source(join_kind)
        join_node = self._join_node_fn(join_kind)
        count = 0
        if length == 2:
            for v in source:
                for i1 in source[v]:
                    for i2 in self.by_input.get(v, []):
                        if _is_induced_chain(
                            [self.ffls[i1], self.ffls[i2]], join_kind, self.edge_set
                        ):
                            count += 1
            return count
        if length == 3:
            for f2 in self.ffls:
                for i1 in source.get(f2[0], []):
                    for i3 in self.by_input.get(join_node(f2), []):
                        if _is_induced_chain(
                            [self.ffls[i1], f2, self.ffls[i3]],
                            join_kind,
                            self.edge_set,
                        ):
                            count += 1
            return count
        raise ValueError("chain length must be 1, 2 or 3")
