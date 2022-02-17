# @Author:  Felix Kramer
# @Date:   2021-10-31T13:17:49+01:00
# @Email:  kramer@mpi-cbg.de
# @Project:  cycle_analysis
# @Last modified by:    Felix Kramer
# @Last modified time: 2021-11-02T11:20:47+01:00
# @License: MIT
import networkx as nx
import numpy as np
import random
from cycle_analysis.cycle_tools_coalescence import  *
from cycle_analysis.cycle_tools_simple import  *


def test_nullity():

    G = nx.grid_graph((5, 5, 1))

    E = nx.number_of_edges(G)
    N = nx.number_of_nodes(G)
    CC = nx.number_connected_components(G)
    nullity = E-N+CC

    T = Coalescence()
    minimum_basis = construct_networkx_basis(G)

    assert len(minimum_basis) == nullity


def test_independence():

    G = nx.grid_graph((5, 5, 1))

    T = Coalescence()
    minimum_basis = construct_networkx_basis(G)

    rows = len(minimum_basis)
    cols = nx.number_of_edges(G)
    E = np.zeros((rows, cols))

    for i, c in enumerate(minimum_basis):

        e_row = np.zeros(cols)
        idx = [j for j, e in enumerate(G.edges()) if c.has_edge(*e)]
        e_row[idx] = 1

        E[i] = e_row

    linear_independent = compute_linear_independence(E.T)

    assert linear_independent


def test_minimal_weight():

    G = nx.grid_graph((7, 7, 1))

    T = Coalescence()
    min_basis = construct_networkx_basis(G)
    min_weight = sum([nx.number_of_edges(c) for c in min_basis])

    sample = 500
    roots = random.choices(list(G.nodes()), k=sample)
    sample_weight = []

    for root in roots:

        list_cycles = compute_cycles_superlist(G,root)
        sample_weight.append(sum([len(lc) for lc in list_cycles]))

    assert np.all(np.array(sample_weight) >= min_weight)


def test_bfs_tree():

    G = nx.grid_graph((7, 7, 1))

    T = Coalescence()

    cc = []
    edges = []
    for n in G.nodes():

        spanning_tree, dict_path = breadth_first_tree(G, n)

        cc.append(nx.number_connected_components(spanning_tree))
        edges.append(nx.number_of_edges(spanning_tree))

    N = nx.number_of_nodes(G)

    assert np.all(np.array(cc) == 1) and np.all(np.array(edges) == N-1)


def test_find_cycle():

    G = nx.grid_graph((7, 7, 1))

    T = Coalescence()

    E = nx.number_of_edges(G)
    N = nx.number_of_nodes(G)
    CC = nx.number_connected_components(G)
    null = E - N + CC
    null_counter = np.zeros(N)

    for i, n in enumerate(G.nodes()):

        spanning_tree, dict_path = breadth_first_tree(G, n)

        diff_graph = nx.difference(G, spanning_tree)
        list_diff = []

        for e in diff_graph.edges():
            new_path, new_edges = find_cycle(dict_path, e, n)
            list_diff.append(len(new_edges)-len(new_path))
            null_counter[i] += 1

    assert np.all(np.array(list_diff) == 0) and np.all(null_counter == null)
