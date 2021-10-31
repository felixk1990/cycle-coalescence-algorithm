# @Author:  Felix Kramer
# @Date:   2021-10-31T13:17:49+01:00
# @Email:  kramer@mpi-cbg.de
# @Project: go-with-the-flow
# @Last modified by:    Felix Kramer
# @Last modified time: 2021-10-31T15:08:07+01:00
# @License: MIT
import networkx as nx
import numpy as np
import random
from cycle_analysis import cycle_tools_coalescence as ctc


def test_nullity():

    G = nx.grid_graph((5, 5, 1))

    E = nx.number_of_edges(G)
    N = nx.number_of_nodes(G)
    CC = nx.number_connected_components(G)
    nullity = E-N+CC

    T = ctc.coalescence()
    minimum_basis = T.construct_networkx_basis(G)

    assert len(minimum_basis) == nullity


def test_independence():

    G = nx.grid_graph((5, 5, 1))

    T = ctc.coalescence()
    minimum_basis = T.construct_networkx_basis(G)

    rows = len(minimum_basis)
    cols = nx.number_of_edges(G)
    E = np.zeros((rows, cols))

    for i, c in enumerate(minimum_basis):

        e_row = np.zeros(cols)
        idx = [j for j, e in enumerate(G.edges()) if c.has_edge(*e)]
        e_row[idx] = 1

        E[i] = e_row

    linear_independent = T.compute_linear_independence(E.T)

    assert linear_independent


def test_minimal_weight():

    G = nx.grid_graph((7, 7, 1))

    T = ctc.coalescence()
    min_basis = T.construct_networkx_basis(G)
    min_weight = sum([nx.number_of_edges(c) for c in min_basis])

    sample = 500
    roots = random.choices(list(G.nodes()), k=sample)
    sample_weight = []

    for root in roots:

        list_cycles = T.compute_cycles_superlist(root)
        sample_weight.append(sum([len(lc) for lc in list_cycles]))

    assert np.all(np.array(sample_weight) >= min_weight)
# def test_attributes():
#
#     pass
