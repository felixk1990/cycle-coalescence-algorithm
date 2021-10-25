# @Author:  Felix Kramer
# @Date:   2021-10-24T22:29:34+02:00
# @Email:  kramer@mpi-cbg.de
# @Project: go-with-the-flow
# @Last modified by:    Felix Kramer
# @Last modified time: 2021-10-24T23:27:32+02:00
# @License: MIT
import networkx as nx
import numpy as np
from cycle_analysis import cycle_tools_coalescence as ctc
from cycle_analysis import test as cat


def test_nested_square():

    n = 7
    G = nx.grid_graph((n, n, 1))
    G = cat.generate_pattern(G, 'nested_square')

    weights = [G.edges[e]['weight'] for e in G.edges()]
    pos = nx.get_node_attributes(G, 'pos')

    T = ctc.coalescence()
    minimum_basis = T.construct_networkx_basis(G)
    cycle_tree = T.calc_cycle_coalescence(G, minimum_basis)
    dict_asymmetry = T.calc_tree_asymmetry()

    y = np.fromiter(dict_asymmetry.values(), dtype=float)

    assert np.all(y == 0)


def test_gradient():

    n = 7
    G = nx.grid_graph((n, n, 1))
    G = cat.generate_pattern(G, 'gradient')

    weights = [G.edges[e]['weight'] for e in G.edges()]
    pos = nx.get_node_attributes(G, 'pos')

    T = ctc.coalescence()
    minimum_basis = T.construct_networkx_basis(G)
    cycle_tree = T.calc_cycle_coalescence(G, minimum_basis)
    dict_asymmetry = T.calc_tree_asymmetry()

    y = np.fromiter(dict_asymmetry.values(), dtype=float)

    assert np.all(y == 1)
