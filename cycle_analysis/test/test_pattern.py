# @Author:  Felix Kramer
# @Date:   2021-10-24T22:29:34+02:00
# @Email:  kramer@mpi-cbg.de
# @Project:  cycle_analysis
# @Last modified by:    Felix Kramer
# @Last modified time: 2021-11-02T10:40:30+01:00
# @License: MIT
import networkx as nx
import numpy as np
from cycle_analysis import cycle_tools_coalescence as ctc
import cycle_analysis.cycle_custom_pattern as ccp


def test_nested_square():

    n = 7
    G = nx.grid_graph((n, n, 1))
    G = ccp.generate_pattern(G, 'nested_square')

    T = ctc.coalescence()
    dict_asymmetry = T.calc_tree_asymmetry()

    y = np.fromiter(dict_asymmetry.values(), dtype=float)

    assert np.all(y == 0)


def test_gradient():

    n = 7
    G = nx.grid_graph((n, n, 1))
    G = ccp.generate_pattern(G, 'gradient')

    T = ctc.coalescence()
    dict_asymmetry = T.calc_tree_asymmetry()

    y = np.fromiter(dict_asymmetry.values(), dtype=float)

    assert np.all(y == 1)
