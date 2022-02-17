# @Author:  Felix Kramer
# @Date:   2021-10-24T22:29:34+02:00
# @Email:  kramer@mpi-cbg.de
# @Project:  cycle_analysis
# @Last modified by:    Felix Kramer
# @Last modified time: 2021-11-04T22:08:56+01:00
# @License: MIT
import networkx as nx
import numpy as np
from cycle_analysis.cycle_tools_coalescence import  *
from cycle_analysis.cycle_tools_simple import  *
from cycle_analysis.cycle_custom_pattern import  *


def test_nested_square():

    n = 7
    G = nx.grid_graph((n, n, 1))
    G = generate_pattern(G, 'nested_square')

    T = Coalescence()
    dict_asymmetry = T.calc_cycle_asymmetry(G)

    y = np.fromiter(dict_asymmetry.values(), dtype=float)

    assert np.all(y == 0)


def test_gradient():

    n = 7
    G = nx.grid_graph((n, n, 1))
    G = generate_pattern(G, 'gradient')

    T = Coalescence()
    dict_asymmetry = T.calc_cycle_asymmetry(G)

    y = np.fromiter(dict_asymmetry.values(), dtype=float)

    assert np.all(y == 1)
