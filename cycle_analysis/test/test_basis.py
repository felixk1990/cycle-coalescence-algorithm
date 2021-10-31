# @Author:  Felix Kramer
# @Date:   2021-10-31T13:17:49+01:00
# @Email:  kramer@mpi-cbg.de
# @Project: go-with-the-flow
# @Last modified by:    Felix Kramer
# @Last modified time: 2021-10-31T13:27:19+01:00
# @License: MIT
import networkx as nx
import numpy as np
from cycle_analysis import cycle_tools_coalescence as ctc

def test_nullity():

    G = nx.grid_graph((5,5,1))

    E = nx.number_of_edges(G)
    N = nx.number_of_nodes(G)
    CC = nx.number_connected_components(G)
    nullity = E-N+CC

    T = ctc.coalescence()
    minimum_basis = T.construct_networkx_basis(G)

    assert len(minimum_basis) == nullity

# def test_independence():
#
#     pass
#
# def test_minimal_weight():
#
#     pass
#
# def test_attributes():
#
#     pass
