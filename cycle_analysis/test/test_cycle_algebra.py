# @Author:  Felix Kramer
# @Date:   2021-11-02T10:39:35+01:00
# @Email:  kramer@mpi-cbg.de
# @Project:  cycle_analysis
# @Last modified by:    Felix Kramer
# @Last modified time: 2021-11-02T11:18:32+01:00
# @License: MIT

import networkx as nx
from cycle_analysis import cycle_tools_coalescence as ctc


def test_merging():

    T = ctc.coalescence()
    n1 = 5
    n2 = 5

    # create dummy cycles that share exactly one edge
    c1 = nx.Graph()
    for i in range(n1):
        c1.add_edge(i, i+1)
    c1.add_edge(0, n1)

    c2 = nx.Graph()
    for i in range(n1-1, n1-1+n2):
        c2.add_edge(i, i+1)
    c2.add_edge(n1-1, n1-1+n2)
    mc1 = T.merge_cycles(c1, c2)

    l11 = nx.number_of_edges(c1)
    l12 = nx.number_of_edges(c2)
    l1m = nx.number_of_edges(mc1)

    d1 = l11+l12-l1m-2

    # create dummy cycles that share no edge
    c1 = nx.Graph()
    for i in range(n1):
        c1.add_edge(i, i+1)
    c1.add_edge(0, n1)

    c2 = nx.Graph()
    for i in range(n1+1, n1+1+n2):
        c2.add_edge(i, i+1)
    c2.add_edge(n1+1, n2)

    mc2 = T.merge_cycles(c1, c2)

    l21 = nx.number_of_edges(c1)
    l22 = nx.number_of_edges(c2)
    l2m = nx.number_of_edges(mc2)
    d2 = l21+l22-l2m

    # create dummy cycles which are identical
    c1 = nx.Graph()
    for i in range(n1):
        c1.add_edge(i, i+1)
    c1.add_edge(0, n1)

    c2 = nx.Graph(c1)

    mc3 = T.merge_cycles(c1, c2)

    l31 = nx.number_of_edges(c1)
    l32 = nx.number_of_edges(c2)
    l3m = nx.number_of_edges(mc3)
    d3 = l31+l32-l3m-(n1+n2+2)

    assert d1 == 0 and d2 == 0 and d3 == 0
