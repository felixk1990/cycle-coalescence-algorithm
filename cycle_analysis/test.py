# @Author: Felix Kramer <kramer>
# @Date:   18-02-2019
# @Email:  felix.kramer@hotmail.de
# @Project: cycle-coalescecne-algorithm
# @Last modified by:    Felix Kramer
# @Last modified time: 2021-10-28T23:15:04+02:00

import networkx as nx
import numpy as np
import random as rd
import cycle_analysis.cycle_tools_simple as cc
# generate example pattern random/nested/gradient for testing baseline values
# of the order parameter


def generate_pattern(nx_graph, mode):

    pos = nx.spectral_layout(nx_graph)
    for n in pos.keys():
        nx_graph.nodes[n]['pos'] = pos[n]

    if 'random' == mode:

        generate_pattern_random(nx_graph)

    elif 'gradient' == mode:
        generate_pattern_gradient(nx_graph)

    elif 'nested_square' == mode:
        generate_pattern_nestedSquare(nx_graph)

    return nx_graph


def generate_pattern_random(nx_graph):

    for e in nx_graph.edges():
        nx_graph.edges[e]['weight'] = rd.uniform(0., 1.)*10.


def generate_pattern_gradient(nx_graph):

    list_n = list(nx_graph.nodes())
    idx_min = np.argmin([nx_graph.nodes[n]['pos'][0] for n in list_n])
    ref_p = nx_graph.nodes[list_n[idx_min]]['pos']
    for e in nx_graph.edges():
        q1 = np.array(nx_graph.nodes[e[0]]['pos'])
        q2 = np.array(nx_graph.nodes[e[1]]['pos'])
        p = (q1 + q2)/2.
        r = np.linalg.norm(p-ref_p)
        nx_graph.edges[e]['weight'] = 1./r


def generate_pattern_nestedSquare(nx_graph):

    iteration = 0
    corners = get_corners(nx_graph)
    my_tiles = get_first_tile(nx_graph, corners)

    E = nx.number_of_edges(nx_graph)
    N = nx.number_of_nodes(nx_graph)

    counter = 0
    go_on = True
    while go_on:

        new_my_tiles = []
        graph_seen = nx.Graph()
        dict_seen = {}

        for tile in my_tiles:
            args = [new_my_tiles, tile, graph_seen, dict_seen, counter]
            sub_tile, counter = calc_new_tile(*args)

        new_nx_graph = nx.Graph()
        for tile in new_my_tiles:
            new_nx_graph = nx.compose(new_nx_graph, tile)

        nx_graph = nx.Graph(new_nx_graph)
        T = cc.simple()
        basis = T.construct_networkx_basis(new_nx_graph)
        simple_basis = [nx.Graph(b) for b in basis]
        for b in simple_basis:
            for n in b.nodes():
                b.nodes[n]['pos'] = nx_graph.nodes[n]['pos']
            for e in b.edges():
                b.edges[e]['weight'] = nx_graph.edges[e]['weight']
        my_tiles = simple_basis

        iteration += 1
        numE = list(nx.get_edge_attributes(sub_tile, 'weight').values())
        numN = nx.number_of_nodes(new_nx_graph)
        if numE == E or numN == N or iteration == 3:
            nx_graph = new_nx_graph
            go_on = False
            break


def calc_new_tile(new_my_tiles, tile, graph_seen, dict_seen, counter):

    list_e = list(tile.edges())
    sub_tile = nx.Graph()

    w = tile.edges[list_e[0]]['weight']
    sub_tile.add_edge(*list_e[0], weight=w)
    push_1 = [0]
    push_2 = []

    for i, e in enumerate(list_e[1:]):
        if (sub_tile.has_node(e[0]) or sub_tile.has_node(e[1])):
            push_2.append(i+1)
        else:
            push_1.append(i+1)

    pos = []
    for i, n in enumerate(tile):
        p = tile.nodes[n]['pos']
        sub_tile.add_node(n, pos=p)
        pos.append(p)

    my_center = counter
    sub_tile.add_node(my_center, pos=np.mean(pos, axis=0))
    counter += 1
    for i, e in enumerate(list_e[1:]):
        sub_tile.add_edge(*e, weight=tile.edges[e]['weight'])

    w = list(nx.get_edge_attributes(sub_tile, 'weight').values())
    sub_w = np.amin(w)/2.

    push_nodes_1 = []
    push_nodes_2 = []

    for i, e in enumerate(list_e):

        if i in push_1:

            if graph_seen.has_edge(*e):

                sub_tile, node_id = use_a_brick(sub_tile, e,  dict_seen)
                push_nodes_1.append(node_id)
            else:
                push_nodes_1.append(counter)
                sub_tile = form_a_brick(tile, sub_tile, counter, e,  dict_seen)
                graph_seen.add_edge(*e)
                counter += 1

        elif i in push_2:

            if graph_seen.has_edge(*e):

                sub_tile, node_id = use_a_brick(sub_tile, e,  dict_seen)
                push_nodes_2.append(node_id)
            else:
                push_nodes_2.append(counter)
                sub_tile = form_a_brick(tile, sub_tile, counter, e,  dict_seen)
                graph_seen.add_edge(*e)
                counter += 1

    for i in push_nodes_1:
        sub_tile.add_edge(my_center, i, weight=sub_w)
    for i in push_nodes_2:
        sub_tile.add_edge(my_center, i, weight=sub_w*0.9)

    new_my_tiles.append(sub_tile)

    return sub_tile, counter


def get_corners(nx_graph):

    dim = len(list(nx.get_node_attributes(nx_graph, 'pos').values())[0])
    corners = []
    if dim == 2:
        corners = [n for n in nx_graph.nodes() if nx_graph.degree(n) == 2]
    elif dim == 3:
        corners = [n for n in nx_graph.nodes() if nx_graph.degree(n) == 3]

    return corners


def get_first_tile(nx_graph, corners):

    side_length = np.sqrt(nx.number_of_nodes(nx_graph))
    w = 5.
    tile = nx.Graph()
    for i, n in enumerate(corners):
        tile.add_node(n, pos=nx_graph.nodes[n]['pos'])
    for i, n in enumerate(corners[:-1]):
        for j, m in enumerate(corners[i+1:]):
            path = nx.shortest_path(nx_graph, n, m)
            if len(path) == side_length:
                tile.add_edge(n, m, weight=w)

    return [tile]


def use_a_brick(sub_tile, edge, dict_seen):

    if edge in dict_seen:
        brick = dict_seen[edge]
    else:
        brick = dict_seen[(edge[1], edge[0])]

    for n in brick.nodes():
        if brick.degree(n) == 2:
            node_id = n

    sub_tile = nx.compose(sub_tile, brick)
    sub_tile.remove_edge(*edge)

    return sub_tile, node_id


def form_a_brick(tile, sub_tile, node_id, edge, dict_seen):

    pos = (tile.nodes[edge[0]]['pos'] + tile.nodes[edge[1]]['pos'])/2.

    brick = nx.Graph()
    brick.add_node(node_id, pos=pos)
    brick.add_edge(node_id, edge[0], weight=sub_tile.edges[edge]['weight'])
    brick.add_edge(node_id, edge[1], weight=sub_tile.edges[edge]['weight'])
    sub_tile = nx.compose(sub_tile, brick)

    sub_tile.remove_edge(*edge)
    dict_seen[edge] = brick
    return sub_tile
