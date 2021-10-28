# @Author: Felix Kramer <kramer>
# @Date:   04-05-2021
# @Email:  kramer@mpi-cbg.de
# @Project: phd_network_remodelling
# @Last modified by:    Felix Kramer
# @Last modified time: 2021-10-28T21:14:32+02:00
import networkx as nx


class toolbox():

    def __init__(self):

        self.G = nx.Graph()

    def extract_path_origin(self, cycle):

        circle = nx.Graph(cycle)
        path = []

        E = nx.edges(circle)
        e0 = list(E)[0]
        circle.remove_edge(*e0)
        sp = nx.shortest_path(circle, e0[0], e0[1])

        for i, p in enumerate(sp):

            path.append(cycle.nodes[p]['pos'])

        return path
