# @Author: Felix Kramer <kramer>
# @Date:   18-02-2019
# @Email:  felix.kramer@hotmail.de
# @Project: cycle-coalescecne-algorithm
# @Last modified by:    Felix Kramer
# @Last modified time: 2021-10-28T21:59:23+02:00
import networkx as nx
import numpy as np
import cycle_analysis.cycle_tools_simple as cycle_tools_simple


class coalescence(cycle_tools_simple.simple, object):

    def __init__(self):
        super(coalescence, self).__init__()
        self.cycle_tree = nx.Graph()
        self.counter_c = 0

    def calc_cycle_asymmetry(self, input_graph):

        minimum_basis = self.construct_networkx_basis(input_graph)
        self.calc_cycle_coalescence(input_graph, minimum_basis)
        tree_asymmetry = self.calc_tree_asymmetry()

        return tree_asymmetry

    def calc_cycle_coalescence(self, input_graph, cycle_basis):

        self.G = nx.Graph(input_graph)

        # create cycle_map_tree with cycles' edges as tree nodes
        for cycle in cycle_basis:

            attributes = {
                'label': 'base',
                'weight': 1.,
                'branch_type': 'none',
                'pos': (-1, -1)
            }

            self.cycle_tree.add_node(tuple(cycle.edges()), **attributes)

        # get the weights of the input graph and sort
        edges = nx.get_edge_attributes(self.G, 'weight')
        sorted_edges = sorted(edges, key=edges.__getitem__)

        # merge the cycles which share an edge
        for e in sorted_edges:

            # check whether all cycles are merged
            if len(cycle_basis) == 1:
                break

            cyc_w_edge = {}

            for i, cycle in enumerate(cycle_basis):
                if cycle.has_edge(*e):
                    cyc_w_edge.update({i: nx.number_of_edges(cycle)})

            if len(cyc_w_edge.values()) >= 2:

                idx_list = sorted(cyc_w_edge, key=cyc_w_edge.__getitem__)

                cycle_1 = cycle_basis[idx_list[0]]
                cycle_2 = cycle_basis[idx_list[1]]
                merged_cycle = self.merge_cycles(cycle_1, cycle_2)

                cycle_basis.remove(cycle_1)
                cycle_basis.remove(cycle_2)
                cycle_basis.append(merged_cycle)

                # build up the merging tree, set leave weights to nodes,
                # set asymetry value to binary branchings
                self.build_cycle_tree(cycle_1, cycle_2, merged_cycle)
                for n in self.cycle_tree.nodes():
                    if self.cycle_tree.nodes[n]['pos'][0] == -1:
                        self.cycle_tree.nodes[n]['pos'] = (self.counter_c, 0)
                        self.counter_c += 1

            else:
                continue

        return self.cycle_tree

    def calc_tree_asymmetry(self):

        dict_asymmetry = {}

        for n in self.cycle_tree.nodes():

            if self.cycle_tree.nodes[n]['branch_type'] == 'vanpelt_2':
                dict_asymmetry[n] = (self.cycle_tree.nodes[n]['asymmetry'])

        return dict_asymmetry

    def build_cycle_tree(self, cycle_1, cycle_2, merged_cycle):

        cyc_E_sets = [cycle_1.edges(), cycle_2.edges(), merged_cycle.edges()]
        cyc_key = [tuple(ces) for ces in cyc_E_sets]
        c_weight = np.zeros(2)

        # build merging tree
        for i in range(2):
            c_weight[i] = self.cycle_tree.nodes[cyc_key[i]]['weight']
            key = cyc_key[i]
            if self.cycle_tree.nodes[key]['label'] == 'base':
                self.cycle_tree.nodes[key]['pos'] = (self.counter_c, 0)
                self.counter_c += 1

        posX1 = self.cycle_tree.nodes[cyc_key[0]]['pos'][0]
        posX2 = self.cycle_tree.nodes[cyc_key[1]]['pos'][0]
        c_x = (posX1+posX2)/2.

        posY1 = self.cycle_tree.nodes[cyc_key[0]]['pos'][1]
        posY2 = self.cycle_tree.nodes[cyc_key[1]]['pos'][1]
        c_y = np.amax([posY1, posY2]) + 2.

        attributes = {
            'pos': (c_x, c_y),
            'label': 'merged',
            'weight': c_weight[0]+c_weight[1]
        }
        self.cycle_tree.add_node(cyc_key[2], **attributes)

        for i in range(2):
            self.cycle_tree.add_edge(cyc_key[i], cyc_key[2])

        # criterium for avoiding redundant branchings
        if c_y >= 6:
            self.cycle_tree.nodes[cyc_key[2]]['branch_type'] = 'vanpelt_2'
            A = (c_weight[0]-c_weight[1])/(c_weight[0]+c_weight[1]-2.)
            self.cycle_tree.nodes[cyc_key[2]]['asymmetry'] = np.absolute(A)
        else:
            self.cycle_tree.nodes[cyc_key[2]]['branch_type'] = 'none'

    def merge_cycles(self, cycle_1, cycle_2):

        cycles_edge_sets = [cycle_1.edges(), cycle_2.edges()]
        merged_cycle = nx.Graph()
        merged_cycle.graph['cycle_weight'] = 0
        for i in range(2):
            for e in cycles_edge_sets[i]:
                if merged_cycle.has_edge(*e):
                    merged_cycle.remove_edge(*e)
                else:
                    merged_cycle.add_edge(*e)

        for e in merged_cycle.edges():
            merged_cycle.graph['cycle_weight'] += self.G.edges[e]['weight']

        list_merged = list(merged_cycle.nodes())
        for n in list_merged:
            if merged_cycle.degree(n) == 0:
                merged_cycle.remove_node(n)

        return merged_cycle

    def compute_cycles_superlist(self, root):

        spanning_tree, dict_path = self.breadth_first_tree(root)
        diff_graph = nx.difference(self.G, spanning_tree)
        list_cycles = []
        for e in diff_graph.edges():

            simple_cycle, cycle_edges = self.find_cycle(dict_path, e, root)
            list_cycles.append(cycle_edges)

        return list_cycles

    def construct_networkx_basis(self, input_graph):

        C = self.construct_minimum_basis(input_graph)

        networkx_basis = []
        for cs in C:
            new_cycle = nx.Graph()
            new_cycle.graph['cycle_weight'] = 0.
            for e in cs:

                new_cycle.add_edge(*e)
                for k, v in self.G.edges[e].items():
                    new_cycle.edges[e][k] = v
                new_cycle.graph['cycle_weight'] += 1.

            for n in new_cycle.nodes():

                for k, v in self.G.nodes[n].items():
                    new_cycle.nodes[n][k] = v

            networkx_basis.append(new_cycle)

        return networkx_basis
