# @Author: Felix Kramer <kramer>
# @Date:   18-02-2019
# @Email:  felix.kramer@hotmail.de
# @Project:  cycle_analysis
# @Last modified by:    Felix Kramer
# @Last modified time: 2021-11-04T22:12:19+01:00
# @License: MIT
import networkx as nx
from dataclasses import dataclass, field
from .cycle_tools_simple import *

@dataclass
class Coalescence():

    """
    This is a class of caolescence algorithms, for the analysis of mesh-like
    spatial networks.

    Returns:
        Coalescence: A custom Coalescence instance object.

    """

    counter_c: int = field(default=0, repr=False)
    cycle_tree: nx.Graph = field(default_factory=nx.Graph, repr=False)

    def calc_cycle_asymmetry(self, input_graph):

        """
        Given a Graph with a cyclic backbone, calculate the minimal
        (topological) cycle basis

        Args:
            input_graph (nx.Graph): A networkx graph with 'many' cycles

        Returns:
            dictionary: A dictionary vertices with respective asymmetries

        """

        minimum_basis = construct_networkx_basis(input_graph)
        for mb in minimum_basis:
            mb.graph['cycle_weight'] = nx.number_of_edges(mb)

        cycle_tree = self.calc_cycle_coalescence(input_graph, minimum_basis)
        tree_asymmetry = self.calc_tree_asymmetry(cycle_tree)

        return tree_asymmetry

    def calc_cycle_coalescence(self, input_graph, cycle_basis):

        """
        Builds the merging tree according to the cycle coalescence algorithm.

        Args:
            input_graph (nx.Graph): A networkx graph with 'many' cycles
            cycle_basis (list): List of networkx cycles

        Returns:
            nx.Graph: Weighted Merging Tree

        """

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
                new_cycle = self.merge_cycles(cycle_1, cycle_2)

                for e in new_cycle.edges():
                    new_cycle.graph['cycle_weight'] += self.G.edges[e]['weight']

                cycle_basis.remove(cycle_1)
                cycle_basis.remove(cycle_2)
                cycle_basis.append(new_cycle)

                # build up the merging tree, set leave weights to nodes,
                # set asymetry value to binary branchings
                self.build_cycle_tree(cycle_1, cycle_2, new_cycle)
                for n in self.cycle_tree.nodes():
                    if self.cycle_tree.nodes[n]['pos'][0] == -1:
                        self.cycle_tree.nodes[n]['pos'] = (self.counter_c, 0)
                        self.counter_c += 1

            else:
                continue

        return self.cycle_tree

    def calc_tree_asymmetry(self, cycle_tree):

        """
        Computes  binary networkx tree and calculates its asymmetry.

        Args:
            cycle_tree (nx.Graph): A networkx graph with weighted vertices

        Returns:
            dictionary: A dictionary which holds the asymetry value for any vertex in the graph

        """

        dict_asymmetry = {}

        for n in cycle_tree.nodes():

            if cycle_tree.nodes[n]['branch_type'] == 'vanpelt_2':
                dict_asymmetry[n] = (cycle_tree.nodes[n]['asymmetry'])

        return dict_asymmetry

    def build_cycle_tree(self, cycle_1, cycle_2, merged_cycle):

        """
        Systematically build a merger tree and save the respective asymetry
        values and spatial layout

        Args:
            cycle_1 (nx.Graph): A networkx graph, cycle
            cycle_2 (nx.Graph): A networkx graph, cycle
            merged_cycle (nx.Graph): A networkx graph, cycle

        """

        cyc_E_sets = [cycle_1.edges(), cycle_2.edges(), merged_cycle.edges()]
        cyc_key = [tuple(ces) for ces in cyc_E_sets]
        c_weight = [0., 0.]

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
        c_y = max([posY1, posY2]) + 2.

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
            self.cycle_tree.nodes[cyc_key[2]]['asymmetry'] = abs(A)
        else:
            self.cycle_tree.nodes[cyc_key[2]]['branch_type'] = 'none'

    def merge_cycles(self, cycle_1, cycle_2):

        """
        Merge two graph cycles according to common cycle arithmetics in Z2 and return the merged cycle.

        Args:
            cycle_1 (nx.Graph): A networkx graph, cycle
            cycle_2 (nx.Graph): A networkx graph, cycle

        Returns:
            nx.Graph: A networkx graph, cycle


        """

        cycles_edge_sets = [cycle_1.edges(), cycle_2.edges()]
        merged_cycle = nx.Graph()
        merged_cycle.graph['cycle_weight'] = 0
        for i in range(2):
            for e in cycles_edge_sets[i]:
                if merged_cycle.has_edge(*e):
                    merged_cycle.remove_edge(*e)
                else:
                    merged_cycle.add_edge(*e)

        list_merged = list(merged_cycle.nodes())
        for n in list_merged:
            if merged_cycle.degree(n) == 0:
                merged_cycle.remove_node(n)

        return merged_cycle
