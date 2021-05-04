# @Author: Felix Kramer <kramer>
# @Date:   18-02-2019
# @Email:  felix.kramer@hotmail.de
# @Project: cycle-coalescecne-algorithm
# @Last modified by:   kramer
# @Last modified time: 04-05-2021

import networkx as nx
import numpy as np
import sys
import simple_cycle_tools

class cycle_coalescence(simple_cycle_tools,object):

    def __init__(self):
        super(cycle_coalescence,self).__init__()

    def calc_cycle_asymmetry(self,input_graph):

        minimum_basis=self.construct_minimum_basis(input_graph)
        cycle_tree=self.calc_cycle_coalescence(input_graph,minimum_basis)
        tree_asymmetry=self.calc_tree_asymmetry(cycle_tree)

        return tree_asymmetry

    def calc_cycle_coalescence(self,input_graph,cycle_basis):

        #create cycle_map_tree with cycles' edges as tree nodes

        cycle_tree=nx.Graph()
        for cycle in cycle_basis:
             cycle_tree.add_node(tuple(cycle.edges(keys=True)),label='base',weight=1.,branch_type='none',pos=(-1,-1))

        # get the weights of the input graph and sort
        edges=nx.get_edge_attributes(input_graph,'weight')
        sorted_edges=sorted(edges,key=edges.__getitem__)
        counter_c=0

        # merge the cycles which share an edge
        for e in sorted_edges:

            #check whether all cycles are merged
            if len(cycle_basis)== 1:
                break
            cycles_with_edge={}

            for i,cycle in enumerate(cycle_basis):
                if cycle.has_edge(*e):
                    cycles_with_edge.update({i:nx.number_of_edges(cycle)})

            if len(cycles_with_edge.values()) >= 2:

                idx_list=sorted(cycles_with_edge,key=cycles_with_edge.__getitem__)

                cycle_1=cycle_basis[idx_list[0]]
                cycle_2=cycle_basis[idx_list[1]]
                cycles_edge_sets=[cycle_1.edges(keys=True),cycle_2.edges(keys=True)]
                merged_cycle=self.merge_cycles(input_graph,cycle_1,cycle_2)

                # build merging tree
                if cycle_tree.nodes[tuple(cycles_edge_sets[0])]['label']=='base':
                    cycle_tree.nodes[tuple(cycles_edge_sets[0])]['pos']=(counter_c,0)
                    counter_c+=1
                if cycle_tree.nodes[tuple(cycles_edge_sets[1])]['label']=='base':
                    cycle_tree.nodes[tuple(cycles_edge_sets[1])]['pos']=(counter_c,0)
                    counter_c+=1

                cycle_basis.remove(cycle_1)
                cycle_basis.remove(cycle_2)
                cycle_basis.append(merged_cycle)

                # build up the merging tree, set leave weights to nodes, set asymetry value to binary branchings
                cycle_keys=[tuple(cycles_edge_sets[0]),tuple(cycles_edge_sets[1]),tuple(merged_cycle.edges(keys=True))]
                self.build_cycle_tree(cycle_tree,cycle_keys)
                for n in cycle_tree.nodes():
                    if cycle_tree.nodes[n]['pos'][0]==-1:
                        cycle_tree.nodes[n]['pos']=(counter_c,0)
                        counter_c+=1

            else:
                continue

        return cycle_tree

    def calc_tree_asymmetry(self,cycle_tree):

        dict_asymmetry={}

        for n in cycle_tree.nodes():

            if cycle_tree.nodes[n]['branch_type']=='vanpelt_2':
                dict_asymmetry[n]=(cycle_tree.nodes[n]['asymmetry'])

        return dict_asymmetry

    def build_cycle_tree(self,cycle_tree,cycle_keys):

        c_x=(cycle_tree.nodes[cycle_keys[0]]['pos'][0]+cycle_tree.nodes[cycle_keys[1]]['pos'][0])/2.
        c_y=np.amax([cycle_tree.nodes[cycle_keys[0]]['pos'][1],cycle_tree.nodes[cycle_keys[1]]['pos'][1]]) + 2.
        c1_weight=cycle_tree.nodes[cycle_keys[0]]['weight']
        c2_weight=cycle_tree.nodes[cycle_keys[1]]['weight']

        cycle_tree.add_node(cycle_keys[2],pos=(c_x,c_y),label='merged',weight=c1_weight+c2_weight)
        cycle_tree.add_edge(cycle_keys[0],cycle_keys[2])
        cycle_tree.add_edge(cycle_keys[1],cycle_keys[2])
        # criterium for avoiding redundant branchings
        if c_y>=6:
            cycle_tree.nodes[cycle_keys[2]]['branch_type']='vanpelt_2'
            cycle_tree.nodes[cycle_keys[2]]['asymmetry']=np.absolute((c1_weight-c2_weight))/(c1_weight+c2_weight-2.)
        else:
            cycle_tree.nodes[cycle_keys[2]]['branch_type']='none'

    def merge_cycles(self,input_graph,cycle_1,cycle_2):

        cycles_edge_sets=[cycle_1.edges(),cycle_2.edges()]
        merged_cycle=nx.MultiGraph()
        merged_cycle.graph['cycle_weight']=0
        for i in range(2):
            for e in cycles_edge_sets[i]:
                if merged_cycle.has_edge(*e):
                    merged_cycle.remove_edge(*e)
                else:
                    merged_cycle.add_edge(*e)

        for e in merged_cycle.edges():
            merged_cycle.graph['cycle_weight']+=input_graph.edges[e]['weight']

        list_merged=list(merged_cycle.nodes())
        for n in list_merged:
            if merged_cycle.degree(n)==0:
                merged_cycle.remove_node(n)

        return merged_cycle
