# @Author: Felix Kramer <kramer>
# @Date:   18-02-2019
# @Email:  felix.kramer@hotmail.de
# @Project: cycle-coalescecne-algorithm
# @Last modified by:   kramer
# @Last modified time: 04-05-2021

import networkx as nx
import numpy as np
import sys
import cycle_tools_simple

class coalescence(cycle_tools_simple.simple,object):

    def __init__(self):
        super(coalescence,self).__init__()

    def calc_cycle_asymmetry(input_graph):

        minimum_basis=self.construct_minimum_basis(input_graph)
        cycle_tree=self.calc_cycle_coalescence(input_graph,minimum_basis)
        tree_asymmetry=self.calc_tree_asymmetry(cycle_tree)

        return tree_asymmetry

    def calc_cycle_coalescence(self,input_graph,cycle_basis):

        #create cycle_map_tree with cycles' edges as tree nodes
        # print([len(cb.edges()) for cb in cycle_basis])
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

    def generate_cycle_lists(self,input_graph):

        total_cycle_dict={}
        total_cycle_list=[]
        # check for graph_type, then check for paralles in the Graph, if existent insert dummy nodes to resolve conflict, cast the network onto simple graph afterwards

        counter=0
        for n in input_graph.nodes():
            # building new tree using breadth first

            # spanning_tree=nx.minimum_spanning_tree(graph,weight='weight')
            spanning_tree=self.breadth_first_tree(input_graph,n)
            diff_graph=nx.difference(input_graph,spanning_tree)
            labels_e={}

            for e in diff_graph.edges():
                p_in=nx.shortest_path(spanning_tree,source=n,target=e[0])
                p_out=nx.shortest_path(spanning_tree,source=n,target=e[1])
                # label pathways
                simple_cycle=nx.MultiGraph(cycle_weight=0.)
                nx.add_path(simple_cycle,p_in)
                nx.add_path(simple_cycle,p_out)
                simple_cycle.add_edge(*e)

                list_n=list(simple_cycle.nodes())
                seen={}
                for m in list(simple_cycle.edges()):
                    num_conncetions=simple_cycle.number_of_edges(*m)
                    if num_conncetions > 1 and m not in seen.keys():
                        seen[m]=1
                    elif num_conncetions > 1:
                        seen[m]+=1
                for m in seen:
                    for i in range(seen[m]):
                        simple_cycle.remove_edge(m[0],m[1],i)
                for q in list_n:
                    if simple_cycle.degree(q)==0:
                        simple_cycle.remove_node(q)

                if nx.is_eulerian(simple_cycle):
                    # relabeling and weighting graph
                    for m in simple_cycle.edges():
                        simple_cycle.graph['cycle_weight']+=1.
                    total_cycle_list.append(simple_cycle)
                    total_cycle_dict.update({counter:nx.number_of_edges(simple_cycle)})
                    counter+=1

        return total_cycle_dict,total_cycle_list

    def construct_minimum_basis(self,input_graph):
        # calc minimum weight basis and construct dictionary for weights of edges, takes a leave-less, connected, N > 1 SimpleGraph as input, no self-loops optimally, deviations are not raising any warnings
        #sort basis vectors according to weight, creating a new minimum weight basis from the total_cycle_list
        nullity=nx.number_of_edges(input_graph)-nx.number_of_nodes(input_graph)+nx.number_connected_components(input_graph)

        total_cycle_dict,total_cycle_list=generate_cycle_lists(input_graph)
        sorted_cycle_list=sorted(total_cycle_dict,key=total_cycle_dict.__getitem__)
        minimum_basis=[]
        EC=nx.MultiGraph()
        counter=0
        total_cycle_list_sort=[total_cycle_list[i] for i in sorted_cycle_list]
        # print([len(tcl.edges()) for tcl in total_cycle_list_sort ])
        for c in sorted_cycle_list:

            cycle_edges_in_basis=True

            for e in total_cycle_list[c].edges(keys=True):
                if not EC.has_edge(*e):
                    EC.add_edge(*e,label=counter)
                    counter+=1
                    cycle_edges_in_basis=False
            #if cycle edges where not part of the supergraph yet then it becomes automatically part of the basis
            if not cycle_edges_in_basis:
                minimum_basis.append(total_cycle_list[c])
            #if cycle edges are already included we check for linear dependece
            else:
                linear_independent=False
                rows=len(list(EC.edges()))
                columns=len(minimum_basis)+1
                E=np.zeros((rows,columns))
                # translate the existent basis vectors into z2 representation
                for idx_c,cycle in enumerate(minimum_basis+[total_cycle_list[c]]):
                    for m in cycle.edges(keys=True):
                        if EC.has_edge(*m):
                            E[EC.edges[m]['label'],idx_c]=1

                # calc echelon form
                a_columns=np.arange(columns-1)
                zwo=np.ones(columns)*2
                for column in a_columns:
                    idx_nz=np.nonzero(E[column:,column])[0]
                    if idx_nz.size:
                        if len(idx_nz)==1:
                            E[column,:],E[idx_nz[0]+column,:]=E[idx_nz[0]+column,:].copy(),E[column,:].copy()
                        else:
                            for r in idx_nz[1:]:
                                aux_E=np.add(E[r+column],E[idx_nz[0]+column])
                                E[r+column]=np.mod(aux_E,zwo)
                            E[column,:],E[idx_nz[0]+column,:]=E[idx_nz[0]+column,:].copy(),E[column,:].copy()
                    else:
                        sys.exit('Error: minimum_weight_basis containing inconsistencies ...')
                # test echelon form for inconsistencies
                for r in range(rows):
                    line_check=np.nonzero(E[r])[0]
                    if len(line_check)==1 and line_check[0]==(columns-1):
                        linear_independent=True
                        break
                if linear_independent:
                    minimum_basis.append(total_cycle_list[c])

            if len(minimum_basis)==nullity:
                break

        if len(minimum_basis)<nullity:
            sys.exit('Error: Cycle basis badly constructed')

        return minimum_basis

    def breadth_first_tree(self,input_graph,root):

        T=nx.Graph()
        push_down={}
        for i,e in enumerate(input_graph.edges()):
            input_graph.edges[e]['label']=i
        for n in input_graph.nodes():
            push_down[n]=False

        push_down[root]=True
        root_queue=[]
        labels_e = input_graph.edges(root,'label')
        dict_labels_e={}
        for le in labels_e:
            dict_labels_e[(le[0],le[1])]=le[2]
        sorted_label_e_list=sorted(dict_labels_e,key=dict_labels_e.__getitem__)

        for e in sorted_label_e_list:

            if e[0]==root:
                if not push_down[e[1]]:
                    T.add_edge(*e)
                    root_queue.append(e[1])
                    push_down[e[1]]=True

            else:
                if not push_down[e[0]]:
                    T.add_edge(*e)
                    root_queue.append(e[1])
                    push_down[e[0]]=True

        while T.number_of_nodes() < input_graph.number_of_nodes():
            new_queue=[]
            for q in root_queue:
                labels_e = input_graph.edges(q,'label')
                dict_labels_e={}
                for le in labels_e:
                    dict_labels_e[(le[0],le[1])]=le[2]
                sorted_label_e_list=sorted(dict_labels_e,key=dict_labels_e.__getitem__)

                for e in sorted_label_e_list:

                    if e[0]==q:
                        if not push_down[e[1]]:
                            T.add_edge(*e)
                            new_queue.append(e[1])
                            push_down[e[1]]=True

                    else:
                        if not push_down[e[0]]:
                            T.add_edge(*e)
                            new_queue.append(e[1])
                            push_down[e[0]]=True
            root_queue=new_queue[:]

        return T

    def path_list(self,input_graph,root):

        leaves=[]
        paths={}

        for n in input_graph.nodes():
            if input_graph.degree(n) == 1:
                leaves.append(n)
        for n in leaves:
            p=nx.shortest_path(nx.Graph(input_graph),source=root,target=n)
            paths[tuple(p)]=len(p)

        return paths
