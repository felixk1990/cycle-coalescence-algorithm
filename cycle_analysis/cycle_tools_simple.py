# @Author: Felix Kramer <kramer>
# @Date:   04-05-2021
# @Email:  kramer@mpi-cbg.de
# @Project: phd_network_remodelling
# @Last modified by:   kramer
# @Last modified time: 04-05-2021



import networkx as nx
import numpy as np
import sys
import cycle_tools

class simple(cycle_tools.toolbox,object):

    def __init__(self):
        super(simple,self).__init__()

    def generate_cycle_lists(self):

        total_cycle_dict={}
        total_cycle_list={}

        self.G=nx.convert_node_labels_to_integers(self.G, first_label=0, ordering='default')
        nx.set_node_attributes(self.G,False,'push')

        # check for graph_type, then check for paralles in the Graph, if existent insert dummy nodes to resolve conflict, cast the network onto simple graph afterwards
        for i,e in enumerate(self.G.edges()):
            self.G.edges[e]['label']=i
        counter=0
        root_sets=[]
        for n in self.G.nodes():
            # building new tree using breadth first
            root_sets.append(self.compute_cycles_superlist(n))

        for list_cycles in root_sets:
            for cycle_edges in list_cycles:
                # relabeling and weighting graph
                total_cycle_list.update({counter:cycle_edges})
                total_cycle_dict.update({counter:[self.G.edges[f]['label'] for f in cycle_edges]})
                counter+=1

        return total_cycle_dict,total_cycle_list

    def find_cycle(self,*args):

        dict_path,e,n=args
        # label pathways

        l1=dict_path[e[1]][::-1]
        l2=dict_path[e[0]][::-1]
        if len(dict_path[e[0]])< len(dict_path[e[1]]):
            l1=dict_path[e[0]][::-1]
            l2=dict_path[e[1]][::-1]

        idx1=0
        idx2=0
        for i,n in enumerate(l1):
            if n in l2:
                idx1=i
                idx2=l2.index(n)
                break
        L2=l2[:idx2]

        new_path=l1[:idx1+1]+L2[::-1]
        new_edges = [(new_path[i],new_path[i+1]) for i,n in enumerate(new_path[:-1]) ]
        new_edges+=[e]

        return new_path, new_edges

    def compute_cycles_superlist(self,root):

        spanning_tree,dict_path=self.breadth_first_tree(root)
        diff_graph=nx.difference(self.G,spanning_tree)
        list_cycles=[]
        for e in diff_graph.edges():

            simple_cycle,cycle_edges=self.find_cycle(dict_path,e,root)
            list_cycles.append(cycle_edges)

        return list_cycles

    def construct_minimum_basis(self,input_graph):

        # calc minimum weight basis and construct dictionary for weights of edges, takes a leave-less, connected, N > 1 SimpleGraph as input, no self-loops optimally, deviations are not raising any warnings
        #sort basis vectors according to weight, creating a new minimum weight basis from the total_cycle_list
        self.G=nx.Graph(input_graph)
        P=nx.number_connected_components(self.G)
        nullity=nx.number_of_edges(self.G)-nx.number_of_nodes(self.G)+P

        total_cycle_dict,total_cycle_list=self.generate_cycle_lists()
        total_cycle_len={}
        for c,e in total_cycle_dict.items():
            total_cycle_len[c]=len(e)
        sorted_cycle_list=sorted(total_cycle_len,key=total_cycle_len.__getitem__)
        # print(len(total_cycle_list))
        minimum_basis=[]
        minimum_label=[]
        EC=nx.Graph()
        counter=0

        for c in sorted_cycle_list:

            cycle_edges_in_basis=True
            new_cycle=total_cycle_list[c]

            for e in new_cycle:
                if not EC.has_edge(*e):
                    EC.add_edge(*e,label=counter)
                    counter+=1
                    cycle_edges_in_basis=False

            #if cycle edges where not part of the supergraph yet then it becomes automatically part of the basis
            if not cycle_edges_in_basis:

                minimum_basis.append(new_cycle)
                aux_label=[EC.edges[e]['label'] for e in new_cycle ]
                minimum_label.append(aux_label)

            #if cycle edges are already included we check for linear dependece
            else:
                E=self.edge_matrix(EC, minimum_basis, minimum_label ,new_cycle)

                linear_independent=self.compute_linear_independence(E)
                # print(linear_independent)
                if linear_independent:
                    minimum_basis.append(new_cycle)
                    aux_label=[EC.edges[e]['label'] for e in new_cycle ]
                    minimum_label.append(aux_label)

            if len(minimum_basis)==nullity:
                break

        if len(minimum_basis)<nullity:
            sys.exit('Error: Cycle basis badly constructed')

        return minimum_basis

    def construct_networkx_basis(self,input_graph):

        C=self.construct_minimum_basis(input_graph)

        networkx_basis=[]
        for cs in C:
            new_cycle=nx.Graph()
            for e in cs:

                new_cycle.add_edge(*e)
                for k,v in self.G.edges[e].items():
                    new_cycle.edges[e][k]=v

            for n in new_cycle.nodes():

                for k,v in self.G.nodes[n].items():
                    new_cycle.nodes[n][k]=v

            networkx_basis.append(new_cycle)

        return networkx_basis

    def edge_matrix(self,*args):

        EC, minimum_basis, minimum_label,new_cycle=args
        rows=len(EC.edges())
        columns=len(minimum_basis)+1
        E=np.zeros((rows,columns))

        for idx_c,cycle in enumerate(minimum_basis):
            E[minimum_label[idx_c],idx_c]=1

        for m in new_cycle:
            E[EC.edges[m]['label'],-1]=1

        return E

    def compute_linear_independence(self,E):

        linear_independent=False
        rows=len(E[:,0])
        columns=len(E[0,:])

        # calc echelon form
        a_columns=np.arange(columns-1)
        for column in a_columns:
            idx_nz=np.nonzero(E[column:,column])[0]
            idx=idx_nz[0]+column

            if len(idx_nz)==1
                E[[column,idx_nz[0]+column],:]=E[[idx_nz[0]+column,column],:]

            else:

                new_idx=idx_nz[1:]+colum
                aux_E=np.add(E[new_idx],E[idx])
                E[new_idx]=np.mod(aux_E,2)
                E[[column,idx_nz[0]+column],:]=E[[idx_nz[0]+column,column],:]

        r=np.nonzero(E[columns-1:,-1])[0]
        if r.size:
            linear_independent=True

        return linear_independent

    def breadth_first_tree(self,root):

        T=nx.Graph()
        push_down=nx.get_node_attributes(self.G,'push')
        len_n=len(self.G.nodes())

        if len(push_down.keys())!=len_n:
            push_down={}
            for n in self.G.nodes():
                push_down[n]=False

        push_down[root]=True
        root_queue=[]

        labels=self.G.edges(root)
        dict_path={}
        dict_path[root]=[root]
        T,dict_path,root_queue=self.compute_sprouts(root,T,labels ,push_down,dict_path,root_queue)

        while T.number_of_nodes() < len_n:
            new_queue=[]
            for q in root_queue:

                labels=self.G.edges(q)
                T,dict_path,new_queue=self.compute_sprouts(q,T,labels,push_down,dict_path,new_queue)

            root_queue=new_queue[:]

        return T,dict_path

    def compute_sprouts(self,*args):

        root,T,labels,push_down,dict_path,root_queue=args
        for e in labels:

            if e[0]==root:
                if not push_down[e[1]]:
                    T.add_edge(*e)
                    root_queue.append(e[1])
                    push_down[e[1]]=True
                    dict_path[e[1]]=dict_path[root]+[e[1]]
            else:
                if not push_down[e[0]]:
                    T.add_edge(*e)
                    root_queue.append(e[0])
                    push_down[e[0]]=True
                    dict_path[e[0]]=dict_path[root]+[e[0]]

        return T,dict_path,root_queue
