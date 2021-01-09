# cycle-coalescence-algorithm
This is a python implementation of the cycle coalescence algorithm as described by Modes et al, 2016.
https://journals.aps.org/prx/pdf/10.1103/PhysRevX.6.031009

The implementation is largely based on networkx and numpy, and enables to build merging trees for any simple undirected graph. Further a routine for analyzing tree asymmetry according to Van Pelt et al, 1992.
https://link.springer.com/article/10.1007/BF02459929

The repository does inlcude a jupyter notebook for testing predefined edge weight distributions on square lattices.
##  Instalation
##  Usage

'''python
import networkx as nx
# generate a dummy graph for testing
G=nx.grid_graph((7,7,1))

# put an edge weight distribution on the system, available are random/gradient/bigradient/nested_square

G=ag.generate_pattern(G,'nested_square')
weights=[G.edges[e]['weight'] for e in G.edges()]
pos=nx.get_node_attributes(G,'pos')
nx.draw_networkx(G,pos=pos,width=weights,with_labels=False,node_size=50,alpha=0.2)
# merge all shortest cycles and calc the merging tree's asymmetry
asymmetry=ag.calc_cycle_asymmetry(G)
print(asymmetry)
'''

##  Requirements
##  Gallery


