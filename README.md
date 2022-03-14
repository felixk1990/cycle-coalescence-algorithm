# cycle-coalescence-algorithm

Have you ever wondered how cycles in graphs form a vector space and encapsulate nesting information? Here is a tool ready to use, enabling you to calculate the cycle bases, mapping them onto a merging tree, and analyze this tree's asymmetry.

##  Introduction
This python module allows users to analyze weighted, undirected simple graphs for their nested cycle structure by performing two major functions: Calculating minimal cycle bases (Horton algorithm) and computing the merging tree (cycle coalescence algorithm). The algorithm is described in "Modes et al,'Extracting Hidden Hierarchies in 3D Distribution Networks', 2016" and basically follows the shown scheme below:
  -  All fundamentals minimal cyles (minimal number of edges) are listed in the weighted graph G and mapped onto the leaves of a new tree T.
  -  Then one identifies the lightest edge e in G and merges the two smallest cycles along this edge, creating a new vertex in the tree T for the merger cycle
  -  remove the original two cycles and proceed with the next lightest edge e until all cycles in G are merged
  -  finally calculate the tree asymmetry using the techniques of "Van-Pelt et al, 'Tree Asymmetry—A Sensitive and Practical Measure for Binary Topological Trees' ,1992"
  -  the asymmetry orderparameter will be be 1 for perfecly asymmetric trees and 0 for perfectly symmetric trees
  ![modes](https://raw.githubusercontent.com/felixk1990/cycle-coalescence-algorithm/main/gallery/modes_merging_algorithm_2016.png)
  Figure taken from: Modes et al,'Extracting Hidden Hierarchies in 3D Distribution Networks', 2016


##  Installation
pip install cycle-analysis

##  Usage
Currently this implementation only supports networkx graphs.
Call cycle_analysis.cycle_coalescence for graph analysis, while cycle_analysis.test provides you with pre-customized functions to put specific weight patterns onto the graph: random/gradient/nested_square
```python
import networkx as nx
import matplotlib.pyplot as plt
from cycle_analysis.cycle_tools_coalescence import Coalescence
from cycle_analysis.cycle_custom_pattern import generate_pattern
from cycle_analysis.cycle_tools_simple import construct_networkx_basis

# generate a dummy graph for testing
# put an edge weight distribution on the system, available are random/gradient/bigradient/nested_square
unweightedG = nx.grid_graph((5, 5, 1))
weightedG = generate_pattern(unweightedG, 'nested_square')

fig,axs = plt.subplots(2, 1, figsize=(10,10))
weights = [weightedG.edges[e]['weight'] for e in weightedG.edges()]
pos = nx.get_node_attributes(weightedG, 'pos')
nx.draw_networkx(weightedG, pos=pos, width=weights, with_labels=False, node_size=50, ax=axs[0] )

# merge all shortest cycles and create merging tree
T = Coalescence()
minimum_basis = construct_networkx_basis(weightedG)
cycle_tree = T.calc_cycle_coalescence(weightedG, minimum_basis)

pos=nx.get_node_attributes(cycle_tree, 'pos')
nx.draw_networkx(cycle_tree, pos=pos, with_labels=False, node_size=50, ax=axs[1])




# plot branching asymmetry in dependence of branching level
dict_asymmetry = T.calc_tree_asymmetry(cycle_tree)
x = [(cycle_tree.nodes[n]['pos'][1]-6)/2. for n in dict_asymmetry]
y = [dict_asymmetry[n] for n in dict_asymmetry]

plt.scatter(x,y)
plt.ylabel('asymmetry')
plt.xlabel('branching level')
plt.grid(True)
plt.show()
```
./notebook contains examples to play with in the form of jupyter notebooks. 
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/felixk1990/cycle-coalescence-algorithm/198727ddd80524cd7197f01e46cc74c33175b6f0?labpath=.%2Fnotebook)
##  Requirements
```python
networkx==2.5
numpy==1.19.1
matplotlib==3.4.3
```
##  Gallery
random weight distribution\
![random](https://raw.githubusercontent.com/felixk1990/cycle-coalescence-algorithm/main/gallery/random.png)

nested square weight distribution\
![nested](https://raw.githubusercontent.com/felixk1990/cycle-coalescence-algorithm/main/gallery/nested_square.png)

gradient weight distribution\
![gradient](https://raw.githubusercontent.com/felixk1990/cycle-coalescence-algorithm/main/gallery/gradient.png)
## Acknowledgement
```cycle-analysis``` written by Felix Kramer

This implementation is based on the cycle coalescence algorithm as described by [Modes et al, 2016](https://journals.aps.org/prx/pdf/10.1103/PhysRevX.6.031009). Please acknowledge if used for any further publication or projects.
