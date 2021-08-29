# cycle-coalescence-algorithm

Have you ever wondered how cycles in graphs form a vector space and encapsulate nesting information? Here is a tool ready to use, enabling you to calculate the cycle bases, mapping them onto a merging tree, and analyze this tree's asymmetry.

##  Introduction
This python module allows users to analyze weighted, undirected simple graphs for their nested cycle structure by performing two major functions: Calculating minimal cycle bases (Horton algorithm) and computing the merging tree (cycle coalescence algorithm). The algorithm is described in "Modes et al,'Extracting Hidden Hierarchies in 3D Distribution Networks', 2016" and basically follows the shown scheme below:
  -  All fundamentals minimal cyles (minimal number of edges) are listed in the weighted graph G and mapped onto the leaves of a new tree T. 
  -  Then one identifies the lightest edge e in G and merges the two smallest cycles along this edge, creating a new vertex in the tree T for the merger cycle
  -  remove the original two cycles and proceed with the next lightest edge e until all cycles in G are merged
  -  finally calculate the tree asymmetry using the techniques of "Van-Pelt et al, 'Tree Asymmetry—A Sensitive and Practical Measure for Binary Topological Trees' ,1992"
  -  the asymmetry orderparameter will be be 1 for perfecly asymmetric trees and 0 for perfectly symmetric trees
  ![modes](./gallery/modes_merging_algorithm_2016.png)
  Figure taken from: Modes et al,'Extracting Hidden Hierarchies in 3D Distribution Networks', 2016


##  Installation
pip install cycle_analysis

##  Usage
Currently this implementation only supports networkx graphs.
Call cycle_analysis.cycle_coalescence for graph analysis, while cycle_analysis.test provides you with pre-customized functions to put specific weight patterns onto the graph: random/gradient/nested_square
```python
import networkx as nx
import cycle_analysis.cycle_coalescence as cc
import cycle_analysis.test as cat

# generate a dummy graph for testing
# put an edge weight distribution on the system, available are random/gradient/nested_square
G=nx.grid_graph((7,7,1))
G=cat.generate_pattern(G,'nested_square')

# merge all shortest cycles and calc the merging tree's asymmetry for each branch
asymmetry=cc.calc_cycle_asymmetry(G)
print(asymmetry)
```
./notebook contains examples to play with in the form of jupyter notebooks
##  Requirements
``` python3.6+ ```,``` networkx ```, ``` numpy ```
##  Gallery
random weight distribution\
![random](./gallery/random.png)

nested square weight distribution\
![nested](./gallery/nested_square.png)

gradient weight distribution\
![gradient](./gallery/gradient.png)
## Acknowledgement
```cycle_analysis``` written by Felix Kramer

This implementation is based on the cycle coalescence algorithm as described by [Modes et al, 2016](https://journals.aps.org/prx/pdf/10.1103/PhysRevX.6.031009). Please acknowledge if used for any further publication or projects.
