# cycle-coalescence-algorithm
##  Introduction
Hello everyone,

I wrote this package during my PhD, when working on the characterization of transport networks.

Have you ever wondered how cycles in graphs form a vector space and encapsulate nesting information? If so, were you never really sure how to deal with this? Here is a tool ready to use, enabling you to calculate the cycle bases, mapping them onto a merging tree, and analyze this tree's asymmetry.

![modes](./gallery/modes_merging_algorithm_2016.png)

This project is based on the algorithm published in 'Extracting Hidden Hierarchies in 3D Distribution Networks' by Modes et al, 2016. Please acknowledge if used for any further publication or projects.

  ./notebook contains examples to play with in the form of jupyter notebooks
##  Installation
python3 -m pip install --index-url https://test.pypi.org/simple/ --no-deps cycle_analysis
##  Usage

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

##  Requirements
``` networkx ```, ``` numpy ```
##  Gallery
random weight distribution\
![random](./gallery/random.png)

nested square weight distribution\
![nested](./gallery/nested_square.png)

gradient weight distribution\
![gradient](./gallery/gradient.png)
## Acknowledgement
```cycle_analysis``` written by Felix Kramer

This implementation is based on the cycle coalescence algorithm as described by Modes et al, 2016.
https://journals.aps.org/prx/pdf/10.1103/PhysRevX.6.031009
