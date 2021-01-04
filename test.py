a=1
b=1
print(a)
def change_me(x):
  x+=1

change_me(a)
print(a)

import networkx as nx
G=nx.Graph()
print(nx.info(G))
def change_graph(G):
    G.add_node(0)
change_graph(G)
print(nx.info(G))

dict_e={}
print(dict_e)
def change_dict(d):
    d[0]=1
change_dict(dict_e)
print(dict_e)

dict_e[(0,1)]=0
dict_e[(1,0)]
