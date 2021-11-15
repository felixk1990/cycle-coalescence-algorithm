# @Author: Felix Kramer <kramer>
# @Date:   04-05-2021
# @Email:  kramer@mpi-cbg.de
# @Project:  cycle_analysis
# @Last modified by:    Felix Kramer
# @Last modified time: 2021-11-04T22:31:34+01:00
import networkx as nx
import numpy as np

class simple_cycles(object):

    """
    A class to generate cycle bases and create a minimal basis in
    networkx formats

    Attributes
    ----------
    G : nx.Graph()
        A simple graph which on which the cycle basis is calcuclated
    """

    def __init__(self):

        self.G = None

    def generate_cycle_lists(self):

        """
        Returns an edge list, and labeled dictionary of cycles drawn from
        a Horton cycle search for all vertices.

        Returns:
            dictionary: A dictionary which of cycles generated from bfs searches
            list: a list of cycles represented by their edge sets

        Raises
        -------
        NotImplementedError
            If no graph is initially set for the backbone Graph G.
        """

        if self.G is None:
            raise RuntimeError("cycle_tools_simple.simple_cycles.G is not set!")

        nx.set_node_attributes(self.G, False, 'push')

        # check for graph_type, then check for paralles in the Graph,
        # if existent insert dummy nodes to resolve conflict,
        # cast the network onto simple graph afterwards
        for i, e in enumerate(self.G.edges()):
            self.G.edges[e]['label'] = i

        root_sets = []
        for n in self.G.nodes():
            # building new tree using breadth first
            root_sets.append(self.compute_cycles_superlist(n))

        key = 0
        cyc_dict = {}
        cyc_list = {}
        for cyc_sets in root_sets:
            for cyc_E in cyc_sets:
                # relabeling and weighting graph
                cyc_list.update({key: cyc_E})

                labels = [self.G.edges[f]['label'] for f in cyc_E]
                cyc_dict.update({key: labels})

                key += 1

        return cyc_dict, cyc_list

    def find_cycle(self, dict_path, e, n):
        """
        Returns an edge list, and node list for a cycle constructed from
        spanning tree + additional edge.

        Args:
            dict_path (dictionary): A dictionary of shortest paths in the bfs tree
            e (tuple): The edge which is to be plugge into the bfs tree and generates a cycle
            n (int): The root of the current bfs tree
        Returns:
            list: A list of vertices for the new cycle
            list: A list of edges for the new cycle

        Raises:
            Exception: description
        """

        # label pathways
        l1 = dict_path[e[1]][::-1]
        l2 = dict_path[e[0]][::-1]
        if len(dict_path[e[0]]) < len(dict_path[e[1]]):
            l1 = dict_path[e[0]][::-1]
            l2 = dict_path[e[1]][::-1]

        idx1 = 0
        idx2 = 0
        for i, n in enumerate(l1):
            if n in l2:
                idx1 = i
                idx2 = l2.index(n)
                break
        L2 = l2[:idx2]

        new_path = l1[:idx1+1]+L2[::-1]
        new_edges = [(p, new_path[i+1]) for i, p in enumerate(new_path[:-1])]
        new_edges += [e]

        return new_path, new_edges

    def compute_cycles_superlist(self, root):
        """
        Returns an edge list of cycles drawn from a Horton cycle search for
        one vertex.

        Args:
            root (int): The root vertex of the current bfs tree

        Returns:
            list: The superlist of cycles from all bfs trees, in edge list representation

        Raises:
            Exception: description

        """

        spanning_tree, dict_path = self.breadth_first_tree(root)
        diff_graph = nx.difference(self.G, spanning_tree)
        list_cycles = []
        for e in diff_graph.edges():

            simple_cycle, cycle_edges = self.find_cycle(dict_path, e, root)
            list_cycles.append(cycle_edges)

        return list_cycles

    def construct_networkx_basis(self, input_graph):
        """
        Return a cycle basis for the input graph, with all elements
        edge lists.

        Args:
            input_graph (nx.Graph): A networkx graph with 'many' cycles

        Returns:
            list: The minimal basis of the graph, represented by a list of networkx graphs.

        Raises:
            Exception: description

        """

        C = self.construct_minimum_basis(input_graph)

        networkx_basis = []
        for cs in C:
            new_cycle = nx.Graph()
            for e in cs:

                new_cycle.add_edge(*e)
                for k, v in self.G.edges[e].items():
                    new_cycle.edges[e][k] = v

            for n in new_cycle.nodes():

                for k, v in self.G.nodes[n].items():
                    new_cycle.nodes[n][k] = v

            networkx_basis.append(new_cycle)

        return networkx_basis

    def construct_minimum_basis(self, input_graph):

        """
        Return a cycle basis for the input graph, with all elements
        edge lists.

        Args:
            input_graph (nx.Graph): A networkx graph

        Returns:
            list: The minimal basis of the graph, represented by a list of edge lists.

        Raises:
            Exception: description

        """

        # calc minimum weight basis and construct dictionary for weights of
        # edges, takes a leave-less, connected, N > 1 SimpleGraph as input,
        # no self-loops optimally, deviations are not raising any warnings
        # sort basis vectors according to weight, creating a new minimum weight
        # basis from the total_cycle_list
        self.G = nx.Graph(input_graph)
        P = nx.number_connected_components(self.G)
        nullity = nx.number_of_edges(self.G)-nx.number_of_nodes(self.G)+P

        cyc_dict, cyc_list = self.generate_cycle_lists()
        cyc_len = {}
        for c, e in cyc_dict.items():
            cyc_len[c] = len(e)
        sorted_cycle_list = sorted(cyc_len, key=cyc_len.__getitem__)

        min_basis = []
        min_label = []
        EC = nx.Graph()
        counter = 0

        for c in sorted_cycle_list:

            cycle_edges_in_basis = True
            new_cycle = cyc_list[c]

            for e in new_cycle:
                if not EC.has_edge(*e):
                    EC.add_edge(*e, label=counter)
                    counter += 1
                    cycle_edges_in_basis = False

            # if cycle edges where not part of the supergraph yet then it
            # becomes automatically part of the basis
            if not cycle_edges_in_basis:

                min_basis.append(new_cycle)
                aux_label = [EC.edges[e]['label'] for e in new_cycle]
                min_label.append(aux_label)

            # if cycle edges are already included we check for linear dependece
            else:
                E = self.edge_matrix(EC, min_label, new_cycle)

                linear_independent = self.compute_linear_independence(E)

                if linear_independent:
                    min_basis.append(new_cycle)
                    aux_label = [EC.edges[e]['label'] for e in new_cycle]
                    min_label.append(aux_label)

            if len(min_basis) == nullity:
                break

        if len(min_basis) < nullity:
            raise RuntimeError('Construction error, not enough cycles found!')

        return min_basis

    def edge_matrix(self, nx_edges, minimum_label, new_cycle):
        """
        Return a binary matrix for operations on Z2, representing current
        cycle candidates and a test cycle.

        Args:
            nx_edges (nx.Graph):A networkx graph backbone being rebuilt with cycle base edges
            minimum_label (list): The labels sorting the edges in the binary cycle matrix.
            new_cycle (list): A list of edges of the cycle to be tested.

        Returns:
            ndarray: Numpy array representing a binary cycle matrix in Z2.

        Raises:
            Exception: description

        """

        rows = len(nx_edges.edges())
        length_basis = len(minimum_label)
        columns = length_basis+1
        E = np.zeros((rows, columns))

        for i in range(length_basis):
            E[minimum_label[i], i] = 1

        for m in new_cycle:
            E[nx_edges.edges[m]['label'], -1] = 1

        return E

    def compute_linear_independence(self, edge_mat):
        
        """
        Return bool whether all columns of E are linear independent in Z2.

        Args:
            edge_mat (ndarray): An ndarray representing a binary cycle matrix in Z2.

        Returns:
            bool: Result indicating whether the columns are linear independent.

        Raises:
            Exception: description

        """

        linear_independent = False
        columns = len(edge_mat[0, :])

        # calc echelon form
        a_columns = np.arange(columns-1)
        for col in a_columns:
            idx_nz = np.nonzero(edge_mat[col:, col])[0]
            idx = idx_nz[0]+col

            if len(idx_nz) == 1:
                edge_mat[[col, idx_nz[0]+col], :] = edge_mat[[idx_nz[0]+col, col], :]

            else:

                new_idx = idx_nz[1:]+col
                aux_E = np.add(edge_mat[new_idx], edge_mat[idx])
                edge_mat[new_idx] = np.mod(aux_E, 2)
                edge_mat[[col, idx_nz[0]+col], :] = edge_mat[[idx_nz[0]+col, col], :]

        r = np.nonzero(edge_mat[columns-1:, -1])[0]
        if r.size:
            linear_independent = True

        return linear_independent

    def breadth_first_tree(self, root):

        """
        Return a bfs-tree from root, as well a dictionary of shortest paths
        between branching points and leaves.

        Args:
            root (int): The root vertex for bfs search.

        Returns:
            nx.Graph: The spanning tree from bfs search
            dictionary:  A dicitonary of shortest paths between branching points and leaves.

        """

        T = nx.Graph()
        push_down = nx.get_node_attributes(self.G, 'push')
        len_n = len(self.G.nodes())

        if len(push_down.keys()) != len_n:
            push_down = {}
            for n in self.G.nodes():
                push_down[n] = False

        push_down[root] = True
        root_queue = []

        labels = self.G.edges(root)
        dict_path = {root: [root]}

        args = [root, T, labels, push_down, dict_path, root_queue]
        self.compute_sprouts(*args)

        while T.number_of_nodes() < len_n:
            new_queue = []
            for q in root_queue:

                labels = self.G.edges(q)
                args = [q, T, labels, push_down, dict_path, new_queue]
                self.compute_sprouts(*args)

            root_queue = new_queue[:]

        return T, dict_path

    def compute_sprouts(self, root, T, labels, push_down, dict_path, queue):


        """
        Update bfs push list and tree structure.
        """

        for e in labels:

            if e[0] == root:
                if not push_down[e[1]]:
                    T.add_edge(*e)
                    queue.append(e[1])
                    push_down[e[1]] = True
                    dict_path[e[1]] = dict_path[root]+[e[1]]
            else:
                if not push_down[e[0]]:
                    T.add_edge(*e)
                    queue.append(e[0])
                    push_down[e[0]] = True
                    dict_path[e[0]] = dict_path[root]+[e[0]]
