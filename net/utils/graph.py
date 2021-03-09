import numpy as np


class Graph():
    """ The Graph to model the skeletons extracted by the openpose

    Args:
        strategy (string): must be one of the follow candidates
        - uniform: Uniform Labeling
        - distance: Distance Partitioning
        - spatial: Spatial Configuration
        For more information, please refer to the section 'Partition Strategies'
            in our paper (https://arxiv.org/abs/1801.07455).

        layout (string): must be one of the follow candidates
        - openpose: Is consists of 18 joints. For more information, please
            refer to https://github.com/CMU-Perceptual-Computing-Lab/openpose#output
        - ntu-rgb+d: Is consists of 25 joints. For more information, please
            refer to https://github.com/shahroudy/NTURGB-D

        max_hop (int): the maximal distance between two connected nodes
        dilation (int): controls the spacing between the kernel points

    """

    def __init__(self, num_nodes, neighbor_links,
                 strategy='uniform',
                 layout='openpose',
                 max_hop=1,
                 dilation=1):
        self.max_hop = max_hop
        self.dilation = dilation

        self.set_edges(num_nodes, neighbor_links)
        self.hop_dis = get_hop_distance(
            self.num_nodes, self.edges, max_hop=max_hop)
        self.set_adjacency(strategy)

    def __str__(self):
        return self.A

    def set_edges(self, num_nodes, neighbor_links, self_link=True):
        self.num_nodes = num_nodes
        # neighbor_links = [(0, 1), (0, 10), (0, 13),
        #                  (1, 2),
        #                  (2, 3), (2, 4), (2, 7),
        #                  (4, 5),
        #                  (5, 6),
        #                  (7, 8),
        #                  (8, 9),
        #                  (10, 11),
        #                  (11, 12),
        #                  (13, 14),
        #                  (14, 15)]
        if self_link:
            self_link = [(i, i) for i in range(self.num_nodes)]
            self.edges = self_link + neighbor_links
        else:
            self.edges = neighbor_links
        self.center = 0

    def set_adjacency(self, strategy):
        valid_hop = range(0, self.max_hop + 1, self.dilation)
        adjacency = np.zeros((self.num_nodes, self.num_nodes))
        for hop in valid_hop:
            adjacency[self.hop_dis == hop] = 1
        normalize_adjacency = normalize_digraph(adjacency)

        if strategy == 'uniform':
            A = np.zeros((1, self.num_nodes, self.num_nodes))
            A[0] = normalize_adjacency
            self.A = A
        elif strategy == 'distance':
            A = np.zeros((len(valid_hop), self.num_nodes, self.num_nodes))
            for i, hop in enumerate(valid_hop):
                A[i][self.hop_dis == hop] = normalize_adjacency[self.hop_dis ==
                                                                hop]
            self.A = A
        elif strategy == 'spatial':
            A = []
            for hop in valid_hop:
                a_root = np.zeros((self.num_nodes, self.num_nodes))
                a_close = np.zeros((self.num_nodes, self.num_nodes))
                a_further = np.zeros((self.num_nodes, self.num_nodes))
                for i in range(self.num_nodes):
                    for j in range(self.num_nodes):
                        if self.hop_dis[j, i] == hop:
                            if self.hop_dis[j, self.center] == self.hop_dis[
                                    i, self.center]:
                                a_root[j, i] = normalize_adjacency[j, i]
                            elif self.hop_dis[j, self.
                                              center] > self.hop_dis[i, self.
                                                                     center]:
                                a_close[j, i] = normalize_adjacency[j, i]
                            else:
                                a_further[j, i] = normalize_adjacency[j, i]
                if hop == 0:
                    A.append(a_root)
                else:
                    A.append(a_root + a_close)
                    A.append(a_further)
            A = np.stack(A)
            self.A = A
        else:
            raise ValueError('The given strategy does not exist')


def get_hop_distance(num_nodes, edges, max_hop=1):
    A = np.zeros((num_nodes, num_nodes))
    for edge in edges:
        A[edge[0], edge[1]] = 1
        A[edge[1], edge[0]] = 1

    # compute hop steps
    hop_dis = np.zeros((num_nodes, num_nodes)) + np.inf
    transfer_mat = [np.linalg.matrix_power(A, d) for d in range(max_hop + 1)]
    arrive_mat = (np.stack(transfer_mat) > 0)
    for d in range(max_hop, -1, -1):
        hop_dis[arrive_mat[d]] = d
    return hop_dis


def normalize_digraph(A):
    Dl = np.sum(A, 0)
    num_node = A.shape[0]
    Dn = np.zeros((num_node, num_node))
    for i in range(num_node):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i]**(-1)
    AD = np.dot(A, Dn)
    return AD


def normalize_un_digraph(A):
    Dl = np.sum(A, 0)
    num_node = A.shape[0]
    Dn = np.zeros((num_node, num_node))
    for i in range(num_node):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i]**(-0.5)
    DAD = np.dot(np.dot(Dn, A), Dn)
    return DAD
