
import networkx as nx
from scipy.sparse import lil_matrix
from operator import mul
from functools import reduce
import matplotlib.pyplot as plt
import numpy as np


class interconnect_nd_toros:

    def __init__(self,dims_list):
        self.dims = len(dims_list)
        self.num_of_nodes = reduce(mul,dims_list,1)
        self.adj_matrix = lil_matrix([self.num_of_nodes,self.num_of_nodes])

        G = nx.cycle_graph(dims_list[0])
        nx.draw(G)

        for i in range(1,len(dims_list)):
            #I = np.eye(dims_list[0])
            currentG = nx.cycle_graph(dims_list[i])

            print(nx.adjacency_matrix(G).toarray())
            G = nx.algorithms.operators.cartesian_product(G,currentG)
            print("new G of :"+str(i+1)+" dims")
            print(nx.adjacency_matrix(G).toarray())
           # H = nx.algorithms.operators.cartesian_product(G,H)
            plt.figure("figure-"+str(i))
            nx.draw_spectral(G)
            #nx.draw_networkx(G)
        plt.show()





torus_g = interconnect_nd_toros([3,4,5])


