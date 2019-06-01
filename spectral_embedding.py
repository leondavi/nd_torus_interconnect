from interconnect_nd_torus import *
import networkx as nx
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objs as go
from mayavi import mlab

import scipy.sparse.linalg as la


def spectral_embedding(G,dim):
    lap_mat = nx.laplacian_matrix(G)
    eigvals,eigvecs = la.eigs(lap_mat.astype(float),k=4,which='SM')
    Coords = eigvecs.T[1:].real
    if dim == 2:
        Coords[-1] = 0
    return Coords.T

def draw_graph3d(graph,dim=3, graph_colormap='winter', bgcolor = (1, 1, 1),
                 node_size=0.01,
                 edge_color=(0.8, 0.8, 0.8), edge_size=0.0002,
                 text_size=0.0008, text_color=(0, 0, 0)):

    #H=nx.Graph()

    # add edges
    # for node, edges in graph.items():
    #     for edge, val in edges.items():
    #         if val == 1:
    #             H.add_edge(node, edge)

    G=nx.convert_node_labels_to_integers(graph)

  #  graph_pos=nx.spring_layout(G, dim=3)

    # numpy array of x,y,z positions in sorted node order
#    xyz=np.array([graph_pos[v] for v in sorted(G)])
    xyz = spectral_embedding(G,dim)
    # scalar colors
    scalars=np.array(G.nodes())+5
    mlab.figure(1, bgcolor=bgcolor)
    mlab.clf()

    #----------------------------------------------------------------------------
    # the x,y, and z co-ordinates are here
    # manipulate them to obtain the desired projection perspective
    pts = mlab.points3d(xyz[:,0], xyz[:,1], xyz[:,2],
                        scalars,
                        scale_factor=node_size,
                        scale_mode='none',
                        colormap=graph_colormap,
                        resolution=20)
    #----------------------------------------------------------------------------

    # for i, (x, y, z) in enumerate(xyz):
    #     label = mlab.text(x, y, str(i), z=z,
    #                       width=text_size, name=str(i), color=text_color)
    #     label.property.shadow = True

    pts.mlab_source.dataset.lines = np.array(G.edges())
    tube = mlab.pipeline.tube(pts, tube_radius=edge_size)
    mlab.pipeline.surface(tube, color=edge_color)

    mlab.show() # interactive window




torus3d = interconnect_nd_toros([5,6,7]).get_G()
torus2d = interconnect_nd_toros([15,15]).get_G()

six_rand_reg_graph = nx.random_regular_graph(6,100)

draw_graph3d(torus2d,dim=3)




