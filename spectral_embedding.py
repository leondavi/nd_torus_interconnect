import argparse

import matplotlib.cm as cm
import matplotlib.colors as mcolors
import networkx as nx
import numpy as np
import plotly.graph_objs as go
import scipy.sparse.linalg as la

from interconnect_nd_torus import *


def spectral_embedding(G,dim):
    lap_mat = nx.laplacian_matrix(G)
    eigvals,eigvecs = la.eigs(lap_mat.astype(float),k=4,which='SM')
    Coords = eigvecs.T[1:].real
    if dim == 2:
        Coords[-1] = 0
    return Coords.T

def draw_graph3d(graph, dim=3, graph_colormap="winter", bgcolor=(1, 1, 1),
                 node_size=0.001,
                 edge_color=(0.8, 0.8, 0.8), edge_size=0.0002,
                 text_color=(0, 0, 0)):

    #H=nx.Graph()

    # add edges
    # for node, edges in graph.items():
    #     for edge, val in edges.items():
    #         if val == 1:
    #             H.add_edge(node, edge)

    G = nx.convert_node_labels_to_integers(graph)

  #  graph_pos=nx.spring_layout(G, dim=3)

    # numpy array of x,y,z positions in sorted node order
#    xyz=np.array([graph_pos[v] for v in sorted(G)])
    xyz = spectral_embedding(G, dim)
    scalars = np.array(G.nodes()) + 5

    _draw_with_plotly(G, xyz, scalars, graph_colormap, bgcolor,
                      node_size, edge_color, edge_size, text_color)




def _draw_with_plotly(G, xyz, scalars, graph_colormap, bgcolor,
                      node_size, edge_color, edge_size, text_color):
    edge_x, edge_y, edge_z = [], [], []
    for u, v in G.edges():
        edge_x.extend([xyz[u, 0], xyz[v, 0], None])
        edge_y.extend([xyz[u, 1], xyz[v, 1], None])
        edge_z.extend([xyz[u, 2], xyz[v, 2], None])

    edge_trace = go.Scatter3d(
        x=edge_x,
        y=edge_y,
        z=edge_z,
        mode='lines',
        line=dict(color=_as_rgb(edge_color), width=max(edge_size * 4000, 1)),
        hoverinfo='none'
    )

    colors = _colors_from_cmap(scalars, graph_colormap)
    node_trace = go.Scatter3d(
        x=xyz[:, 0],
        y=xyz[:, 1],
        z=xyz[:, 2],
        mode='markers',
        marker=dict(size=max(node_size * 6000, 4), color=colors, opacity=0.9),
        hoverinfo='text',
        text=[f"node {n}" for n in G.nodes()],
        textfont=dict(color=_as_rgb(text_color)),
    )

    bg_rgb = _as_rgb(bgcolor)
    fig = go.Figure(data=[edge_trace, node_trace])
    fig.update_layout(scene=dict(bgcolor=bg_rgb), paper_bgcolor=bg_rgb)
    fig.show()


def _colors_from_cmap(values, cmap_name):
    cmap = cm.get_cmap(cmap_name)
    norm = mcolors.Normalize(vmin=np.min(values), vmax=np.max(values))
    rgba = cmap(norm(values))[:, :3]
    rgb = (rgba * 255).astype(int)
    return [f"rgb({r},{g},{b})" for r, g, b in rgb]


def _as_rgb(color):
    if isinstance(color, str):
        return color
    r, g, b = color
    if max(r, g, b) <= 1:
        r, g, b = [int(x * 255) for x in (r, g, b)]
    else:
        r, g, b = [int(x) for x in (r, g, b)]
    return f"rgb({r},{g},{b})"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize ND torus interconnect graphs")
    parser.add_argument("--dim", type=int, default=3,
                        help="Spectral embedding dimension for visualization")
    parser.add_argument("--torus", nargs="*", type=int, default=[15, 15],
                        help="List of ring sizes describing the torus")
    args = parser.parse_args()

    torus_graph = interconnect_nd_toros(args.torus).get_G()
    draw_graph3d(torus_graph, dim=args.dim)




