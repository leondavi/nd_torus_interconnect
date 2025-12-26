"""Shared utilities for building spectral embeddings and Plotly visualizations."""

from __future__ import annotations

import matplotlib.cm as cm
import matplotlib.colors as mcolors
import networkx as nx
import numpy as np
import plotly.graph_objs as go
import scipy.sparse.linalg as la


def spectral_embedding(graph: nx.Graph, dim: int, k: int | None = None) -> np.ndarray:
    """Return a spectral embedding of *graph* in ``dim`` dimensions."""
    lap_mat = nx.laplacian_matrix(graph)
    eig_count = k or max(dim + 1, 4)
    eigvals, eigvecs = la.eigs(lap_mat.astype(float), k=eig_count, which="SM")
    coords = eigvecs.T[1 : dim + 1].real
    if dim == 2:
        coords[-1] = 0
    return coords.T


def draw_graph3d(
    graph: nx.Graph,
    dim: int = 3,
    graph_colormap: str = "winter",
    bgcolor: tuple[float, float, float] = (1, 1, 1),
    node_size: float = 0.001,
    edge_color: tuple[float, float, float] = (0.8, 0.8, 0.8),
    edge_size: float = 0.0002,
    text_color: tuple[float, float, float] = (0, 0, 0),
    normalize: bool = False,
    scale_range: tuple[float, float] = (0.0, 1.0),
) -> None:
    """Render *graph* using a spectral embedding and Plotly."""
    graph = nx.convert_node_labels_to_integers(graph)
    xyz = spectral_embedding(graph, dim)
    if xyz.shape[1] < 3:
        padding = np.zeros((xyz.shape[0], 3 - xyz.shape[1]))
        xyz = np.hstack([xyz, padding])
    if normalize:
        xyz = _normalize_coords(xyz, scale_range)
    scalars = np.array(graph.nodes()) + 5

    _draw_with_plotly(
        graph,
        xyz,
        scalars,
        graph_colormap,
        bgcolor,
        node_size,
        edge_color,
        edge_size,
        text_color,
    )


def _draw_with_plotly(
    graph: nx.Graph,
    xyz: np.ndarray,
    scalars: np.ndarray,
    graph_colormap: str,
    bgcolor: tuple[float, float, float],
    node_size: float,
    edge_color: tuple[float, float, float],
    edge_size: float,
    text_color: tuple[float, float, float],
) -> None:
    edge_x, edge_y, edge_z = [], [], []
    for u, v in graph.edges():
        edge_x.extend([xyz[u, 0], xyz[v, 0], None])
        edge_y.extend([xyz[u, 1], xyz[v, 1], None])
        edge_z.extend([xyz[u, 2], xyz[v, 2], None])

    edge_trace = go.Scatter3d(
        x=edge_x,
        y=edge_y,
        z=edge_z,
        mode="lines",
        line=dict(color=_as_rgb(edge_color), width=max(edge_size * 4000, 1)),
        hoverinfo="none",
    )

    colors = _colors_from_cmap(scalars, graph_colormap)
    node_trace = go.Scatter3d(
        x=xyz[:, 0],
        y=xyz[:, 1],
        z=xyz[:, 2],
        mode="markers",
        marker=dict(size=max(node_size * 6000, 4), color=colors, opacity=0.9),
        hoverinfo="text",
        text=[f"node {n}" for n in graph.nodes()],
        textfont=dict(color=_as_rgb(text_color)),
    )

    bg_rgb = _as_rgb(bgcolor)
    fig = go.Figure(data=[edge_trace, node_trace])
    fig.update_layout(scene=dict(bgcolor=bg_rgb), paper_bgcolor=bg_rgb)
    fig.show()


def _colors_from_cmap(values: np.ndarray, cmap_name: str) -> list[str]:
    cmap = cm.get_cmap(cmap_name)
    norm = mcolors.Normalize(vmin=np.min(values), vmax=np.max(values))
    rgba = cmap(norm(values))[:, :3]
    rgb = (rgba * 255).astype(int)
    return [f"rgb({r},{g},{b})" for r, g, b in rgb]


def _as_rgb(color: tuple[float, float, float] | str) -> str:
    if isinstance(color, str):
        return color
    r, g, b = color
    if max(r, g, b) <= 1:
        r, g, b = [int(x * 255) for x in (r, g, b)]
    else:
        r, g, b = [int(x) for x in (r, g, b)]
    return f"rgb({r},{g},{b})"


def _normalize_coords(xyz: np.ndarray, scale_range: tuple[float, float]) -> np.ndarray:
    lower, upper = scale_range
    if upper <= lower:
        raise ValueError("scale_range upper bound must exceed lower bound")
    span = upper - lower
    normalized = np.array(xyz, copy=True)
    for axis in range(normalized.shape[1]):
        column = normalized[:, axis]
        col_min = column.min()
        col_max = column.max()
        if np.isclose(col_max, col_min):
            normalized[:, axis] = lower + span / 2
        else:
            normalized[:, axis] = lower + (column - col_min) * span / (col_max - col_min)
    return normalized
