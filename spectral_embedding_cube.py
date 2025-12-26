"""Render spectral embeddings of 3D cube graphs."""

from __future__ import annotations

import argparse

import networkx as nx

from common import draw_graph3d


def build_cube_graph(size: int) -> nx.Graph:
    if size < 2:
        raise ValueError("Cube size must be at least 2.")
    dimensions = [range(size) for _ in range(3)]
    return nx.grid_graph(dim=dimensions, periodic=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize cube spectral embeddings")
    parser.add_argument("--dim", type=int, default=3,
                        help="Spectral embedding dimension for visualization")
    parser.add_argument("--size", type=int, default=4,
                        help="Number of lattice points per cube edge (>=2)")
    parser.add_argument("--node-size", type=float, default=0.002,
                        help="Marker size multiplier for Plotly nodes")
    parser.add_argument("--edge-size", type=float, default=0.001,
                        help="Edge thickness multiplier for Plotly segments")
    parser.add_argument("--no-normalize", dest="normalize", action="store_false",
                        help="Keep raw spectral embedding coordinates instead of scaling to a unit cube")
    parser.set_defaults(normalize=True)
    args = parser.parse_args()

    cube_graph = build_cube_graph(args.size)
    draw_graph3d(
        cube_graph,
        dim=args.dim,
        node_size=args.node_size,
        edge_size=args.edge_size,
        normalize=args.normalize,
    )


if __name__ == "__main__":
    main()
