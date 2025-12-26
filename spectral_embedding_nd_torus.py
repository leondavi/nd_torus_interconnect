import argparse

from common import draw_graph3d
from interconnect_nd_torus import interconnect_nd_toros


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize ND torus interconnect graphs")
    parser.add_argument("--dim", type=int, default=3,
                        help="Spectral embedding dimension for visualization")
    parser.add_argument("--torus", nargs="*", type=int, default=[15, 15],
                        help="List of ring sizes describing the torus")
    args = parser.parse_args()

    torus_graph = interconnect_nd_toros(args.torus).get_G()
    draw_graph3d(torus_graph, dim=args.dim)


if __name__ == "__main__":
    main()




