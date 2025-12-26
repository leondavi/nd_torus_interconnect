# nd torus interconnect graph
Generates interconnect nd torus graph with netowrkx
and prints it as 3d graph.

Possible spectral embedding layouts of 2d and 3d. 

## Installation (uv)
1. Install [uv](https://github.com/astral-sh/uv) if you have not already, for example via Homebrew: `brew install uv`.
2. From the repository root, install the runtime dependencies declared in `pyproject.toml` with `uv sync`. This will create a `.venv` that uv manages for you.
3. Whenever you need to run a script, either activate the environment (`source .venv/bin/activate`) or prefix the command with `uv run`, e.g. `uv run python spectral_embedding.py`.

Run `spectral_embedding_nd_torus.py` for torus graphs or `spectral_embedding_cube.py` for cube lattices.

## Usage
- Torus: `uv run python spectral_embedding_nd_torus.py --torus 5 6 7 --dim 3`.
- Cube: `uv run python spectral_embedding_cube.py --size 4 --dim 3` (use `--node-size`/`--edge-size` to tweak marker scales). The cube embedding is normalized to fill the unit cube; pass `--no-normalize` to inspect the raw spectral coordinates.
- Adjust the spectral embedding dimensionality with `--dim` to target 2D or 3D projections.

2d interconnected torus: 
![alt text](https://drive.google.com/uc?export=download&id=1zIVDfn8804Bt3L4Hh_-9bmUW2DZBTx4Q)

3d interconntected torus: 
![alt text](https://drive.google.com/uc?export=download&id=1kAX1Wm1wjlpVBv5HqIlHM0tGTGeohiX0)

<img width="1207" height="1044" alt="Screenshot 2025-12-26 at 23 54 22" src="https://github.com/user-attachments/assets/cd67fe3d-3f26-4a13-a0aa-27d60a449675" />
