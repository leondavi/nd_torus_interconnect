# nd torus interconnect graph
Generates interconnect nd torus graph with netowrkx
and prints it as 3d graph.

Possible spectral embedding layouts of 2d and 3d. 

## Installation (uv)
1. Install [uv](https://github.com/astral-sh/uv) if you have not already, for example via Homebrew: `brew install uv`.
2. From the repository root, install the runtime dependencies declared in `pyproject.toml` with `uv sync`. This will create a `.venv` that uv manages for you.
3. Whenever you need to run a script, either activate the environment (`source .venv/bin/activate`) or prefix the command with `uv run`, e.g. `uv run python spectral_embedding.py`.

Run `spectral_embedding.py`

## Usage
- Launch the interactive Plotly figure: `uv run python spectral_embedding.py`.
- Change the spectral embedding dimensionality with `--dim`, e.g. `uv run python spectral_embedding.py --dim 2`.
- Customize the torus size via `--torus`, e.g. `uv run python spectral_embedding.py --torus 5 6 7`.

2d interconnected torus: 
![alt text](https://drive.google.com/uc?export=download&id=1zIVDfn8804Bt3L4Hh_-9bmUW2DZBTx4Q)

3d interconntected torus: 
![alt text](https://drive.google.com/uc?export=download&id=1kAX1Wm1wjlpVBv5HqIlHM0tGTGeohiX0)
