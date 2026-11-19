# Tiny-SEM simulation code

Simulation code for the paper *Semantic Edge Caching with Energy-Aware TinyML Inference for the 6G Metaverse: A Hierarchical Game Approach* (S. Hosny and J. Tadrous, IEEE ICC 2027 submission).

The repository has two implementations of the same simulator:

| Folder | What it is |
|---|---|
| `python/` | The simulator that produced the numbers in the paper. |
| `matlab/` | A MATLAB port of the full simulator, with its own README. It reproduces Fig. 2, Fig. 3, Table I and the numbers quoted in Sec. V. |

## Python

Requires Python 3.9 or newer, NumPy and Matplotlib.

| File | Purpose |
|---|---|
| `tinysem.py` | System model, Tiny-SEM, all baselines and ablations, and the follower-game helpers. |
| `experiments.py` | Runs the table and the parameter sweeps; writes `results.json`. |
| `game_analysis.py` | Equilibrium uniqueness and welfare analysis of the follower game; writes `game_results.json`. |
| `plots.py` | Draws the evaluation figure (`fig_eval.pdf`, `fig_eval.png`) from the two JSON files. |

```
cd python
python3 experiments.py
python3 game_analysis.py
python3 plots.py
```

Outputs are written next to the scripts. The runs use all CPU cores through `multiprocessing`.

## MATLAB

Requires MATLAB R2021a or newer; no toolboxes are needed. See `matlab/README.md`.

```matlab
cd matlab
tinysem_smoke_test               % checks the port against the paper's values
R = run_all_experiments(true);   % quick study to test the pipeline
make_figures
```
