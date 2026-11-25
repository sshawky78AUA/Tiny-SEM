# Tiny-SEM simulation code

Simulation code for the paper *Semantic Edge Caching with Energy-Aware TinyML Inference for the 6G Metaverse: A Hierarchical Game Approach* (S. Hosny and J. Tadrous, IEEE ICC 2027 submission).

The simulator is in `matlab/`. It runs the full system of the paper: users with Bernoulli activity and Zipf intents over the ImageNet-1k catalog, the four MobileNetV1 TinyML models with the clustered confusion model, the edge cache, backhaul congestion, Tiny-SEM (leader placement and the users' logit drift-plus-penalty rule), all baselines and ablations, and the follower-game analysis. It regenerates Fig. 2, Fig. 3, Table I and the numbers quoted in Sec. V, and you can rerun any of them with your own parameters.

## Requirements

- MATLAB R2021a or newer; no toolboxes are required. With the Parallel Computing Toolbox, `run_all_experiments` runs in parallel automatically.
- Or GNU Octave (free), version 8 or newer: add the `octave_compat/` folder to the path, which supplies MATLAB's `RandStream`. Octave's random numbers differ from MATLAB's, but the averages agree; the smoke test passes in Octave 8.4. `make_figures` uses `tiledlayout` and `exportgraphics`, so it needs MATLAB.

## Quick start

```matlab
cd matlab
% addpath('../octave_compat')            % Octave only
tinysem_smoke_test                       % a few minutes: checks the simulator against the paper's values

P = tinysem_params('drift_every', 1000); % default scenario of the paper, with demand drift
o = tinysem_run(P, 0, 'tinysem');        % one run, seed 0
fprintf('hit %.3f, utility %.3f, worst E/Ebar %.3f\n', o.hit, o.util, o.energy_ratio_max);
```

## Changing parameters

Every parameter of Sec. V is a field of `tinysem_params`, and any of them can be overridden by name:

```matlab
P = tinysem_params('N', 100, 'cache_frac', 0.05, 'V', 2, 'Rbh', 250);
o = tinysem_run(P, 3, 'cmarl', 'alpha_mu', 0.02);
```

`tinysem_sweep` varies one parameter across several methods and seeds and prints a table with 95% confidence half-widths:

```matlab
tinysem_sweep('cache_frac', [0.05 0.1 0.2], {'tinysem', 'static', 'cmarl'}, 0:4);
tinysem_sweep('V', [2 10 50], {'tinysem', 'tinysem_hard'}, 0:2, 'drift_every', 1000);
T = tinysem_sweep('budget_frac', [0.2 0.4 0.8], {'tinysem'}, 0:4);   % T holds one row per run
```

Main parameters (defaults of the paper):

| Parameter | Default | Meaning |
|---|---|---|
| `N` | 50 | users |
| `O`, `G` | 1000, 10 | catalog size, semantic cluster size |
| `zipf`, `omega` | 0.8, 0.3 | Zipf exponent, weight of the personal intent profile |
| `a_lo`, `a_hi` | 0.3, 0.7 | range of activity probabilities |
| `cache_frac` | 0.10 | cache capacity, as a share of the catalog volume |
| `acc`, `mmac`, `eta_E` | MobileNetV1 values | model accuracies, MACs, energy per MAC |
| `beta` | 0.6 | share of errors inside the semantic cluster |
| `Rbh` | 500 | backhaul capacity (MB per 1-s slot) |
| `budget_frac` | 0.4 | energy budget, as a share of a_n e_max |
| `V`, `tau`, `margin` | 10, 0.2, 0.03 | Lyapunov trade-off, entropy temperature, budget margin theta |
| `gamma`, `gamma_d` | 0.005, 0.1 | forgetting factors of the demand and congestion estimates |
| `T`, `warm` | 5000, 1000 | slots per run, warm-up slots excluded from metrics |
| `drift_every` | 0 | redraw popularity every k slots (0 = stationary; the paper uses 1000) |

Methods (`tinysem_method`): `tinysem`, `tinysem_hard`, `static`, `cmarl`, `lru_maxacc`, `agnostic`, and the ablations `abl_nofb`, `abl_cum`, `abl_lru`, `abl_maxacc_greedy`.

## Reproducing the paper

```matlab
R = run_all_experiments(true);   % quick check: 3 seeds, T = 2500
R = run_all_experiments();       % full study of the paper (about 1,500 runs); saves tinysem_results.mat
make_figures                     % Fig. 2, Fig. 3, Table I and the quoted numbers
```

The full study takes a few hours on one core. See `matlab/README.md` for the file list and the model conventions.

## Folders

| Folder | What it is |
|---|---|
| `matlab/` | The simulator of the paper. |
| `octave_compat/` | `RandStream`, `rand`, `randi` and `randperm` wrappers so the simulator runs in GNU Octave. Not needed in MATLAB. |
| `archive_python_earlier_model/` | An earlier Python version of the model with different parameters. It does not reproduce the paper and is kept for reference only. |
