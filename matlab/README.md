# Tiny-SEM simulator — MATLAB version

MATLAB implementation of the simulation study in *Semantic Edge Caching with
Energy-Aware TinyML Inference for the 6G Metaverse: A Hierarchical Game Approach*
(ICC 2027 submission). It runs the full simulation: the system model, Tiny-SEM, all
baselines and ablations, and the follower-game analysis. It then regenerates
Fig. 2, Fig. 3, Table I and the numbers quoted in Sec. V.

## Requirements

- MATLAB R2021a or newer. It uses `tiledlayout`, `exportgraphics`, `xline` with
  vectors, and implicit expansion.
- No toolboxes are required.
- The Parallel Computing Toolbox is optional. If present, `run_all_experiments`
  uses `parfor` automatically.

## Quick start

```matlab
tinysem_smoke_test          % a few minutes: checks the port against the paper's values
R = run_all_experiments(true);   % quick study (3 seeds, T = 2500) to test the pipeline
make_figures                % figures + Table I from tinysem_results.mat

R = run_all_experiments();  % full study used in the paper (~1,500 runs)
make_figures
```

To run a single simulation:

```matlab
P = tinysem_params('drift_every', 1000);           % default scenario with demand drift
o = tinysem_run(P, 0, 'tinysem');                  % seed 0
fprintf('hit %.3f, utility %.3f, worst E/Ebar %.3f\n', o.hit, o.util, o.energy_ratio_max);

o = tinysem_run(P, 0, 'cmarl', 'alpha_mu', 0.02);   % constrained MARL baseline
o = tinysem_run(P, 0, 'tinysem', 'hard', true);     % Tiny-SEM-H (tau -> 0)
o = tinysem_run(P, 0, 'tinysem', 'feedback', false); % ablation: no congestion feedback
```

## Files

| File | Purpose |
|---|---|
| `tinysem_params.m` | All parameters of Sec. V, with name/value overrides |
| `tinysem_world.m`, `tinysem_make_demand.m` | Per-seed topology, users, channels, intent distributions (Zipf global and personal) |
| `tinysem_noisy_expect.m` | E_{n,k}[f] under the clustered confusion model (gives h, w, c) |
| `tinysem_sample_requests.m` | Intents and mislabeled reports |
| `tinysem_greedy_cache.m` | Density-greedy knapsack placement |
| `tinysem_run.m` | One simulation run: Tiny-SEM, Static Stackelberg, C-MARL, LRU-MA, energy-agnostic, and all ablations |
| `tinysem_method.m` | Maps paper labels (`tinysem`, `static`, `cmarl`, `abl_cum`, …) to run options |
| `tinysem_game_*.m` | Follower game: setup, best-response iteration, welfare, social-optimum lower bound |
| `run_all_experiments.m` | Every experiment (E1–E3, Table I, sensitivity, drift, game) → `tinysem_results.mat` |
| `make_figures.m` | Fig. 2 (`fig_eval.pdf`), Fig. 3 (`fig_drift.pdf`), Table I, paired gains, LDP, recovery times |
| `tinysem_smoke_test.m` | Consistency check against the paper's values |

## Model conventions (match the paper)

- **Time and units.** One slot is Δ = 1 s; sizes are in MB and energy is in mJ. The
  congestion sensitivity is ε = 1/(R_bh·Δ).
- **Energy normalization.** The controller measures energy in units of e_max, so V
  is dimensionless.
- **Device policy.** Each device uses the logit rule
  p_{n,k} ∝ exp(u_{n,k}/τ − Z_n e_k/(V τ)), where action 0 is "skip" (no request,
  no energy). Each device uses Ĝ_{−n} = d̂ / g_n.
- **Virtual queue.** It uses the tightened budget (1−θ)·Ebar_n with θ = `margin`.
- **What is averaged where.** Energy compliance is averaged over the whole horizon;
  performance metrics exclude the `warm` slots.
- **Seeds.** For a given seed, all methods share the same topology and demand.
  Dynamics use a separate stream (seed + 10000).

## Relation to the Python reference implementation

The paper's numbers were produced with the Python version of this simulator, which
the MATLAB code mirrors function by function.

MATLAB's random streams differ from NumPy's, so individual runs differ. Averages
agree within the reported confidence intervals. `tinysem_smoke_test` checks this,
with tolerances of several CI widths.

This MATLAB port was written and statically checked (MISS_HIT parser and linter:
no issues) but **not executed** by its author, because no MATLAB/Octave license
was available. Please run `tinysem_smoke_test` first. If any check fails, the
printed table shows which quantity deviates.
