function R = run_all_experiments(quick)
%RUN_ALL_EXPERIMENTS Reproduce every experiment of Sec. V of the paper.
%
%   R = run_all_experiments()       full study (about 1,500 runs)
%   R = run_all_experiments(true)   quick version: 3 seeds, T = 2500 (sanity check)
%
%   Uses PARFOR automatically when the Parallel Computing Toolbox is
%   available (otherwise a plain loop).  Results are saved to
%   tinysem_results.mat and consumed by make_figures.m.
%
%   Experiments (labels as in the paper):
%     E1    utility / backlog vs. V (stationary)                -> Fig. 2(a)
%     E2    hit ratio and latency vs. cache size (drift)        -> Fig. 2(b)
%     E3    worst-user energy vs. budget (stationary)           -> Fig. 2(c)
%     TAB   all methods + ablations, drift, 20 seeds            -> Table I
%     TABS  same, stationary                                    -> text
%     SENS  error structure / accuracy profile (drift)          -> "Sensitivity"
%     DRIFT hit-ratio time series (drift, 8 seeds)              -> Fig. 3
%     GAME  contraction factor and welfare vs. tau, N           -> Fig. 2(d), text

if nargin < 1, quick = false; end
if quick
    S_SWEEP = 0:2;  S_TAB = 0:2;  S_DRIFT = 0:1;  S_GAME = 0:1;  T = 2500;  WARM = 500;
else
    S_SWEEP = 0:9;  S_TAB = 0:19; S_DRIFT = 0:7;  S_GAME = 0:4;  T = 5000;  WARM = 1000;
end
base = {'T', T, 'warm', WARM};

jobs = {};
add = @(exp, x, m, s, kv) {exp, x, m, s, kv};
% ---- E1
for V = [1 2 5 10 20 50 100]
    for s = S_SWEEP
        jobs{end+1} = add('E1', V, 'tinysem', s, {'V', V});
        jobs{end+1} = add('E1', V, 'tinysem_hard', s, {'V', V});
    end
end
for s = S_SWEEP
    for m = {'static', 'cmarl', 'lru_maxacc'}
        jobs{end+1} = add('E1', 0, m{1}, s, {});
    end
end
% ---- E2
for cf = [0.02 0.05 0.10 0.15 0.20]
    for m = {'tinysem', 'tinysem_hard', 'static', 'cmarl', 'lru_maxacc'}
        for s = S_SWEEP
            jobs{end+1} = add('E2', cf, m{1}, s, {'cache_frac', cf, 'drift_every', 1000});
        end
    end
end
% ---- E3 (plus Tiny-SEM with V = 2)
for bf = [0.1 0.2 0.4 0.6 0.8 1.0]
    for m = {'tinysem', 'static', 'cmarl', 'lru_maxacc', 'agnostic'}
        for s = S_SWEEP
            jobs{end+1} = add('E3', bf, m{1}, s, {'budget_frac', bf});
        end
    end
    for s = S_SWEEP
        jobs{end+1} = add('E3V2', bf, 'tinysem', s, {'budget_frac', bf, 'V', 2});
    end
end
% ---- TAB / TABS
labels = {'tinysem','tinysem_hard','static','cmarl','lru_maxacc','agnostic', ...
          'abl_nofb','abl_cum','abl_lru','abl_maxacc_greedy'};
for m = labels
    for s = S_TAB
        jobs{end+1} = add('TAB',  0, m{1}, s, {'drift_every', 1000});
        jobs{end+1} = add('TABS', 0, m{1}, s, {});
    end
end
% ---- SENS
res_acc = [0.644 0.672 0.691 0.706];  res_mac = [186 290 418 569];   % Howard'17 Tab. 7
variants = {'uniform', {'beta', 9/999}; 'beta09', {'beta', 0.9}; ...
            'resprof', {'acc', res_acc, 'mmac', res_mac}};
for v = 1:size(variants, 1)
    for m = {'tinysem', 'static', 'cmarl'}
        for s = S_SWEEP
            jobs{end+1} = add(['SENS_' variants{v,1}], 0, m{1}, s, ...
                              [{'drift_every', 1000}, variants{v,2}]);
        end
    end
end
% ---- DRIFT time series
for m = {'tinysem', 'abl_cum', 'static', 'cmarl', 'lru_maxacc'}
    for s = S_DRIFT
        jobs{end+1} = add('DRIFT', 0, m{1}, s, {'drift_every', 1000});
    end
end

nj = numel(jobs);
fprintf('Running %d simulation jobs...\n', nj);
res = cell(nj, 1);
tic;
if license('test', 'Distrib_Computing_Toolbox') && ~isempty(ver('parallel'))
    parfor j = 1:nj
        res{j} = run_job(jobs{j}, base);
    end
else
    for j = 1:nj
        res{j} = run_job(jobs{j}, base);
        if mod(j, 50) == 0, fprintf('  %d/%d  (%.0f s)\n', j, nj, toc); end
    end
end
fprintf('Simulations done in %.0f s.\n', toc);

R.jobs = jobs;
R.res  = res;
R.quick = quick;
R.game = run_game(S_GAME, quick);
save('tinysem_results.mat', 'R', '-v7.3');
fprintf('Saved tinysem_results.mat\n');
end

% ======================================================================
function o = run_job(job, base)
[exp, x, label, seed, kv] = job{:}; %#ok<ASGLU>
[method, opts] = tinysem_method(label);
P = tinysem_params(base{:}, kv{:});
o = tinysem_run(P, seed, method, opts{:});
if ~strcmp(exp, 'DRIFT')          % keep time series only where needed
    o = rmfield(o, {'hit_t', 'sem_t'});
end
end

function Gr = run_game(seeds, quick)
% Contraction factor and welfare of the follower game (Theorem 1 checks).
taus = [0.01 0.02 0.05 0.1 0.2 0.4 0.8];
cases = {};
for s = seeds
    cases{end+1} = [500 50 s];   %#ok<AGROW>
    cases{end+1} = [100 50 s];   %#ok<AGROW>
    for N = [10 20 30 40], cases{end+1} = [500 N s]; end %#ok<AGROW>
end
if quick, starts = 5; else, starts = 30; end
Gr = struct('Rbh', {}, 'N', {}, 'seed', {}, 'tau', {}, 'kbound', {}, ...
            'kemp', {}, 'spread', {}, 'eff', {});
for c = 1:numel(cases)
    Rbh = cases{c}(1);  N = cases{c}(2);  seed = cases{c}(3);
    P = tinysem_params('Rbh', Rbh, 'N', N, 'T', 3000, 'warm', 1000);
    o = tinysem_run(P, seed, 'tinysem');
    mu = o.Z_user / (P.energy(end) * P.V);           % energy prices Z_n/(V e_max)
    G = tinysem_game_setup(P, seed);
    rs = RandStream('mt19937ar', 'Seed', seed + 77);
    amax = max(G.W.a);
    kb = P.lam2 * (P.N - 1) * amax * G.cbar ./ (4 * taus * (1 - amax * G.cbar));
    Wso = tinysem_game_social_opt(G, mu, rs);
    for i = 1:numel(taus)
        finals = cell(starts, 1);  kemp = 0;
        for r = 1:starts
            p0 = dirichlet_rows(rs, P.N, P.K + 1);
            [finals{r}, ke] = tinysem_game_equilibrium(G, mu, taus(i), p0);
            kemp = max(kemp, ke);
        end
        spread = 0;
        for r = 2:starts, spread = max(spread, max(abs(finals{r}(:) - finals{1}(:)))); end
        Wne = tinysem_game_welfare(G, finals{1}, mu);
        Gr(end+1) = struct('Rbh', Rbh, 'N', N, 'seed', seed, 'tau', taus(i), ...
            'kbound', kb(i), 'kemp', kemp, 'spread', spread, 'eff', Wso / Wne); %#ok<AGROW>
    end
    fprintf('  game case %d/%d done\n', c, numel(cases));
end
end

function p = dirichlet_rows(rs, n, k)
% Uniform random points on the simplex (Dirichlet(1,...,1)).
g = -log(rand(rs, n, k));
p = g ./ sum(g, 2);
end
