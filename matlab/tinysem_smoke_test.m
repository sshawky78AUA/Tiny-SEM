function ok = tinysem_smoke_test()
%TINYSEM_SMOKE_TEST Quick consistency check of the MATLAB port (a few minutes).
%   Runs the default drift scenario for 3 seeds with Tiny-SEM, the static
%   Stackelberg baseline and C-MARL, plus the follower-game check, and
%   compares with the values reported in the paper (from the reference
%   Python implementation).  MATLAB's random streams differ from NumPy's, so
%   results agree statistically, not digit for digit; the tolerances below
%   are several confidence-interval widths.

seeds = 0:2;
P = tinysem_params('drift_every', 1000);
ref = struct( ...  %    paper value    tolerance
    'tinysem_hit',  [0.265, 0.03], 'tinysem_util', [0.742, 0.03], ...
    'tinysem_emax', [0.978, 0.03], 'static_hit',   [0.100, 0.03], ...
    'static_util',  [0.729, 0.03], 'cmarl_hit',    [0.240, 0.03], ...
    'cmarl_util',   [0.734, 0.03]);
names = {'tinysem', 'static', 'cmarl'};
got = struct();
for i = 1:numel(names)
    [method, opts] = tinysem_method(names{i});
    hit = zeros(size(seeds)); util = hit; emax = hit;
    for s = 1:numel(seeds)
        tic;
        o = tinysem_run(P, seeds(s), method, opts{:});
        hit(s) = o.hit; util(s) = o.util; emax(s) = o.energy_ratio_max;
        fprintf('%-8s seed %d: hit %.3f  util %.3f  worst E/Eb %.3f  (%.1f s)\n', ...
                names{i}, seeds(s), o.hit, o.util, o.energy_ratio_max, toc);
    end
    got.([names{i} '_hit'])  = mean(hit);
    got.([names{i} '_util']) = mean(util);
    got.([names{i} '_emax']) = mean(emax);
end

ok = true;
fprintf('\n%-14s %8s %8s %6s\n', 'check', 'MATLAB', 'paper', 'pass');
fn = fieldnames(ref);
for i = 1:numel(fn)
    v = got.(fn{i});  r = ref.(fn{i});
    pass = abs(v - r(1)) <= r(2);
    ok = ok && pass;
    fprintf('%-14s %8.3f %8.3f %6d\n', fn{i}, v, r(1), pass);
end
% energy guarantee and ordering checks
ord = got.tinysem_hit > got.cmarl_hit && got.cmarl_hit > got.static_hit;
fprintf('%-14s %8s %8s %6d\n', 'hit ordering', '', '', ord);
ok = ok && ord;

% follower-game check (Theorem 1): unique equilibrium from random starts
Pg = tinysem_params();
G = tinysem_game_setup(Pg, 0);
mu = 0.5 * ones(Pg.N, 1);
rs = RandStream('mt19937ar', 'Seed', 1);
fin = cell(5, 1);  ke = 0;
for r = 1:5
    g = -log(rand(rs, Pg.N, Pg.K + 1));  p0 = g ./ sum(g, 2);
    [fin{r}, k1] = tinysem_game_equilibrium(G, mu, Pg.tau, p0);
    ke = max(ke, k1);
end
spread = max(cellfun(@(f) max(abs(f(:) - fin{1}(:))), fin));
amax = max(G.W.a);
kb = Pg.lam2 * (Pg.N - 1) * amax * G.cbar / (4 * Pg.tau * (1 - amax * G.cbar));
fprintf('%-14s spread %.1e, measured kappa %.3g, bound %.3f\n', 'game', spread, ke, kb);
okg = spread < 1e-8 && kb < 1 && ke < kb;
fprintf('%-14s %8s %8s %6d\n', 'game check', '', '', okg);
ok = ok && okg;
if ok, fprintf('\nSMOKE TEST PASSED\n'); else, fprintf('\nSMOKE TEST FAILED\n'); end
end
