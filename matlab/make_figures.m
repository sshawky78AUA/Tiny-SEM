function make_figures(file)
%MAKE_FIGURES Regenerate Fig. 2, Fig. 3, Table I and the numbers quoted in Sec. V.
%   make_figures()                     reads tinysem_results.mat
%   make_figures('my_results.mat')
%   Writes fig_eval.pdf, fig_drift.pdf and prints Table I and key statistics.

if nargin < 1, file = 'tinysem_results.mat'; end
S = load(file);  R = S.R;

col.tinysem = [42 120 214]/255;  col.tinysem_hard = col.tinysem;
col.static = [235 104 52]/255;   col.cmarl = [27 175 122]/255;
col.lru_maxacc = [237 161 0]/255; col.agnostic = [232 123 164]/255;
mk.tinysem = 'o'; mk.tinysem_hard = 'o'; mk.static = 's'; mk.cmarl = '^';
mk.lru_maxacc = 'd'; mk.agnostic = 'v';
lab.tinysem = 'Tiny-SEM (\tau=0.2)'; lab.tinysem_hard = 'Tiny-SEM-H (\tau\rightarrow0)';
lab.static = 'Static Stackelberg'; lab.cmarl = 'C-MARL'; lab.lru_maxacc = 'LRU-MA';
lab.agnostic = 'Energy-agnostic';

fig = figure('Units', 'inches', 'Position', [1 1 7.16 2.0], 'Color', 'w');
tl = tiledlayout(fig, 1, 4, 'TileSpacing', 'compact', 'Padding', 'compact');

% ---------------- (a) utility relative to static, paired by seed, vs V
ax = nexttile(tl); hold(ax, 'on');
[sb, ub] = getm(R, 'E1', 'static', 0, 'util');
for m = {'tinysem', 'tinysem_hard'}
    Vs = unique_x(R, 'E1', m{1});
    mu_ = zeros(size(Vs)); ci = mu_;
    for i = 1:numel(Vs)
        [s, u] = getm(R, 'E1', m{1}, Vs(i), 'util');
        [~, ia, ib] = intersect(s, sb);
        d = 100 * (u(ia) ./ ub(ib) - 1);
        [mu_(i), ci(i)] = mean_ci(d);
    end
    plot_ci(ax, Vs, mu_, ci, col.(m{1}), mk.(m{1}), strcmp(m{1}, 'tinysem_hard'), lab.(m{1}));
end
[sc, uc] = getm(R, 'E1', 'cmarl', 0, 'util');
[~, ia, ib] = intersect(sc, sb);
yline(ax, 0, ':', 'Color', col.static, 'LineWidth', 1);
yline(ax, mean(100 * (uc(ia) ./ ub(ib) - 1)), ':', 'Color', col.cmarl, 'LineWidth', 1);
set(ax, 'XScale', 'log'); grid(ax, 'on');
xlabel(ax, 'Lyapunov parameter V'); ylabel(ax, 'Utility vs. static (%)');
title(ax, '(a)');

% ---------------- (b) hit ratio vs cache size under drift
ax = nexttile(tl); hold(ax, 'on');
for m = {'tinysem', 'tinysem_hard', 'static', 'cmarl', 'lru_maxacc'}
    [xs, mu_, ci] = sweep(R, 'E2', m{1}, 'hit');
    plot_ci(ax, 100 * xs, 100 * mu_, 100 * ci, col.(m{1}), mk.(m{1}), ...
            strcmp(m{1}, 'tinysem_hard'), lab.(m{1}));
end
grid(ax, 'on'); xlabel(ax, 'Cache size C (% of catalog volume)'); ylabel(ax, 'Hit ratio (%)');
title(ax, '(b)');

% ---------------- (c) worst-user energy over budget
ax = nexttile(tl); hold(ax, 'on');
for m = {'tinysem', 'static', 'cmarl', 'lru_maxacc', 'agnostic'}
    [xs, mu_, ci] = sweep(R, 'E3', m{1}, 'energy_ratio_max');
    plot_ci(ax, 100 * xs, mu_, ci, col.(m{1}), mk.(m{1}), false, lab.(m{1}));
end
[xs, mu_, ci] = sweep(R, 'E3V2', 'tinysem', 'energy_ratio_max');
plot_ci(ax, 100 * xs, mu_, ci, col.tinysem, 'o', true, 'Tiny-SEM (V=2)');
yline(ax, 1, '--k');
set(ax, 'YScale', 'log'); ylim(ax, [0.5 12]); grid(ax, 'on');
xlabel(ax, 'Budget (% of a_n e_{max})'); ylabel(ax, 'Worst-user energy / budget');
title(ax, '(c)');

% ---------------- (d) contraction factor: bound vs measured
ax = nexttile(tl); hold(ax, 'on');
Gm = R.game;
styles = {500, '-', '4 Gb/s'; 100, '--', '0.8 Gb/s'};
for i = 1:2
    sel = [Gm.Rbh] == styles{i,1} & [Gm.N] == 50;
    taus = unique([Gm(sel).tau]);
    kb = arrayfun(@(t) mean([Gm(sel & [Gm.tau] == t).kbound]), taus);
    ke = arrayfun(@(t) max([Gm(sel & [Gm.tau] == t).kemp]), taus);
    plot(ax, taus, kb, styles{i,2}, 'Color', col.tinysem, 'LineWidth', 1.1, ...
         'DisplayName', ['bound \kappa, ' styles{i,3}]);
    plot(ax, taus, ke, styles{i,2}, 'Color', col.static, 'Marker', 'o', 'MarkerSize', 3, ...
         'LineWidth', 1.1, 'DisplayName', ['measured, ' styles{i,3}]);
end
yline(ax, 1, ':k', 'HandleVisibility', 'off');
set(ax, 'XScale', 'log', 'YScale', 'log'); grid(ax, 'on');
xlabel(ax, 'Temperature \tau'); ylabel(ax, 'Contraction factor');
legend(ax, 'Location', 'northeast', 'FontSize', 5);
title(ax, '(d)');
exportgraphics(fig, 'fig_eval.pdf', 'ContentType', 'vector');

% ---------------- Fig. 3: drift time series
fig2 = figure('Units', 'inches', 'Position', [1 1 3.45 1.7], 'Color', 'w');
ax = axes(fig2); hold(ax, 'on');
names = {'tinysem', 'abl_cum', 'cmarl', 'lru_maxacc', 'static'};
dl = {'Tiny-SEM (EMA)', 'Tiny-SEM, cumulative est.', 'C-MARL', 'LRU-MA', 'Static Stackelberg'};
dc = {col.tinysem, col.tinysem, col.cmarl, col.lru_maxacc, col.static};
ls = {'-', '--', '-', '-', '-'};
rec = struct();
for i = 1:numel(names)
    H = drift_series(R, names{i});
    sm = movmean(H, 50);
    plot(ax, 1:numel(sm), 100 * sm, ls{i}, 'Color', dc{i}, 'LineWidth', 1, 'DisplayName', dl{i});
    rec.(names{i}) = recovery(H);
end
xline(ax, 1000:1000:4000, ':', 'Color', [0.54 0.54 0.52], 'HandleVisibility', 'off');
xlim(ax, [500 5000]); ylim(ax, [0 40]); grid(ax, 'on');
xlabel(ax, 'Slot t (popularity redrawn every 1000 slots)'); ylabel(ax, 'Hit ratio (%)');
legend(ax, 'Location', 'northoutside', 'NumColumns', 3, 'FontSize', 5);
exportgraphics(fig2, 'fig_drift.pdf', 'ContentType', 'vector');

% ---------------- Table I and key numbers
fprintf('\nTABLE I  (drift, mean over seeds; 95%% Student-t CI half-width in brackets)\n');
fprintf('%-20s %7s %6s %6s %8s %13s %6s\n', 'Method', 'Util', 'Hit%', 'Corr', 'Backhaul', 'Lat mean/p95', 'E/Eb');
labels = {'tinysem','tinysem_hard','static','cmarl','lru_maxacc','agnostic', ...
          'abl_nofb','abl_cum','abl_lru','abl_maxacc_greedy'};
for i = 1:numel(labels)
    [~, u] = getm(R, 'TAB', labels{i}, 0, 'util');
    [~, hh] = getm(R, 'TAB', labels{i}, 0, 'hit');
    [~, cpa] = getm(R, 'TAB', labels{i}, 0, 'correct_per_active');
    [~, bh] = getm(R, 'TAB', labels{i}, 0, 'backhaul');
    [~, lm] = getm(R, 'TAB', labels{i}, 0, 'lat_mean');
    [~, lp] = getm(R, 'TAB', labels{i}, 0, 'lat_p95');
    [~, em] = getm(R, 'TAB', labels{i}, 0, 'energy_ratio_max');
    [mu_, ci] = mean_ci(u);
    fprintf('%-20s %5.3f[%.3f] %5.1f %6.3f %8.1f %6.0f/%-6.0f %6.2f\n', labels{i}, mu_, ci, ...
            100*mean(hh), mean(cpa), mean(bh), 1000*mean(lm), 1000*mean(lp), mean(em));
end
fprintf('\nPaired gains of Tiny-SEM under drift (mean [95%% CI]):\n');
for b = {'static', 'cmarl', 'lru_maxacc'}
    for f = {'hit', 'backhaul', 'lat_mean', 'util'}
        [s1, a] = getm(R, 'TAB', 'tinysem', 0, f{1});
        [s2, c] = getm(R, 'TAB', b{1}, 0, f{1});
        [~, ia, ib] = intersect(s1, s2);
        [mu_, ci] = mean_ci(100 * (a(ia) ./ c(ib) - 1));
        fprintf('  vs %-11s %-9s %+7.1f%% [%.1f]\n', b{1}, f{1}, mu_, ci);
    end
end
[~, ab] = getm(R, 'TAB', 'tinysem', 0, 'Abar');
P = tinysem_params();
m_ = min(P.beta / (P.G - 1), (1 - P.beta) / (P.O - P.G));
A = mean(ab);
fprintf('\nRequest-weighted accuracy %.3f -> LDP epsilon %.2f (uniform errors: %.2f)\n', ...
        A, log(A / ((1 - A) * m_)), log(A * (P.O - 1) / (1 - A)));
[~, sg] = getm(R, 'TAB', 'tinysem', 0, 'sig_bits');
fprintf('Signaling: %.1f bits/slot of cache deltas\n', mean(sg));
fprintf('\nRecovery to 90%% of pre-change hit ratio (slots, per change):\n');
fn = fieldnames(rec);
for i = 1:numel(fn), fprintf('  %-12s %s\n', fn{i}, mat2str(rec.(fn{i}))); end
sel = [Gm.Rbh] == 500 & [Gm.N] == 50;
fprintf('\nGame (4 Gb/s, N=50): max spread %.1e; max measured contraction %.3g\n', ...
        max([Gm(sel).spread]), max([Gm(sel).kemp]));
for t = [0.01 0.2]
    fprintf('  welfare(SO lower bound)/welfare(NE) at tau=%.2f: %.4f\n', t, ...
            mean([Gm(sel & [Gm.tau] == t).eff]));
end
end

% ======================================================================
function [seeds, vals] = getm(R, ex, label, x, field)
seeds = []; vals = [];
for j = 1:numel(R.jobs)
    jb = R.jobs{j};
    if strcmp(jb{1}, ex) && strcmp(jb{3}, label) && abs(jb{2} - x) < 1e-12
        seeds(end+1) = jb{4};             %#ok<AGROW>
        vals(end+1) = R.res{j}.(field);   %#ok<AGROW>
    end
end
[seeds, o] = sort(seeds);  vals = vals(o);
end

function xs = unique_x(R, ex, label)
xs = [];
for j = 1:numel(R.jobs)
    jb = R.jobs{j};
    if strcmp(jb{1}, ex) && strcmp(jb{3}, label), xs(end+1) = jb{2}; end %#ok<AGROW>
end
xs = unique(xs);
end

function [xs, mu_, ci] = sweep(R, ex, label, field)
xs = unique_x(R, ex, label);
mu_ = zeros(size(xs)); ci = mu_;
for i = 1:numel(xs)
    [~, v] = getm(R, ex, label, xs(i), field);
    [mu_(i), ci(i)] = mean_ci(v);
end
end

function [m, ci] = mean_ci(v)
v = v(:);  n = numel(v);  m = mean(v);
if n < 2, ci = 0; return; end
ci = tinv_975(n - 1) * std(v) / sqrt(n);
end

function q = tinv_975(df)
% 97.5% Student-t quantile (no Statistics Toolbox needed): bisection on the CDF.
f = @(t) 1 - 0.5 * betainc(df ./ (df + t.^2), df/2, 0.5) - 0.975;
lo = 0; hi = 50;
for k = 1:80, mid = (lo + hi)/2; if f(mid) > 0, hi = mid; else, lo = mid; end, end
q = (lo + hi) / 2;
end

function plot_ci(ax, x, y, ci, c, m, hollow, name)
fill(ax, [x(:); flipud(x(:))], [y(:) - ci(:); flipud(y(:) + ci(:))], c, ...
     'FaceAlpha', 0.15, 'EdgeColor', 'none', 'HandleVisibility', 'off');
if hollow, fc = 'w'; ls = '--'; else, fc = c; ls = '-'; end
plot(ax, x, y, ls, 'Color', c, 'Marker', m, 'MarkerSize', 3.5, 'MarkerFaceColor', fc, ...
     'LineWidth', 1.1, 'DisplayName', name);
end

function H = drift_series(R, label)
H = [];
for j = 1:numel(R.jobs)
    jb = R.jobs{j};
    if strcmp(jb{1}, 'DRIFT') && strcmp(jb{3}, label)
        H = [H, R.res{j}.hit_t(:)];   %#ok<AGROW>
    end
end
H = mean(H, 2);
end

function r = recovery(H)
% Slots after each popularity change until the 50-slot moving average
% regains 90% of the mean hit ratio of the 200 slots before the change.
sm = movmean(H, 50);
r = nan(1, 3);  ks = [2000 3000 4000];
for i = 1:3
    k = ks(i);
    if numel(H) < k + 1000, continue; end
    pre = mean(H(k-199:k));
    seg = sm(k+26:k+1000);
    idx = find(seg >= 0.9 * pre, 1);
    if ~isempty(idx), r(i) = idx + 25; end
end
end
