function T = tinysem_sweep(param, values, methods, seeds, varargin)
%TINYSEM_SWEEP Run several methods over a range of one parameter and print a table.
%   T = tinysem_sweep(param, values, methods, seeds) runs every method in
%   METHODS (cell array of labels accepted by tinysem_method) for each value
%   in VALUES of the parameter PARAM (a field of tinysem_params) and each seed,
%   prints the mean and 95% confidence half-width of the main metrics, and
%   returns one row per run in the struct array T.
%   Extra name/value pairs set other parameters for every run.
%
%   Examples
%     tinysem_sweep('cache_frac', [0.05 0.1 0.2], {'tinysem', 'static', 'cmarl'}, 0:4);
%     tinysem_sweep('V', [2 10 50], {'tinysem'}, 0:2, 'drift_every', 1000);
%     tinysem_sweep('', [], {'tinysem', 'lru_maxacc'}, 0:4, 'N', 100);  % no sweep, just compare
if nargin < 3 || isempty(methods), methods = {'tinysem'}; end
if nargin < 4 || isempty(seeds), seeds = 0; end
if isempty(param), values = NaN; end
if ischar(methods), methods = {methods}; end
keys = {'util', 'hit', 'sem', 'energy_ratio', 'energy_ratio_max', 'backhaul', 'lat_mean'};
T = [];
fprintf('%-14s %-12s', 'value', 'method');
fprintf('%20s', keys{:}); fprintf('\n');
for v = values
    for i = 1:numel(methods)
        args = varargin;
        if ~isempty(param), args = [args, {param, v}]; end
        P = tinysem_params(args{:});
        [method, opts] = tinysem_method(methods{i});
        vals = zeros(numel(seeds), numel(keys));
        for s = 1:numel(seeds)
            o = tinysem_run(P, seeds(s), method, opts{:});
            row = struct('param', param, 'value', v, 'method', methods{i}, 'seed', seeds(s));
            for k = 1:numel(keys)
                vals(s, k) = o.(keys{k});
                row.(keys{k}) = o.(keys{k});
            end
            if isempty(T), T = row; else, T(end + 1) = row; end %#ok<AGROW>
        end
        fprintf('%-14s %-12s', num2str(v), methods{i});
        for k = 1:numel(keys)
            [m, h] = mean_ci(vals(:, k));
            fprintf('%20s', sprintf('%.4g +/- %.2g', m, h));
        end
        fprintf('\n');
    end
end
end

function [m, h] = mean_ci(v)
m = mean(v);
n = numel(v);
if n < 2, h = 0; return; end
t = [12.706 4.303 3.182 2.776 2.571 2.447 2.365 2.306 2.262 2.228];
if n - 1 <= numel(t), q = t(n - 1); else, q = 1.96; end
h = q * std(v) / sqrt(n);
end
