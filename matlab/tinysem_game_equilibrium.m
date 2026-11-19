function [p, kappa_emp, resid] = tinysem_game_equilibrium(G, mu, tau, p0, iters)
%TINYSEM_GAME_EQUILIBRIUM Jacobi best-response iteration of the logit follower game.
%   mu  : N x 1 energy prices Z_n/(V e_max)    tau : temperature
%   p0  : N x (K+1) initial mixed strategies   iters : number of iterations (400)
%   Returns the final profile, the largest measured contraction ratio of
%   y_n = -log g_n between successive iterations, and the final step size.

if nargin < 5, iters = 400; end
P = G.P;  W = G.W;
p = p0;
ys = zeros(iters, P.N);
for it = 1:iters
    g = 1 - W.a .* sum(p(:, 2:end) .* G.c, 2);
    y = -log(g);
    ys(it, :) = y';
    Gm = exp(-(sum(y) - y));                     % G_{-n} = prod_{m~=n} g_m
    p = logit_br(G, Gm, mu, tau);
end
d = max(abs(diff(ys, 1, 1)), [], 2);             % sup-norm step sizes
ratios = d(2:end) ./ max(d(1:end-1), 1e-300);
good = d(1:end-1) > 1e-13;
r = ratios(good);
if isempty(r), kappa_emp = 0; else, kappa_emp = max(r(1:min(50, numel(r)))); end
resid = d(end);
end

function p = logit_br(G, Gm, mu, tau)
P = G.P;
u = P.lam1 * P.acc + P.lam2 * (G.h + G.w .* Gm);
s = [zeros(P.N, 1), (u - mu .* G.en) / tau];
s = s - max(s, [], 2);
p = exp(s);
p = p ./ sum(p, 2);
end
