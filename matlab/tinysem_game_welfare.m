function Wf = tinysem_game_welfare(G, p, mu)
%TINYSEM_GAME_WELFARE Unregularized social welfare of a follower profile p,
%   sum_n a_n sum_{k>=1} p_{n,k} (u_{n,k}(G_{-n}) - mu_n e_k / e_max).
P = G.P;  W = G.W;
g = 1 - W.a .* sum(p(:, 2:end) .* G.c, 2);
Gm = prod(g) ./ g;
u = P.lam1 * P.acc + P.lam2 * (G.h + G.w .* Gm);
val = sum(p(:, 2:end) .* (u - mu .* G.en), 2);
Wf = sum(W.a .* val);
end
