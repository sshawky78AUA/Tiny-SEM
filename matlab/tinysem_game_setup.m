function G = tinysem_game_setup(P, seed, x)
%TINYSEM_GAME_SETUP Static follower game for a fixed cache (Theorem 1 checks).
%   G = tinysem_game_setup(P, seed)      uses the greedy cache of the mean demand.
%   G = tinysem_game_setup(P, seed, x)   uses the given cache vector x.
%   Returns h, w, c (N x K), normalized energies en (1 x K), the world W,
%   and cbar = max_{n,k} c_{n,k}.

W = tinysem_world(P, seed);
b = W.b;
if nargin < 3 || isempty(x)
    x = tinysem_greedy_cache(mean(W.q, 1)', b, W.C);
end
expb = exp(-P.eps * b);
G.W  = W;
G.P  = P;
G.x  = x;
G.en = P.energy / P.energy(end);
G.h  = tinysem_noisy_expect(W, x, P.acc, P.beta);
G.w  = tinysem_noisy_expect(W, (1 - x) .* expb, P.acc, P.beta);
G.c  = 1 - G.h - G.w;
G.cbar = max(G.c(:));
end
