function W = tinysem_world(P, seed)
%TINYSEM_WORLD Static randomness of one run: sizes, users, channels, demand.
%   The world depends only on the seed, so all methods see the same
%   topology and demand for a given seed (common random numbers).

W.rs = RandStream('mt19937ar', 'Seed', seed);
rs = W.rs;
W.P  = P;
W.b  = P.b_lo + (P.b_hi - P.b_lo) * rand(rs, P.O, 1);     % object sizes (MB)
W.cl = floor((0:P.O-1)' / P.G) + 1;                         % cluster index (1-based)
W.nc = P.O / P.G;
W.a  = P.a_lo + (P.a_hi - P.a_lo) * rand(rs, P.N, 1);       % activity probabilities
W.snr = 10 .^ ((P.snr_lo + (P.snr_hi - P.snr_lo) * rand(rs, P.N, 1)) / 10);
W.C  = P.cache_frac * sum(W.b);
W    = tinysem_make_demand(W);
W.Ebar = P.budget_frac * W.a * P.energy(end);               % energy budgets (mJ/slot)
end
