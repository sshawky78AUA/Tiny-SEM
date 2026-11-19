function best = tinysem_game_social_opt(G, mu, rs, restarts, sweeps)
%TINYSEM_GAME_SOCIAL_OPT Best pure profile found by coordinate ascent on welfare.
%   This is a LOWER BOUND on the social optimum (local search, several restarts).
if nargin < 4, restarts = 5; end
if nargin < 5, sweeps = 30; end
P = G.P;  N = P.N;  K = P.K;
I = eye(K + 1);
best = -inf;
for r = 1:restarts
    k = randi(rs, [0, K], N, 1);
    for s = 1:sweeps
        changed = false;
        for n = randperm(rs, N)
            vals = zeros(K + 1, 1);
            for kk = 0:K
                k2 = k;  k2(n) = kk;
                vals(kk + 1) = tinysem_game_welfare(G, I(k2 + 1, :), mu);
            end
            [~, kb] = max(vals);
            if kb - 1 ~= k(n), k(n) = kb - 1; changed = true; end
        end
        if ~changed, break; end
    end
    best = max(best, tinysem_game_welfare(G, I(k + 1, :), mu));
end
end
