function W = tinysem_make_demand(W)
%TINYSEM_MAKE_DEMAND Draw intent distributions q_n (N x O).
%   q_n = (1-omega) * global Zipf + omega * personal Zipf, each over a
%   random permutation of the catalog.  Called again at every demand drift.

P = W.P;
glob = zipf_pmf(P.O, P.zipf, W.rs);
q = zeros(P.N, P.O);
for n = 1:P.N
    q(n, :) = (1 - P.omega) * glob + P.omega * zipf_pmf(P.O, P.zipf, W.rs);
end
W.q = q;
W.qcum = cumsum(q, 2);
W.qcum(:, end) = 1;
end

function p = zipf_pmf(O, s, rs)
w = (1:O) .^ (-s);
w = w / sum(w);
p = zeros(1, O);
p(randperm(rs, O)) = w;
end
