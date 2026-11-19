function [ostar, rep] = tinysem_sample_requests(W, users, ks, rs)
%TINYSEM_SAMPLE_REQUESTS Draw intents and (possibly mislabeled) reports.
%   users : indices of requesting users,  ks : their models (1..K).
%   ostar : true intents (1-based object ids),  rep : reported labels.

P = W.P;
G = P.G;  O = P.O;
users = users(:);  ks = ks(:);
m = numel(users);
u = rand(rs, m, 1);
ostar = sum(W.qcum(users, :) < u, 2) + 1;      % inverse-CDF sampling
ostar = min(ostar, O);
acc = P.acc(ks);  acc = acc(:);
rep = ostar;
wrong = rand(rs, m, 1) >= acc;
if any(wrong)
    idx  = find(wrong);
    o0   = ostar(idx) - 1;                       % 0-based ids
    base = floor(o0 / G) * G;                    % first id of the cluster
    same = rand(rs, numel(idx), 1) < P.beta;
    off  = randi(rs, [1, G-1], numel(idx), 1);
    cand_same = base + mod(o0 - base + off, G);  % other object, same cluster
    oc   = randi(rs, [0, O-G-1], numel(idx), 1);
    cand_out = oc + G * (oc >= base);            % object of another cluster
    rep(idx) = (same .* cand_same + (~same) .* cand_out) + 1;
end
end
