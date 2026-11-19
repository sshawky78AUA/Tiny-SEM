function E = tinysem_noisy_expect(W, f, acc, beta, q)
%TINYSEM_NOISY_EXPECT  E_{n,k}[f] for every user n and model k  (N x K).
%   Mean of f over the label user n reports with model k, under the
%   clustered confusion model: correct w.p. A_k; otherwise a uniformly
%   chosen other object of the same cluster w.p. beta, or a uniformly
%   chosen object of another cluster w.p. 1-beta.  Cost O(N*O).

if nargin < 5, q = W.q; end
P = W.P;
G = P.G;  O = P.O;
f = f(:);
S_cl  = accumarray(W.cl, f, [W.nc 1]);       % cluster sums of f
S_tot = sum(f);
qf  = q * f;                                  % correct label
inn = q * (S_cl(W.cl) - f) / (G - 1);         % same cluster, other object
out = q * (S_tot - S_cl(W.cl)) / (O - G);     % other cluster
noise = beta * inn + (1 - beta) * out;
acc = acc(:)';                                % 1 x K
E = qf * acc + noise * (1 - acc);
end
