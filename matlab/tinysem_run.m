function out = tinysem_run(P, seed, method, varargin)
%TINYSEM_RUN Simulate one run of Tiny-SEM or a baseline.
%
%   out = tinysem_run(P, seed, method, 'Option', value, ...)
%
%   method : 'tinysem'         Tiny-SEM (logit drift-plus-penalty + EMA/greedy leader)
%            'static_stackel'  Static Stackelberg baseline (offline LP, frozen cache)
%            'cmarl'           Constrained MARL (Lagrangian Q-learning + decayed LFU)
%            'lru_maxacc'      LRU caching + most accurate model, token-bucket throttled
%            'maxacc_agnostic' most accurate model always, Tiny-SEM cache, no budget
%   options: 'feedback'  (true)   use the congestion broadcast (Tiny-SEM)
%            'hard'      (false)  hard argmax instead of logit (Tiny-SEM-H)
%            'cache'     ('')     override cache: 'greedy' | 'lru' | 'lfu' | 'static'
%            'estimator' ('ema')  demand estimator: 'ema' | 'cum'
%            'alpha_mu'  (0.5)    dual step of C-MARL (paper uses 0.02)
%
%   out : struct of scalar metrics plus time series hit_t, sem_t.

opt = struct('feedback', true, 'hard', false, 'cache', '', ...
             'estimator', 'ema', 'alpha_mu', 0.5);
for i = 1:2:numel(varargin), opt.(varargin{i}) = varargin{i+1}; end

W  = tinysem_world(P, seed);
rs = RandStream('mt19937ar', 'Seed', seed + 10000);   % dynamics randomness
N = P.N;  O = P.O;  K = P.K;
e  = P.energy(:)';          % 1 x K (mJ)
eK = e(end);
en = e / eK;                % energy in units of e_max (controller units)
eps_ = P.eps;
b  = W.b;
expb = exp(-eps_ * b);
acc = P.acc(:)';            % 1 x K

if opt.hard, tau = 1e-4; else, tau = P.tau; end
defaults = struct('tinysem','greedy','lru_maxacc','lru','static_stackel','static', ...
                  'cmarl','lfu','maxacc_agnostic','greedy');
cache_kind = defaults.(method);
if ~isempty(opt.cache), cache_kind = opt.cache; end

% ---------------------------------------------------------------- leader state
Qhat = ones(O, 1) / O;
cum_counts = zeros(O, 1);
lfu = zeros(O, 1);
lru_last = zeros(O, 1);  lru_in = false(O, 1);  lru_used = 0;  lru_clock = 0;
x = tinysem_greedy_cache(Qhat + 1e-9 * rand(rs, O, 1), b, W.C);
d_hat = 1.0;
cache_updates = 0;
sig_bits = 0;

% ---------------------------------------------------------------- follower state
Z = zeros(N, 1);
[h, w] = user_terms(x);
static_p = [];
nbins = 5;
Qtab = zeros(N, nbins, K + 1);
mu = zeros(N, 1);
spent = zeros(N, 1);

% ---------------------------------------------------------------- logs
Tm = P.T - P.warm;
log_hit = nan(Tm, 1);  log_sem = nan(Tm, 1);  log_acc = nan(Tm, 1);
log_phi = nan(Tm, 1);  log_r = zeros(Tm, 1);  log_eta = zeros(Tm, 1);
log_Z = zeros(Tm, 1);  log_nreq = zeros(Tm, 1);  log_nact = zeros(Tm, 1);
lat_all = cell(Tm, 1);
E_cum = zeros(N, 1);  Zsum = zeros(N, 1);
Abar_sum = 0;  Abar_cnt = 0;
hit_t = zeros(P.T, 1);  sem_t = zeros(P.T, 1);

for t = 0:P.T-1
    % ------------------------------------------------ demand drift
    if P.drift_every > 0 && t > 0 && mod(t, P.drift_every) == 0
        W = tinysem_make_demand(W);
        if ~strcmp(cache_kind, 'lru'), [h, w] = user_terms(x); end
    end
    active = rand(rs, N, 1) < W.a;

    % ------------------------------------------------ follower decisions
    switch method
        case 'tinysem'
            if opt.feedback
                Gm = d_hat * ones(N, 1);
                psi = h + w;
                for it = 1:2             % G_{-n} = d_hat / g_n(p_n)
                    p = logit_policy(Gm);
                    g = 1 - W.a .* sum(p(:, 2:end) .* (1 - psi), 2);
                    Gm = min(max(d_hat ./ g, 0), 1);
                end
            else
                p = logit_policy(ones(N, 1));
            end
            k = sample_action(p);
        case 'lru_maxacc'
            allow = spent + eK <= W.Ebar * (t + 1) + eK;     % token bucket
            k = K * allow;
        case 'maxacc_agnostic'
            k = K * ones(N, 1);
        case 'static_stackel'
            if isempty(static_p), static_p = solve_static(h, w); end
            k = sample_action(static_p);
        case 'cmarl'
            sbin = min(floor(d_hat * nbins), nbins - 1) + 1;
            epsx = max(0.02, 0.5 / sqrt(1 + t / 100));
            [~, ga] = max(squeeze(Qtab(:, sbin, :)), [], 2);  % greedy action (1-based)
            explore = rand(rs, N, 1) < epsx;
            ra = randi(rs, [0, K], N, 1);
            k = explore .* ra + (~explore) .* (ga - 1);
        otherwise
            error('tinysem_run:method', 'Unknown method %s', method);
    end

    % ------------------------------------------------ requests and delivery
    req = active & (k >= 1);
    users = find(req);
    ks = k(users);
    En = zeros(N, 1);  En(users) = e(ks);
    r = zeros(N, 1);  phi = zeros(N, 1);
    eta = 0;
    if ~isempty(users)
        [ostar, rep] = tinysem_sample_requests(W, users, ks, rs);
        if strcmp(cache_kind, 'lru'), xh = double(lru_in(rep)); else, xh = x(rep); end
        eta = sum((1 - xh) .* b(rep));
        ph = xh + (1 - xh) * exp(-eps_ * eta);
        phi(users) = ph;
        r(users) = P.lam1 * acc(ks)' + P.lam2 * ph;
        correct = rep == ostar;
        hfade = -log(rand(rs, numel(users), 1));            % Exp(1) Rayleigh power
        Ra = P.W * log2(1 + W.snr(users) .* hfade) / 8e6;   % MB/s
        lat = b(rep) ./ max(Ra, 1e-3) + (1 - xh) * eta / P.Rbh;
        if t >= P.warm
            i = t - P.warm + 1;
            log_hit(i) = mean(xh);  log_sem(i) = mean(xh .* correct);
            log_acc(i) = mean(correct);  lat_all{i} = lat;
        end
        hit_t(t+1) = mean(xh);  sem_t(t+1) = mean(xh .* correct);
        if strcmp(cache_kind, 'lru')
            for j = 1:numel(rep), lru_access(rep(j)); end
        end
        cnt = accumarray(rep, 1, [O 1]);
        if strcmp(opt.estimator, 'ema')
            Qhat = (1 - P.gamma) * Qhat + P.gamma * cnt / numel(rep);
        end
        cum_counts = cum_counts + cnt;
        lfu = 0.999 * lfu + cnt;
    end
    if strcmp(opt.estimator, 'cum') && sum(cum_counts) > 0
        Qhat = cum_counts / sum(cum_counts);
    end
    d_hat = (1 - P.gamma_d) * d_hat + P.gamma_d * exp(-eps_ * eta);

    % ------------------------------------------------ queues / duals
    Z = max(Z + En - (1 - P.margin) * W.Ebar, 0);
    spent = spent + En;
    E_cum = E_cum + En;
    if strcmp(method, 'cmarl')
        rew = r - mu .* En;
        sb = min(floor(d_hat * nbins), nbins - 1) + 1;
        idx = find(active);
        lin = sub2ind(size(Qtab), idx, sb * ones(size(idx)), k(idx) + 1);
        Qtab(lin) = Qtab(lin) + 0.1 * (rew(idx) - Qtab(lin));
        mu = max(mu + opt.alpha_mu * (En - W.Ebar) / eK, 0);
    end

    % ------------------------------------------------ cache update
    if strcmp(cache_kind, 'greedy') || (strcmp(cache_kind, 'static') && t == P.static_freeze)
        xn = tinysem_greedy_cache(Qhat, b, W.C);
        if ~isequal(xn, x)
            sig_bits = sig_bits + sum(abs(xn - x)) * ceil(log2(O));
            x = xn;  cache_updates = cache_updates + 1;
            [h, w] = user_terms(x);
        end
    elseif strcmp(cache_kind, 'lfu') && mod(t, 10) == 0
        xn = tinysem_greedy_cache(lfu, b, W.C);
        if ~isequal(xn, x), x = xn; cache_updates = cache_updates + 1; end
    end
    if strcmp(method, 'static_stackel') && t == P.static_freeze
        static_p = [];                 % re-solve the offline LP for the frozen cache
    end

    % ------------------------------------------------ logging
    if t >= P.warm
        i = t - P.warm + 1;
        if any(req), log_phi(i) = mean(phi(req)); end
        log_r(i) = sum(r) / N;
        log_eta(i) = eta;
        log_Z(i) = mean(Z);
        log_nreq(i) = sum(req);  log_nact(i) = sum(active);
        Zsum = Zsum + Z;
        if strcmp(method, 'tinysem')
            pa = p(:, 2:end);
            Abar = sum(pa .* acc, 2) ./ max(sum(pa, 2), 1e-12);
            Abar_sum = Abar_sum + mean(Abar);  Abar_cnt = Abar_cnt + 1;
        end
    end
end

avgE = E_cum / P.T;                     % energy over the whole horizon
lat_all = vertcat(lat_all{:});
if isempty(lat_all), lat_all = nan; end
out.hit = mean(log_hit, 'omitnan');
out.sem = mean(log_sem, 'omitnan');
out.acc = mean(log_acc, 'omitnan');
out.phi = mean(log_phi, 'omitnan');
out.util = mean(log_r);
out.energy_ratio = mean(avgE ./ W.Ebar);
out.energy_ratio_max = max(avgE ./ W.Ebar);
out.viol_frac = mean(avgE > W.Ebar * 1.001);
out.backhaul = mean(log_eta);
out.lat_mean = mean(lat_all);
out.lat_p95 = prctile(lat_all, 95);
out.Zbar = mean(log_Z);
out.req_rate = mean(log_nreq) / N;
out.cache_updates = cache_updates / P.T;
out.sig_bits = sig_bits / P.T;
out.serve = sum(log_nreq) / max(sum(log_nact), 1);
out.correct_per_active = out.serve * out.acc;
out.Z_user = Zsum / Tm;
out.hit_t = hit_t;
out.sem_t = sem_t;
if Abar_cnt > 0, out.Abar = Abar_sum / Abar_cnt; else, out.Abar = nan; end

% ====================================================================== nested
    function [h_, w_] = user_terms(xc)
        h_ = tinysem_noisy_expect(W, xc, acc, P.beta);
        w_ = tinysem_noisy_expect(W, (1 - xc) .* expb, acc, P.beta);
    end

    function pp = logit_policy(Gm_)
        % p_{n,k} ∝ exp(u_{n,k}/tau - Z_n e_k/(V tau)), action 0 = skip (u=0, e=0)
        u_ = P.lam1 * acc + P.lam2 * (h + w .* Gm_);                 % N x K
        s_ = [zeros(N, 1), u_ / tau - (Z / eK) .* en / (P.V * tau)];
        s_ = s_ - max(s_, [], 2);
        pp = exp(s_);
        pp = pp ./ sum(pp, 2);
    end

    function kk = sample_action(pp)
        cp = cumsum(pp, 2);
        kk = sum(cp < rand(rs, N, 1), 2);       % 0 = skip, 1..K = model
        kk = min(kk, K);
    end

    function pp = solve_static(h_, w_)
        % maximize sum_k p_k u_k(G=1) s.t. a_n sum_k p_k e_k <= Ebar_n
        u_ = P.lam1 * acc + P.lam2 * (h_ + w_);
        pp = zeros(N, K + 1);
        ee = [0, e];
        for n = 1:N
            cap = W.Ebar(n) / W.a(n);
            uu = [0, u_(n, :)];
            best_v = 0;  best_p = [1, zeros(1, K)];
            for ii = 1:K+1
                for jj = 1:K+1
                    if ee(ii) <= cap && cap <= ee(jj) && ee(jj) > ee(ii)
                        lam = (cap - ee(ii)) / (ee(jj) - ee(ii));
                        v = (1 - lam) * uu(ii) + lam * uu(jj);
                        if v > best_v
                            vec = zeros(1, K + 1);
                            vec(ii) = vec(ii) + 1 - lam;  vec(jj) = vec(jj) + lam;
                            best_v = v;  best_p = vec;
                        end
                    end
                    if ee(jj) <= cap && uu(jj) > best_v
                        vec = zeros(1, K + 1);  vec(jj) = 1;
                        best_v = uu(jj);  best_p = vec;
                    end
                end
            end
            pp(n, :) = best_p;
        end
    end

    function lru_access(o)
        lru_clock = lru_clock + 1;
        if lru_in(o)
            lru_last(o) = lru_clock;
            return
        end
        if b(o) > W.C, return; end
        while lru_used + b(o) > W.C
            cand = find(lru_in);
            [~, j] = min(lru_last(cand));
            v = cand(j);
            lru_in(v) = false;  lru_used = lru_used - b(v);
        end
        lru_in(o) = true;  lru_last(o) = lru_clock;  lru_used = lru_used + b(o);
    end
end
