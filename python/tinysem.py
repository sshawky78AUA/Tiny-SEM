"""Tiny-SEM simulator (revised model for ICC 2027 resubmission).

Model summary (matches the revised paper):
  * N users, Bernoulli activity a_n; catalog of O objects (ImageNet-1k classes),
    sizes b_o ~ U[1,10] MB, grouped in semantic clusters of size G.
  * K TinyML models (MobileNetV1 width 0.25/0.5/0.75/1.0, Howard et al. 2017,
    Table 6): top-1 accuracy A_k, energy e_k = MACs_k * eta_E.  Action 0 = skip.
  * Confusion: correct w.p. A_k; else same-cluster w.p. beta, other cluster
    w.p. 1-beta (uniform within each set).
  * Missed volume eta(t); delivery quality phi = H + (1-H) exp(-eps*eta),
    eps = 1/(R_bh * Delta).  Reward r = lam1*A_k + lam2*phi (0 for skip/idle).
  * Tiny-SEM follower: logit (entropy-regularized) drift-plus-penalty choice
        p_k ∝ exp(u_k/tau - Z e_k/(V tau)),
        u_k = lam1 A_k + lam2 (h_k + w_k * G_minus_n),
    with G_minus_n = d_hat / g_n (product form).
  * Tiny-SEM leader: EMA demand estimate + density-greedy knapsack.
"""
import numpy as np
from collections import OrderedDict

MOBILENET_ACC = np.array([0.506, 0.637, 0.684, 0.706])   # Howard'17 Tab.6
MOBILENET_MMAC = np.array([41., 149., 325., 569.])        # million MACs


class Params:
    def __init__(self, **kw):
        self.N = 50
        self.O = 1000
        self.G = 10                # cluster size
        self.zipf = 0.8
        self.omega = 0.3           # personalization weight
        self.a_lo, self.a_hi = 0.3, 0.7
        self.b_lo, self.b_hi = 1.0, 10.0      # MB
        self.cache_frac = 0.06     # C as fraction of catalog volume
        self.acc = MOBILENET_ACC.copy()
        self.eta_E = 1e-3          # mJ per million MAC  (= 1 pJ/MAC)
        self.beta = 0.6            # same-cluster error mass
        self.lam1, self.lam2 = 1.0, 1.0
        self.Rbh = 250.0           # backhaul MB per slot (2 Gb/s, Delta = 1 s)
        self.W = 100e6             # per-user access bandwidth (Hz)
        self.snr_lo, self.snr_hi = 10.0, 25.0  # mean SNR dB
        self.budget_frac = 0.4     # E_bar_n = frac * a_n * e_max
        self.V = 50.0
        self.tau = 0.4
        self.gamma = 0.02          # EMA forgetting factor (demand)
        self.gamma_d = 0.1         # congestion EMA
        self.T = 5000
        self.warm = 1000
        self.drift_at = None       # slot of popularity shift (None = stationary)
        self.uplink_bits = int(np.ceil(np.log2(1000)))   # semantic request
        self.raw_bits = 224 * 224 * 3 * 8                 # raw RGB frame
        for k, v in kw.items():
            setattr(self, k, v)

    @property
    def energy(self):
        return MOBILENET_MMAC * self.eta_E      # mJ, models 1..K

    @property
    def eps(self):
        return 1.0 / self.Rbh


def zipf_pmf(O, s, rng):
    w = np.arange(1, O + 1, dtype=float) ** (-s)
    w /= w.sum()
    out = np.empty(O)
    out[rng.permutation(O)] = w
    return out


class World:
    """Static randomness of one seed: popularity, sizes, users, channels."""

    def __init__(self, P, seed):
        rng = np.random.default_rng(seed)
        self.rng = rng
        self.P = P
        O, N, G = P.O, P.N, P.G
        self.b = rng.uniform(P.b_lo, P.b_hi, O)
        self.cl = np.arange(O) // G
        self.nc = O // G
        self.a = rng.uniform(P.a_lo, P.a_hi, N)
        self.snr = 10 ** (rng.uniform(P.snr_lo, P.snr_hi, N) / 10)
        self.C = P.cache_frac * self.b.sum()
        self.make_demand(rng)
        self.Ebar = P.budget_frac * self.a * P.energy[-1]

    def make_demand(self, rng):
        P = self.P
        glob = zipf_pmf(P.O, P.zipf, rng)
        q = np.empty((P.N, P.O))
        for n in range(P.N):
            q[n] = (1 - P.omega) * glob + P.omega * zipf_pmf(P.O, P.zipf, rng)
        self.q = q
        self.qcum = np.cumsum(q, axis=1)
        self.qcum[:, -1] = 1.0


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def noisy_expect(world, f, acc, beta, q=None):
    """E_{n,k}[f(o_n)] for k=1..K under the clustered confusion model.
    Returns (N, K)."""
    P = world.P
    q = world.q if q is None else q
    G, O = P.G, P.O
    S_cl = np.bincount(world.cl, weights=f, minlength=world.nc)
    S_tot = f.sum()
    qf = q @ f
    inn = q @ (S_cl[world.cl] - f) / (G - 1)
    out = q @ (S_tot - S_cl[world.cl]) / (O - G)
    noise = beta * inn + (1 - beta) * out
    return acc[None, :] * qf[:, None] + (1 - acc[None, :]) * noise[:, None]


def greedy_cache(Qhat, b, C):
    order = np.argsort(-Qhat, kind="stable")
    x = np.zeros_like(b)
    used = 0.0
    for o in order:
        if used + b[o] <= C:
            x[o] = 1.0
            used += b[o]
        if C - used < 1.0:      # min object size
            break
    return x


def sample_requests(world, users, ks, rng):
    """Sample intents and (possibly corrupted) reports for active users."""
    P = world.P
    u = rng.random(len(users))
    ostar = (world.qcum[users] < u[:, None]).sum(1)
    ostar = np.minimum(ostar, P.O - 1)
    acc = P.acc[ks - 1]
    r = rng.random(len(users))
    rep = ostar.copy()
    wrong = r >= acc
    if wrong.any():
        idx = np.where(wrong)[0]
        same = rng.random(len(idx)) < P.beta
        G = P.G
        base = (ostar[idx] // G) * G
        # same cluster, different object
        off = rng.integers(1, G, len(idx))
        cand_same = base + (ostar[idx] - base + off) % G
        # other cluster
        oc = rng.integers(0, P.O - G, len(idx))
        cand_out = np.where(oc >= base, oc + G, oc)
        rep[idx] = np.where(same, cand_same, cand_out)
    return ostar, rep


class LRU:
    def __init__(self, b, C):
        self.b, self.C = b, C
        self.d = OrderedDict()
        self.used = 0.0

    def x(self, O):
        x = np.zeros(O)
        if self.d:
            x[list(self.d.keys())] = 1.0
        return x

    def access(self, o):
        if o in self.d:
            self.d.move_to_end(o)
            return
        if self.b[o] > self.C:
            return
        while self.used + self.b[o] > self.C:
            k, _ = self.d.popitem(last=False)
            self.used -= self.b[k]
        self.d[o] = True
        self.used += self.b[o]


# --------------------------------------------------------------------------
# main simulation
# --------------------------------------------------------------------------
def run(P, seed, method="tinysem", **opt):
    """method in {tinysem, lru_maxacc, static_stackel, cmarl, maxacc_agnostic}
    opt flags for ablations: feedback(bool), hard(bool), cache('greedy'|'lru'|'lfu'),
    estimator('ema'|'cum'), uniform_belief(bool)."""
    world = World(P, seed)
    rng = np.random.default_rng(seed + 10_000)
    N, O, K = P.N, P.O, len(P.acc)
    e = P.energy
    eK = e[-1]
    eps = P.eps
    b = world.b
    expb = np.exp(-eps * b)

    feedback = opt.get("feedback", True)
    hard = opt.get("hard", False)
    tau = P.tau if not hard else 1e-4
    estimator = opt.get("estimator", "ema")
    cache_kind = opt.get("cache", {"tinysem": "greedy", "lru_maxacc": "lru",
                                   "static_stackel": "static", "cmarl": "lfu",
                                   "maxacc_agnostic": "greedy"}[method])
    if method == "tinysem" and "cache" in opt:
        cache_kind = opt["cache"]

    # ---- leader state
    Qhat = np.full(O, 1.0 / O)
    cum_counts = np.zeros(O)
    lfu = np.zeros(O)
    lru = LRU(b, world.C)
    x = greedy_cache(Qhat + 1e-9 * rng.random(O), b, world.C)
    d_hat = 1.0
    cache_updates = 0

    # ---- follower state
    Z = np.zeros(N)
    acc = P.acc

    def user_terms(x):
        h = noisy_expect(world, x, acc, P.beta)
        w = noisy_expect(world, (1 - x) * expb, acc, P.beta)
        return h, w

    h, w = user_terms(x)

    # static Stackelberg: fixed policy from G=1 LP (budget in expectation)
    static_p = None
    # constrained MARL
    nbins = 5
    Qtab = np.zeros((N, nbins, K + 1))
    mu = np.zeros(N)
    alpha_mu = opt.get("alpha_mu", 0.5)
    # token bucket for throttled baselines
    spent = np.zeros(N)

    # ---- logs
    L = {k: [] for k in ["hit", "sem", "phi", "r", "E", "eta", "lat", "Z",
                         "act", "acc", "nreq"]}
    E_cum = np.zeros(N)
    r_cum = np.zeros(N)
    lat_all = []
    eps_ldp_num = 0.0
    eps_ldp_den = 0.0
    hit_t = np.zeros(P.T)
    sem_t = np.zeros(P.T)

    def solve_static(h, w):
        # maximize sum_k p_k u_k(G=1) s.t. a sum_k p_k e_k <= Ebar  (2-model mix)
        u = P.lam1 * acc[None, :] + P.lam2 * (h + w * 1.0)     # N x K
        p = np.zeros((N, K + 1))
        for n in range(N):
            cap = world.Ebar[n] / world.a[n]
            best = (-1, None)
            # pure feasible + mixes with skip / pairs
            cands = [np.r_[1.0, np.zeros(K)]]
            vals = [0.0]
            ee = np.r_[0.0, e]
            uu = np.r_[0.0, u[n]]
            for i in range(K + 1):
                for j in range(K + 1):
                    if ee[i] <= cap <= ee[j] and ee[j] > ee[i]:
                        lam = (cap - ee[i]) / (ee[j] - ee[i])
                        v = (1 - lam) * uu[i] + lam * uu[j]
                        vec = np.zeros(K + 1); vec[i] += 1 - lam; vec[j] += lam
                        cands.append(vec); vals.append(v)
                    if ee[j] <= cap:
                        vec = np.zeros(K + 1); vec[j] = 1
                        cands.append(vec); vals.append(uu[j])
            p[n] = cands[int(np.argmax(vals))]
        return p

    for t in range(P.T):
        if P.drift_at is not None and t == P.drift_at:
            world.make_demand(world.rng)
            if cache_kind != "lru":
                h, w = user_terms(x)

        active = rng.random(N) < world.a
        # ------------------------------------------------ follower decisions
        if method == "tinysem":
            g_self_cache = None
            psi = h + w
            if feedback:
                # own factor g_n under current logit policy is needed for G_{-n};
                # one fixed-point step using last d_hat
                G_minus = np.full(N, d_hat)
                for _ in range(2):
                    u = P.lam1 * acc[None, :] + P.lam2 * (h + w * G_minus[:, None])
                    s = np.c_[np.zeros(N), u / tau - Z[:, None] * e[None, :] / (P.V * tau)]
                    s -= s.max(1, keepdims=True)
                    p = np.exp(s); p /= p.sum(1, keepdims=True)
                    g = 1 - world.a * (p[:, 1:] * (1 - psi)).sum(1)
                    G_minus = np.clip(d_hat / g, 0, 1)
            else:
                u = P.lam1 * acc[None, :] + P.lam2 * (h + w * 1.0)
                s = np.c_[np.zeros(N), u / tau - Z[:, None] * e[None, :] / (P.V * tau)]
                s -= s.max(1, keepdims=True)
                p = np.exp(s); p /= p.sum(1, keepdims=True)
            cp = np.cumsum(p, 1)
            k = (cp < rng.random(N)[:, None]).sum(1)
            k = np.minimum(k, K)
        elif method in ("lru_maxacc",) or (method == "tinysem_maxacc"):
            k = np.full(N, K)
            allow = spent + eK <= world.Ebar * (t + 1) + eK
            k = np.where(allow, K, 0)
        elif method == "maxacc_agnostic":
            k = np.full(N, K)
        elif method == "static_stackel":
            if static_p is None:
                static_p = solve_static(h, w)
            cp = np.cumsum(static_p, 1)
            k = np.minimum((cp < rng.random(N)[:, None]).sum(1), K)
        elif method == "cmarl":
            sbin = min(int(d_hat * nbins), nbins - 1)
            epsx = max(0.02, 0.5 / np.sqrt(1 + t / 100))
            greedy_a = Qtab[:, sbin, :].argmax(1)
            explore = rng.random(N) < epsx
            k = np.where(explore, rng.integers(0, K + 1, N), greedy_a)
        else:
            raise ValueError(method)

        req = active & (k >= 1)
        users = np.where(req)[0]
        ks = k[users]
        En = np.zeros(N)
        En[users] = e[ks - 1]
        r = np.zeros(N)
        Hn = np.zeros(N)
        phi = np.zeros(N)
        if len(users):
            ostar, rep = sample_requests(world, users, ks, rng)
            if cache_kind == "lru":
                xh = np.array([1.0 if o in lru.d else 0.0 for o in rep])
            else:
                xh = x[rep]
            eta = ((1 - xh) * b[rep]).sum()
            ph = xh + (1 - xh) * np.exp(-eps * eta)
            Hn[users] = xh
            phi[users] = ph
            r[users] = P.lam1 * acc[ks - 1] + P.lam2 * ph
            correct = rep == ostar
            # latency: access download + backhaul drain for misses
            hfade = rng.exponential(1.0, len(users))
            Ra = P.W * np.log2(1 + world.snr[users] * hfade) / 8e6   # MB/s
            lat = b[rep] / np.maximum(Ra, 1e-3) + (1 - xh) * eta / P.Rbh
            if t >= P.warm:
                L["hit"].append(xh.mean()); L["sem"].append((xh * correct).mean())
                L["acc"].append(correct.mean()); lat_all.append(lat)
                # LDP of realised per-user mix ~ via mean accuracy of requests
            hit_t[t] = xh.mean(); sem_t[t] = (xh * correct).mean()
            # leader observes reports
            if cache_kind == "lru":
                for o in rep:
                    lru.access(int(o))
            Y = np.bincount(rep, minlength=O) / len(rep)
            if estimator == "ema":
                Qhat = (1 - P.gamma) * Qhat + P.gamma * Y
            cum_counts += np.bincount(rep, minlength=O)
            lfu = 0.999 * lfu + np.bincount(rep, minlength=O)
        else:
            eta = 0.0
        if estimator == "cum" and cum_counts.sum() > 0:
            Qhat = cum_counts / cum_counts.sum()

        d_hat = (1 - P.gamma_d) * d_hat + P.gamma_d * np.exp(-eps * eta)
        # ---------------------------------------------------- queues / duals
        Z = np.maximum(Z + En - world.Ebar, 0)
        spent += En
        if method == "cmarl":
            rew = r - mu * En
            sb = min(int(d_hat * nbins), nbins - 1)
            idx = np.where(active)[0]
            Qtab[idx, sb, k[idx]] += 0.1 * (rew[idx] - Qtab[idx, sb, k[idx]])
            mu = np.maximum(mu + alpha_mu * (En - world.Ebar) / eK, 0)

        # ---------------------------------------------------- cache update
        if cache_kind in ("greedy",) or (cache_kind == "static" and t == 300):
            xn = greedy_cache(Qhat, b, world.C)
            if not np.array_equal(xn, x):
                x = xn; cache_updates += 1
                h, w = user_terms(x)
        elif cache_kind == "lfu" and t % 10 == 0:
            xn = greedy_cache(lfu, b, world.C)
            if not np.array_equal(xn, x):
                x = xn; cache_updates += 1
        elif cache_kind == "lru":
            pass

        if t >= P.warm:
            L["phi"].append(phi[req].mean() if req.any() else np.nan)
            L["r"].append(r.sum() / N)
            L["E"].append(En.copy())
            L["eta"].append(eta)
            L["Z"].append(Z.mean())
            L["nreq"].append(req.sum())
            if method == "tinysem":
                pa = p[:, 1:]
                Abar = (pa * acc).sum(1) / np.maximum(pa.sum(1), 1e-12)
                eps_ldp_num += Abar.mean(); eps_ldp_den += 1

    Tm = P.T - P.warm
    E_arr = np.array(L["E"])                       # Tm x N
    avgE = E_arr.mean(0)
    lat_all = np.concatenate(lat_all) if lat_all else np.array([np.nan])
    out = dict(
        hit=np.nanmean(L["hit"]), sem=np.nanmean(L["sem"]),
        acc=np.nanmean(L["acc"]), phi=np.nanmean(L["phi"]),
        util=np.mean(L["r"]), energy_ratio=np.mean(avgE / world.Ebar),
        energy_ratio_max=np.max(avgE / world.Ebar),
        viol_frac=np.mean(avgE > world.Ebar * 1.001),
        backhaul=np.mean(L["eta"]), lat_mean=np.mean(lat_all),
        lat_p95=np.percentile(lat_all, 95), Zbar=np.mean(L["Z"]),
        req_rate=np.mean(L["nreq"]) / N,
        cache_updates=cache_updates / P.T,
        hit_t=hit_t, sem_t=sem_t,
    )
    if eps_ldp_den:
        out["Abar"] = eps_ldp_num / eps_ldp_den
    return out
