import sys, json, os
import numpy as np
from multiprocessing import Pool
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from tinysem import (Params, run, game_setup, solve_equilibrium, welfare,
                     social_opt)

TAUS = [0.01, 0.02, 0.05, 0.1, 0.2, 0.4, 0.8]
SEEDS = list(range(5))


def kappa_bound(P, world):
    cbar = 1 - np.exp(-P.eps * world.b.max())
    amax = world.a.max()
    return lambda tau: P.lam2 * (P.N - 1) * amax * cbar / (4 * tau * (1 - amax * cbar))


def one(args):
    Rbh, N, seed = args
    P = Params(Rbh=Rbh, N=N, T=3000, warm=1000)
    o = run(P, seed, 'tinysem')
    mu = o['Z_user'] / (P.energy[-1] * P.V)
    world, en, h, w = game_setup(P, seed)
    kb = kappa_bound(P, world)
    rng = np.random.default_rng(seed + 77)
    rows = []
    for tau in TAUS:
        finals = []
        kemp = 0.0
        resid = 0.0
        for r in range(30):
            p0 = rng.dirichlet(np.ones(len(P.acc) + 1), size=N)
            p, ke, res = solve_equilibrium(P, world, en, h, w, mu, tau, p0)
            finals.append(p)
            kemp = max(kemp, ke)
            resid = max(resid, res)
        ref = finals[0]
        spread = max(np.abs(f - ref).max() for f in finals)
        W_ne = welfare(P, world, en, h, w, ref, mu)
        rows.append(dict(tau=tau, spread=float(spread), kemp=float(kemp),
                         kbound=float(kb(tau)), resid=float(resid),
                         W_ne=float(W_ne)))
    W_so = social_opt(P, world, en, h, w, mu, rng)
    for r in rows:
        r['eff'] = float(W_so / r['W_ne'])
    return dict(Rbh=Rbh, N=N, seed=seed, rows=rows, W_so=float(W_so))


if __name__ == '__main__':
    jobs = [(R, 50, s) for R in [500.0, 100.0] for s in SEEDS]
    jobs += [(500.0, N, s) for N in [10, 20, 30, 40] for s in SEEDS]
    with Pool(2) as pool:
        out = pool.map(one, jobs)
    json.dump(out, open(os.path.join(HERE, 'game_results.json'), 'w'))
    print('done')
