import sys, json, time, itertools, os
import numpy as np
from multiprocessing import Pool
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from tinysem import Params, run

T, WARM = 5000, 1000
SEEDS_SWEEP = list(range(10))
SEEDS_TAB = list(range(20))
KEYS = ['hit', 'sem', 'util', 'serve', 'correct_per_active', 'energy_ratio',
        'energy_ratio_max', 'viol_frac', 'backhaul', 'lat_mean', 'lat_p95',
        'Zbar', 'sig_bits', 'cache_updates', 'acc', 'phi']

METHODS = {
    'tinysem':      ('tinysem', {}),
    'tinysem_hard': ('tinysem', {'hard': True}),
    'static':       ('static_stackel', {}),
    'cmarl':        ('cmarl', {'alpha_mu': 0.02}),
    'lru_maxacc':   ('lru_maxacc', {}),
    'agnostic':     ('maxacc_agnostic', {}),
    # ablations
    'abl_nofb':     ('tinysem', {'feedback': False}),
    'abl_cum':      ('tinysem', {'estimator': 'cum'}),
    'abl_lru':      ('tinysem', {'cache': 'lru'}),
    'abl_maxacc_greedy': ('lru_maxacc', {'cache': 'greedy'}),
}


def job(args):
    exp, x, mname, seed, pkw = args
    meth, opt = METHODS[mname]
    P = Params(T=T, warm=WARM, **pkw)
    o = run(P, seed, meth, **opt)
    res = {k: float(o[k]) for k in KEYS}
    if 'Abar' in o:
        res['Abar'] = float(o['Abar'])
    return exp, x, mname, seed, res


def main():
    jobs = []
    # E1: utility/backlog vs V (stationary)
    for V in [1, 2, 5, 10, 20, 50, 100]:
        for s in SEEDS_SWEEP:
            jobs.append(('E1', V, 'tinysem', s, {'V': V}))
            jobs.append(('E1', V, 'tinysem_hard', s, {'V': V}))
    for s in SEEDS_SWEEP:
        for m in ['static', 'cmarl', 'lru_maxacc']:
            jobs.append(('E1', 0, m, s, {}))
    # E2: hit/latency vs cache size (drift)
    for cf in [0.02, 0.05, 0.10, 0.15, 0.20]:
        for m in ['tinysem', 'tinysem_hard', 'static', 'cmarl', 'lru_maxacc']:
            for s in SEEDS_SWEEP:
                jobs.append(('E2', cf, m, s, {'cache_frac': cf, 'drift_every': 1000}))
    # E3: energy compliance vs budget (stationary)
    for bf in [0.1, 0.2, 0.4, 0.6, 0.8, 1.0]:
        for m in ['tinysem', 'static', 'cmarl', 'lru_maxacc', 'agnostic']:
            for s in SEEDS_SWEEP:
                jobs.append(('E3', bf, m, s, {'budget_frac': bf}))
    # TAB: default drift scenario, all methods + ablations
    for m in METHODS:
        for s in SEEDS_TAB:
            jobs.append(('TAB', 0, m, s, {'drift_every': 1000}))
            jobs.append(('TABS', 0, m, s, {}))     # stationary counterpart
    print(len(jobs), 'jobs', flush=True)
    t0 = time.time()
    out = []
    with Pool(2) as pool:
        for i, r in enumerate(pool.imap_unordered(job, jobs, chunksize=4)):
            out.append(r)
            if i % 100 == 0:
                print(i, round(time.time() - t0), flush=True)
    json.dump(out, open(os.path.join(HERE, 'results.json'), 'w'))
    print('done', round(time.time() - t0))


if __name__ == '__main__':
    main()
