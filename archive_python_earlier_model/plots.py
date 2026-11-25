import json, sys, os
HERE = os.path.dirname(os.path.abspath(__file__))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict

R = json.load(open(os.path.join(HERE, 'results.json')))
GR = json.load(open(os.path.join(HERE, 'game_results.json')))

COL = {'tinysem': '#2a78d6', 'tinysem_hard': '#2a78d6', 'static': '#eb6834',
       'cmarl': '#1baf7a', 'lru_maxacc': '#eda100', 'agnostic': '#e87ba4'}
MK = {'tinysem': 'o', 'tinysem_hard': 'o', 'static': 's', 'cmarl': '^',
      'lru_maxacc': 'D', 'agnostic': 'v'}
LS = {'tinysem_hard': '--'}
LAB = {'tinysem': r'Tiny-SEM ($\tau$=0.2)', 'tinysem_hard': r'Tiny-SEM-H ($\tau\to0$)',
       'static': 'Static Stackelberg', 'cmarl': 'C-MARL', 'lru_maxacc': 'LRU-MA',
       'agnostic': 'Energy-agnostic'}

plt.rcParams.update({'font.size': 7, 'font.family': 'serif', 'axes.linewidth': 0.5,
                     'xtick.major.width': 0.5, 'ytick.major.width': 0.5,
                     'axes.edgecolor': '#52514e', 'axes.labelcolor': '#0b0b0b',
                     'xtick.color': '#52514e', 'ytick.color': '#52514e',
                     'legend.frameon': False, 'grid.color': '#e4e3df',
                     'grid.linewidth': 0.4})


def agg(exp, key):
    d = defaultdict(list)
    for e, x, m, s, res in R:
        if e == exp:
            d[(m, x)].append(res[key])
    out = {}
    for k, v in d.items():
        v = np.array(v)
        out[k] = (v.mean(), 1.96 * v.std(ddof=1) / np.sqrt(len(v)))
    return out


def series(a, m):
    xs = sorted(x for (mm, x) in a if mm == m)
    return np.array(xs), np.array([a[(m, x)][0] for x in xs]), np.array([a[(m, x)][1] for x in xs])


def line(ax, x, y, ci, m, **kw):
    ax.plot(x, y, color=COL[m], marker=MK[m], ms=3.2, lw=1.1, ls=LS.get(m, '-'),
            label=LAB[m], mfc='white' if m == 'tinysem_hard' else COL[m], mew=0.8, **kw)
    ax.fill_between(x, y - ci, y + ci, color=COL[m], alpha=0.15, lw=0)


fig, axs = plt.subplots(1, 4, figsize=(7.16, 1.75))

# (a) utility vs V
ax = axs[0]
U = agg('E1', 'util')
for m in ['tinysem', 'tinysem_hard']:
    x, y, c = series(U, m); line(ax, x, y, c, m)
for m in ['static', 'cmarl']:
    v, c = U[(m, 0)]
    ax.axhline(v, color=COL[m], lw=0.9, ls=':')
    ax.text(1.05, v + 0.0015, LAB[m], color='#52514e', fontsize=5.5, va='bottom')
ax.set_xscale('log'); ax.set_xlabel('Lyapunov parameter $V$')
ax.set_ylabel('Utility (reward/user/slot)'); ax.grid(True)
ax.set_title('(a) Utility vs. $V$', fontsize=7)
ax2 = ax.inset_axes([0.55, 0.12, 0.42, 0.38])
Zb = agg('E1', 'Zbar')
x, y, c = series(Zb, 'tinysem')
ax2.plot(x, y, color='#52514e', marker='o', ms=2, lw=0.8)
ax2.set_xscale('log'); ax2.set_yscale('log'); ax2.tick_params(labelsize=4.5, length=1.5)
ax2.set_title(r'backlog $\bar Z$ (mJ)', fontsize=5, pad=1)

# (b) hit ratio vs cache size (drift)
ax = axs[1]
Hh = agg('E2', 'hit')
for m in ['tinysem', 'tinysem_hard', 'static', 'cmarl', 'lru_maxacc']:
    x, y, c = series(Hh, m); line(ax, 100 * x, 100 * y, 100 * c, m)
ax.set_xlabel('Cache size $C$ (% of catalog volume)'); ax.set_ylabel('Hit ratio (%)')
ax.grid(True); ax.set_title('(b) Hit ratio under demand drift', fontsize=7)

# (c) energy compliance
ax = axs[2]
Em = agg('E3', 'energy_ratio_max')
for m in ['tinysem', 'static', 'cmarl', 'lru_maxacc', 'agnostic']:
    x, y, c = series(Em, m); line(ax, 100 * x, y, c, m)
ax.axhline(1.0, color='#0b0b0b', lw=0.8, ls='--')
ax.text(12, 1.03, 'budget', fontsize=5.5, color='#52514e')
ax.set_yscale('log'); ax.set_ylim(0.6, 12)
ax.set_xlabel(r'Budget $\bar E_n$ (% of $a_n e_{\max}$)')
ax.set_ylabel('Worst-user energy / budget'); ax.grid(True)
ax.set_title('(c) Energy compliance', fontsize=7)

# (d) uniqueness: kappa bound vs empirical, two backhaul regimes
ax = axs[3]
for Rbh, ls, lab in [(500.0, '-', '4 Gb/s'), (100.0, '--', '0.8 Gb/s')]:
    runs = [g for g in GR if g['Rbh'] == Rbh and g['N'] == 50]
    taus = [r['tau'] for r in runs[0]['rows']]
    kb = np.mean([[r['kbound'] for r in g['rows']] for g in runs], 0)
    ke = np.max([[r['kemp'] for r in g['rows']] for g in runs], 0)
    ax.plot(taus, kb, color='#2a78d6', ls=ls, lw=1.1, label=r'bound $\kappa$, ' + lab)
    ax.plot(taus, ke, color='#eb6834', ls=ls, marker='o', ms=2.8, lw=1.1, label=r'measured, ' + lab)
ax.axhline(1, color='#0b0b0b', lw=0.8, ls=':')
ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlabel(r'Temperature $\tau$'); ax.set_ylabel('Contraction factor')
ax.grid(True); ax.set_title('(d) Equilibrium contraction', fontsize=7)
ax.legend(fontsize=5, loc='lower left', handlelength=2.2)

h, l = axs[1].get_legend_handles_labels()
h2, l2 = axs[2].get_legend_handles_labels()
for hh, ll in zip(h2, l2):
    if ll not in l:
        h.append(hh); l.append(ll)
fig.legend(h, l, loc='upper center', ncol=6, fontsize=6, bbox_to_anchor=(0.5, 1.06),
           handlelength=2.2, columnspacing=1.2)
fig.tight_layout(pad=0.3, w_pad=0.8)
fig.savefig(os.path.join(HERE, 'fig_eval.pdf'), bbox_inches='tight')
fig.savefig(os.path.join(HERE, 'fig_eval.png'), dpi=220, bbox_inches='tight')
print('ok')
