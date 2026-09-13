"""
Report figures for the radar measurement characterization (2026-09-03):

  ladder.pdf  (fig:ladder)  the residual distribution, step by step
    (a) raw per-return Doppler residuals, fast racing, with N(0, sigma_rob)
    (b) normal Q-Q of the deployed S3 clean stream, all three flights
    (c) tail mass beyond k*sigma relative to a Gaussian, fast racing, for
        one global sigma / the per-return law on the clean stream / the law
        on the deployed S3 stream
  law.pdf     (fig:law)     the per-return noise law
    (a-c) robust width of the clean-stream residual binned by |v|, theta, phi
    (d)   leave-one-factor-out whitening adequacy, fast racing (the protocol
          of notebook 12 section 4: binned robust fit, then the robust scale
          of d_j / sigma_j along each law input)

Data: the estimator-free per-return tables of
analysis/characterize_shape_contradiction.py (cached under .cache/shape),
the same residuals notebook 12 uses; the consensus prefilter is the deployed
reve-style 3-point RANSAC (threshold 0.15 m/s, seeded).

Usage: python3 gen_characterization.py   (from report/figures/)
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats as sp_stats
from scipy.optimize import least_squares

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'analysis'))
import characterize_shape_contradiction as C          # noqa: E402
import paper_style                                    # noqa: E402
from paper_style import TEXTWIDTH_IN                  # noqa: E402

paper_style.apply()

BAGS = [('slow racing', 'slow_racing_best_velocity', '#0072B2', 'o'),
        ('fast racing', 'fast_racing_best_velocity', '#D55E00', 's'),
        ('backflips',   'backflips_best_velocity',   '#009E73', '^')]
FAST = 'fast_racing_best_velocity'
RACING = ['slow_racing_best_velocity', 'fast_racing_best_velocity']
S0_DEP, COEF_DEP = 0.0827, 0.03291          # deployed DIRCOS constants
COS_FLOOR = 0.15
RANSAC_T = 0.15


def robust_sigma(x):
    x = np.asarray(x)
    return 1.4826 * np.median(np.abs(x - np.median(x)))


def ransac_mask(T, rng, iters=150):
    """Deployed reve-style 3-point ego-velocity consensus on the unwrapped
    Doppler, per frame; frames with fewer than five returns bypass."""
    keep = np.zeros(len(T['r']), bool)
    H = np.column_stack([T['sx'], T['sy'], T['sz']])
    for f in np.unique(T['fid']):
        idx = np.flatnonzero(T['fid'] == f)
        if len(idx) < 5:
            keep[idx] = True
            continue
        Hf, vf = H[idx], T['unw'][idx]
        best = None
        for _ in range(iters):
            s = rng.choice(len(idx), 3, replace=False)
            try:
                vc = np.linalg.solve(Hf[s], vf[s])
            except np.linalg.LinAlgError:
                continue
            inl = np.abs(Hf @ vc - vf) < RANSAC_T
            if best is None or inl.sum() > best.sum():
                best = inl
        keep[idx] = best if (best is not None and best.sum() >= 5) else True
    return keep


def dircos_X(P):
    """X_j of the deployed law: transverse velocity in direction-cosine form."""
    cphi = np.maximum(P['sx'], COS_FLOOR)
    return np.sqrt((P['vsy'] - P['sy'] * P['vsx'] / cphi) ** 2
                   + (P['vsz'] - P['sz'] * P['vsx'] / cphi) ** 2)


def sigma_law(P, s0=S0_DEP, coef=COEF_DEP):
    return np.sqrt(s0 ** 2 + (coef * dircos_X(P)) ** 2)


def tail_ratio(z, ks):
    z = (z - np.median(z)) / robust_sigma(z)
    out = []
    for k in ks:
        emp = np.mean(np.abs(z) > k)
        gau = 2 * (1 - sp_stats.norm.cdf(k))
        out.append(emp / gau)
    return np.array(out)


def mixture2(w, iters=500):
    """Zero-mean two-component Gaussian mixture by EM (deterministic init from
    the robust sigma), notebook 12 cell 16.  Fitted on the whitened residual
    r/sigma_j; returns (weight of the wide component, sigma_core, sigma_wide)."""
    w = np.asarray(w, float); w = w[np.isfinite(w)]
    s1, s2, p = robust_sigma(w), 5 * robust_sigma(w), 0.9
    for _ in range(iters):
        a = p * np.exp(-0.5 * (w / s1) ** 2) / s1
        b = (1 - p) * np.exp(-0.5 * (w / s2) ** 2) / s2
        g = a / np.maximum(a + b, 1e-300)
        p = g.mean()
        s1 = np.sqrt(max(np.sum(g * w ** 2) / max(np.sum(g), 1e-12), 1e-12))
        s2 = np.sqrt(max(np.sum((1 - g) * w ** 2) / max(np.sum(1 - g), 1e-12), 1e-12))
    return 1 - p, s1, s2


def ladder_stream(T, extra_keep=None):
    """Per-return view for the LADDER panels under the deployed convention
    (2026-09-07 thin-frame entry): non-aliased returns of EVERY frame, no
    6-sigma clip, no frame-size cut; thin frames pass the prefilter whole
    exactly as the front end lets them.  Only what sigma_law() needs."""
    m = ~T['alias']
    if extra_keep is not None:
        m &= extra_keep
    fid = T['fid'][m]
    P = {k: T[k][m] for k in ('r', 'sx', 'sy', 'sz')}
    for c, nm in zip(range(3), ('vsx', 'vsy', 'vsz')):
        P[nm] = T['f_vs3'][fid, c]
    return P


def binned_sigma(x, r, edges):
    c, s = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (x >= lo) & (x < hi)
        if m.sum() >= 25:
            c.append(x[m].mean()); s.append(robust_sigma(r[m]))
    return np.array(c), np.array(s)


# ------------------------------------------------------------ load
tabs = {key: C.table(key) for _, key, _, _ in BAGS}
rng = np.random.default_rng(0)
S1 = {key: C.point_view(tabs[key], alias_cut=True, speed_cut=False, min_n=5)
      for key in tabs}
RM = {key: ransac_mask(tabs[key], rng) for key in tabs}
S3 = {key: C.point_view(tabs[key], alias_cut=True, speed_cut=False, min_n=5,
                        extra_keep=RM[key]) for key in tabs}
# ladder streams, deployed convention (S1/S3 above keep the fit-set hygiene
# for the law panels: 6-sigma clip, N_f >= 5, as characterize_pointwise_noise)
L1 = {key: ladder_stream(tabs[key]) for key in tabs}
L3 = {key: ladder_stream(tabs[key], extra_keep=RM[key]) for key in tabs}

print('ladder numbers, deployed convention (thin frames pass through; no clip)')
print(f"{'flight':<12} {'stream':<18} {'n':>5} {'sig_rob':>8} {'tail>3s':>8} "
      f"{'ratio@2s':>9} {'ratio@3s':>9}")
for label, key, _, _ in BAGS:
    for name, P, z in (('before, global', L1[key], L1[key]['r']),
                       ('before, law', L1[key], L1[key]['r'] / sigma_law(L1[key])),
                       ('after, law', L3[key], L3[key]['r'] / sigma_law(L3[key]))):
        zz = (z - np.median(z)) / robust_sigma(z)
        t2, t3 = tail_ratio(z, [2.0, 3.0])
        print(f"{label:<12} {name:<18} {len(z):>5} {robust_sigma(P['r']):>8.3f} "
              f"{100*np.mean(np.abs(zz) > 3):>7.1f}% {t2:>9.2f} {t3:>9.2f}")
    for name, P in (('before/law', L1[key]), ('after/law', L3[key])):
        w = P['r'] / sigma_law(P)
        pw, s1, s2 = mixture2(w - np.median(w))
        print(f"{label:<12} {'mixture on ' + name:<20} companion {100*pw:.1f}% of returns, "
              f"{s2/s1:.1f}x wider than the core (core {s1:.2f} law-sigma)")

# ============================================================ ladder.pdf
# Two panels (Timo, 2026-09-07): (a) the standardized residual histogram
# before and after everything, (b) the tail-mass attribution.  Fast racing.
fig, axes = plt.subplots(1, 2, figsize=(TEXTWIDTH_IN, 2.4))

# (a) before / after everything, in units of sigma, log density
ax = axes[0]
r_raw = tabs[FAST]['r']
z_raw = (r_raw - np.median(r_raw)) / robust_sigma(r_raw)
P3 = L3[FAST]
z_dep = P3['r'] / sigma_law(P3)
z_dep = z_dep - np.median(z_dep)
edges = np.linspace(-6, 6, 61)
ax.hist(z_raw, bins=edges, density=True, histtype='step', color='0.3', lw=0.9,
        label='raw, $r/\\sigma_{\\mathrm{rob}}$')
ax.hist(z_dep, bins=edges, density=True, histtype='stepfilled', color='#009E73',
        alpha=0.45, edgecolor='#009E73', lw=0.9,
        label='deployed stream, $r/\\sigma_j$')
x = np.linspace(-6, 6, 400)
ax.plot(x, sp_stats.norm.pdf(x), color='#D55E00', lw=0.9, label='$N(0,1)$')
ax.set_yscale('log'); ax.set_ylim(4e-4, 3.0); ax.set_xlim(-6, 6)
ax.set_xlabel('standardized residual ($\\sigma$)')
ax.set_ylabel('density')
ax.set_title('(a) residuals, fast racing')
ax.legend(loc='upper right', handlelength=1.4, borderpad=0.3, fontsize=7)

# (b) tail mass ratio, fast racing
ax = axes[1]
ks = np.linspace(0.5, 4.0, 36)
P1 = L1[FAST]
ax.plot(ks, tail_ratio(P1['r'], ks), color='0.45', label='before prefilter, global $\\sigma$')
ax.plot(ks, tail_ratio(P1['r'] / sigma_law(P1), ks), color='#0072B2',
        label='before prefilter, law $\\sigma_j$')
ax.plot(ks, tail_ratio(P3['r'] / sigma_law(P3), ks), color='#009E73',
        label='after prefilter, law $\\sigma_j$')
ax.axhline(1, color='k', lw=0.6, ls='--')
ax.set_yscale('log')
ax.set_xlabel('threshold $k$ ($\\sigma$)')
ax.set_ylabel('mass beyond $k\\sigma$ / Gaussian')
ax.set_title('(b) tail mass, fast racing')
ax.legend(loc='upper left', handlelength=1.4, borderpad=0.3, fontsize=7)

fig.tight_layout(pad=0.4, w_pad=1.6)
out = Path(__file__).parent / 'ladder'
fig.savefig(f'{out}.pdf'); fig.savefig(f'{out}.png')
print(f'wrote {out}.pdf')
qq = {}
for label, key, _, _ in BAGS:
    z = L3[key]['r']; z = (z - np.median(z)) / robust_sigma(z); zc = z[np.abs(z) < 3]
    (_, _), (_, _, rv) = sp_stats.probplot(zc, dist='norm', plot=None)
    print(f'  Q-Q R^2 inside |z|<3, after prefilter, {label}: {rv**2:.4f}')

# ============================================================ law.pdf
V_EDGES = np.array([0, 0.75, 1.5, 2.25, 3, 4, 5, 6.5, 8])
TH_EDGES = np.array([0, 30, 50, 65, 80, 95, 110, 130, 180])
PHI_EDGES = np.array([0, 20, 35, 45, 55, 62, 70, 78, 88])

fig, axes = plt.subplots(1, 4, figsize=(TEXTWIDTH_IN, 1.85))
for label, key, color, mk in BAGS:
    P = S1[key]
    vb = np.column_stack([P['vbx'], P['vby'], P['vbz']])
    ub = np.column_stack([P['ux'], P['uy'], P['uz']])
    nv = np.maximum(np.linalg.norm(vb, axis=1), 1e-6)
    cos_th = np.clip(np.einsum('ij,ij->i', ub, vb) / nv, -1, 1)
    th = np.degrees(np.arccos(cos_th))
    phi = np.degrees(np.arccos(np.clip(P['sx'], -1, 1)))
    for ax, xval, edges in [(axes[0], P['v'], V_EDGES), (axes[1], th, TH_EDGES),
                            (axes[2], phi, PHI_EDGES)]:
        c, s = binned_sigma(xval, P['r'], edges)
        ax.plot(c, s, '-' + mk, color=color, ms=2.5, label=label)
axes[0].set_xlabel('$|\\mathbf{v}|$ (m/s)'); axes[0].set_title('(a) speed')
axes[1].set_xlabel('$\\theta$ (deg)');       axes[1].set_title('(b) angle to velocity')
axes[2].set_xlabel('$\\phi$ (deg)');         axes[2].set_title('(c) off boresight')
axes[0].set_ylabel('robust $\\sigma$ of residual (m/s)')
axes[0].legend(loc='upper left', handlelength=1.2, borderpad=0.2, fontsize=6)
for ax in axes[:3]:
    ax.set_ylim(0, 0.7)

# (d) leave-one-factor-out whitening, fast racing: the protocol of notebook 12
# section 4.  Per-return fit of sigma_j^2 = s0_bag^2 + (c X_j)^2 to the robust
# width of the within-frame deviation d_j in 10 quantile bins of the predictor,
# both racing flights jointly (per-bag floor, one shared coefficient, soft-L1
# loss); then the robust scale of d_j / sigma_j binned along each law input.
# A correct model gives 1 in every bin.
PT = {k: C.point_view(tabs[k]) for k in RACING}   # 6-sigma clip, alias cut, N_f >= 6, |v| >= 0.2


def pred_X(P):
    s1 = np.sqrt(P['sin2']); cphi = np.maximum(P['sx'], COS_FLOOR)
    return {'full': P['v'] * s1 / cphi, 'no |v|': s1 / cphi,
            'no sin': P['v'] / cphi, 'no 1/cos': P['v'] * s1}


def fit_pt(name, n_bins=10, min_n=120):
    c, s, b = [], [], []
    for bi, k in enumerate(RACING):
        X = pred_X(PT[k])[name]
        ed = np.unique(np.percentile(X, np.linspace(0, 100, n_bins + 1)))
        for lo, hi in zip(ed[:-1], ed[1:]):
            m = (X >= lo) & (X < hi)
            if m.sum() >= min_n:
                c.append(X[m].mean()); s.append(robust_sigma(PT[k]['d'][m])); b.append(bi)
    c, s, b = np.array(c), np.array(s), np.array(b, int)
    mdl = lambda th: np.sqrt(np.abs(th[b]) ** 2 + (np.abs(th[2]) * c) ** 2)
    fsc = max(1.4826 * np.median(np.abs(s - np.median(s))), 1e-4)
    sol = least_squares(lambda th: mdl(th) - s, np.r_[[np.median(s)] * 2, [1e-3]],
                        loss='soft_l1', f_scale=fsc, max_nfev=40000)
    res = mdl(sol.x) - s
    r2 = 1 - float(res @ res) / float(((s - s.mean()) ** 2).sum())
    return np.abs(sol.x), r2


fits = {name: fit_pt(name) for name in ('full', 'no |v|', 'no sin', 'no 1/cos')}
Pf = PT[FAST]; i_f = RACING.index(FAST)
STRATA = [('$|\\mathbf{v}|$', Pf['v'], 'no |v|',
           np.array([0.2, 1.0, 1.75, 2.5, 3.25, 4.0, 5.0, 7.0])),
          ('$\\sin\\theta$', np.sqrt(Pf['sin2']), 'no sin',
           np.array([0.0, 0.3, 0.5, 0.65, 0.8, 0.9, 1.0])),
          ('$\\phi$', np.degrees(np.arccos(np.clip(Pf['sx'], -1, 1))), 'no 1/cos',
           np.array([0, 25, 40, 50, 60, 90]))]
ax = axes[3]
xoff, ticks, ticklabels, spreads = 0, [], [], {}
for label, xval, missing, ED in STRATA:
    for name, color, mk, lab in [(missing, '#7B52AB', 's', 'law missing that factor'),
                                 ('full', '#2CA02C', '^', 'full law')]:
        th, _ = fits[name]
        w = Pf['d'] / np.sqrt(th[i_f] ** 2 + (th[2] * pred_X(Pf)[name]) ** 2)
        cells = []
        for lo, hi in zip(ED[:-1], ED[1:]):
            m = (xval >= lo) & (xval < hi)
            if m.sum() >= 60:
                cells.append(robust_sigma(w[m]))
        cells = np.array(cells)
        spreads[(label, name)] = float(cells.max() / cells.min())
        ax.plot(xoff + np.arange(len(cells)), cells, '-' + mk, color=color,
                ms=2.5, label=lab if xoff == 0 else None)
    ticks.append(xoff + (len(ED) - 2) / 2); ticklabels.append(label)
    xoff += len(ED) + 1
ax.axhline(1, color='k', lw=0.6, ls='--')
ax.set_xticks(ticks); ax.set_xticklabels(ticklabels)
ax.set_ylabel('robust scale of $d_j/\\sigma_j$')
ax.set_title('(d) whitening test')
ax.set_ylim(0.55, 1.75)
ax.legend(loc='upper left', handlelength=1.2, borderpad=0.2, fontsize=6)

fig.tight_layout(pad=0.3, w_pad=0.8)
out = Path(__file__).parent / 'law'
fig.savefig(f'{out}.pdf'); fig.savefig(f'{out}.png')
print(f'wrote {out}.pdf')
for name, (th, r2) in fits.items():
    print(f'  fit {name:<9} R2={r2:.3f}  coef={th[2]:.5f}  floors slow/fast={th[0]:.3f}/{th[1]:.3f}'
          f'  dphi0={np.degrees(th[2] * np.sqrt(2)):.1f} deg')
for (label, name), sp in spreads.items():
    print(f'  whitening spread fast racing along {label:<16} {name:<9} {sp:.2f}x')
