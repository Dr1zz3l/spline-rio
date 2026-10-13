#!/usr/bin/env python3
"""Simulate what the IWR6843AOP reports for the bearing of an ALIASED return
(2026-09-16).  Geometry from TI SWRU546E Fig. 4-8 (lambda = 5 mm): RX1..4 a
2x2 square at lambda/2 (RX3 bottom-left, RX1 bottom-right, RX4 top-left, RX2
top-right); TX3 at the origin, TX1 lambda above it, TX2 lambda to its right.
Virtual element position = TX + RX.  Chirp order (best_velocity.cfg chirpCfg
0/1/2 = TX1/TX2/TX3).  TDM Doppler compensation with the WRAPPED Doppler
leaves the residual phase 2 pi k t / 3 on the elements of the t-th TX in the
chirp order (TX1: 0, TX2: 2 pi k/3, TX3: 4 pi k/3) for alias order k.

Reported bearing = peak of the 2D Bartlett spectrum (a zero-padded 2D angle
FFT) over the direction cosines (cx = azimuth, cz = elevation) in [-1, 1],
which is what the SDK's 2D angle estimation does up to its grid.  Printed:
for k = 0, +1, -1 and a grid of true bearings, the reported (cx', cz'), the
shift, the peak height relative to the unaliased peak, and whether a second
lobe within 3 dB exists (ambiguous pick).
"""
import numpy as np

lam = 1.0
rx = {'RX3': (0.0, 0.0), 'RX1': (0.5, 0.0), 'RX4': (0.0, 0.5), 'RX2': (0.5, 0.5)}
tx = {'TX1': (0.0, 1.0), 'TX2': (1.0, 0.0), 'TX3': (0.0, 0.0)}
order = ['TX1', 'TX2', 'TX3']
pos, grp = [], []
for t in order:
    for r in rx.values():
        pos.append((tx[t][0] + r[0], tx[t][1] + r[1])); grp.append(order.index(t))
pos = np.array(pos); grp = np.array(grp)
print("virtual elements (x, z) in lambda:", sorted(set(map(tuple, pos))))

G = np.linspace(-1, 1, 401)
CX, CZ = np.meshgrid(G, G, indexing='ij')
steer = np.exp(-1j * 2 * np.pi * (pos[:, 0][:, None, None] * CX[None] + pos[:, 1][:, None, None] * CZ[None]))   # (12, ncx, ncz)
valid = CX ** 2 + CZ ** 2 <= 1.0


def report(cx, cz, k):
    sig = np.exp(1j * 2 * np.pi * (pos[:, 0] * cx + pos[:, 1] * cz)) * np.exp(1j * 2 * np.pi * k * grp / 3.0)
    spec = np.abs(np.einsum('n,nij->ij', sig, steer)) ** 2
    spec[~valid] = 0
    i, j = np.unravel_index(np.argmax(spec), spec.shape)
    pk = spec[i, j]
    # second lobe: local maxima within 3 dB but > 0.15 in cosine away
    from scipy.ndimage import maximum_filter
    loc = (spec == maximum_filter(spec, size=9)) & (spec > 0.5 * pk) & valid
    ii, jj = np.where(loc)
    far = [(G[a], G[b], spec[a, b] / pk) for a, b in zip(ii, jj) if abs(G[a] - G[i]) + abs(G[b] - G[j]) > 0.15]
    return G[i], G[j], pk / 144.0, far


print("\ntrue (cx, cz) -> reported (cx', cz'); shift; peak/ideal; other lobes within 3 dB")
for k in (0, 1, -1):
    print(f"== k = {k:+d}")
    for cx in (-0.6, -0.3, 0.0, 0.3, 0.6):
        for cz in (-0.6, -0.3, 0.0, 0.3, 0.6):
            if cx * cx + cz * cz > 0.8: continue
            rx_, rz_, pk, far = report(cx, cz, k)
            print(f"   ({cx:+.1f},{cz:+.1f}) -> ({rx_:+.3f},{rz_:+.3f})  shift ({rx_-cx:+.3f},{rz_-cz:+.3f})  peak {pk:.2f}  lobes: " + ', '.join(f"({a:+.2f},{b:+.2f}) {c:.2f}" for a, b, c in far))
# statistics of the shift over a dense grid of true bearings inside the 60 deg cone
print("\nshift statistics over |cx|,|cz| <= 0.7 (k = +1 / -1): mean, std of (dcx, dcz); fraction with a competing lobe")
for k in (1, -1):
    sh = []; amb = 0; n = 0
    for cx in np.linspace(-0.7, 0.7, 29):
        for cz in np.linspace(-0.7, 0.7, 29):
            if cx * cx + cz * cz > 0.7 ** 2: continue
            rx_, rz_, pk, far = report(cx, cz, k); sh.append((rx_ - cx, rz_ - cz)); amb += len(far) > 0; n += 1
    sh = np.array(sh)
    print(f"   k = {k:+d}: mean shift ({sh[:,0].mean():+.3f}, {sh[:,1].mean():+.3f}), std ({sh[:,0].std():.3f}, {sh[:,1].std():.3f}), competing lobe in {100*amb/n:.0f} % of bearings")
