"""Derive the radar's measurement specification from its own .cfg file.

Every number the sensor model needs -- coherent processing interval, Doppler
bin, velocity ambiguity, range resolution, duty cycle -- follows from the
chirp configuration the driver uploads to the IWR6843.  Reading them off the
config is better than inferring them from the data, and it lets the inferred
values serve as an independent check instead of as the source.

The two configurations used in this work:
  6843AOP_best_velocity.cfg  the headline flights (fine Doppler, low ambiguity)
  6843AOP_3d.cfg             the older flights (coarse Doppler, high ambiguity)

Relations used (standard TDM-MIMO FMCW):
  T_chirp = idleTime + rampEndTime                    per transmitted chirp
  T_loop  = n_TX * T_chirp                            Doppler sample spacing
  T_CPI   = numLoops * T_loop                         coherent processing interval
  v_max   = lambda / (4 * T_loop)                     unambiguous velocity
  dv      = 2 * v_max / numLoops = lambda/(2*T_CPI)   Doppler bin
  B_valid = slope * (numAdcSamples / digOutSampleRate)
  dr      = c / (2 * B_valid)                         range resolution

Run from analysis/:  ../.venv/bin/python3 radar_config_spec.py
"""
import math
import re
import sys
from pathlib import Path

C_LIGHT = 299_792_458.0

CFG_DIR = (Path(__file__).resolve().parent.parent / 'mmwave_ti_ros'
           / 'ros1_driver' / 'src' / 'ti_mmwave_rospkg' / 'cfg')
PROFILES = {
    'best_velocity (headline flights)': '6843AOP_best_velocity.cfg',
    '3d (old-firmware flights)': '6843AOP_3d.cfg',
}


def parse_cfg(path):
    """Pull the chirp-timing commands out of a TI mmWave .cfg."""
    prof = chirps = frame = None
    clutter = None
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if line.startswith('profileCfg'):
            prof = [float(x) for x in line.split()[1:]]
        elif line.startswith('chirpCfg'):
            chirps = (chirps or 0) + 1
        elif line.startswith('frameCfg'):
            frame = [float(x) for x in line.split()[1:]]
        elif line.startswith('clutterRemoval'):
            clutter = int(line.split()[-1])
    if prof is None or frame is None:
        raise ValueError(f'{path}: missing profileCfg/frameCfg')
    return prof, chirps, frame, clutter


def spec(path):
    prof, n_chirp_cfg, frame, clutter = parse_cfg(path)
    # profileCfg: id start_ghz idle_us adcStart_us rampEnd_us txPow txPhase
    #             slope_mhz_per_us txStart_us numAdcSamples sampleRate_ksps ...
    start_ghz, idle_us, adc_start_us, ramp_end_us = prof[1], prof[2], prof[3], prof[4]
    slope = prof[7]                      # MHz/us
    n_adc, rate_ksps = prof[9], prof[10]
    # frameCfg: chirpStartIdx chirpEndIdx numLoops numFrames periodicity_ms ...
    chirp_start, chirp_end, n_loops, _, period_ms = frame[:5]
    n_tx = int(chirp_end - chirp_start + 1)

    adc_time_us = n_adc / rate_ksps * 1e3            # us
    b_valid_mhz = slope * adc_time_us
    dr = C_LIGHT / (2 * b_valid_mhz * 1e6)

    t_chirp = (idle_us + ramp_end_us) * 1e-6
    t_loop = n_tx * t_chirp
    t_cpi = n_loops * t_loop
    # carrier at the centre of the VALID sweep
    f_c = (start_ghz + slope * adc_start_us / 1e3 + b_valid_mhz / 2e3) * 1e9
    lam = C_LIGHT / f_c
    v_max = lam / (4 * t_loop)
    dv = 2 * v_max / n_loops
    return dict(n_tx=n_tx, n_tx_cfg=n_chirp_cfg, n_loops=int(n_loops),
                t_chirp=t_chirp, t_loop=t_loop, t_cpi=t_cpi,
                period=period_ms * 1e-3, duty=t_cpi / (period_ms * 1e-3),
                lam=lam, v_max=v_max, dv=dv, dr=dr, b_valid=b_valid_mhz,
                adc_time_us=adc_time_us, clutter=clutter,
                quant=dv / 12 ** 0.5)


def main():
    S = {}
    for label, fname in PROFILES.items():
        p = CFG_DIR / fname
        if not p.exists():
            print(f'missing: {p}', file=sys.stderr)
            continue
        S[label] = spec(p)

    rows = [
        ('TX used (TDM) / Doppler loops', lambda s: f"{s['n_tx']} / {s['n_loops']}"),
        ('chirp / Doppler-sample spacing', lambda s: f"{1e6*s['t_chirp']:.0f} / {1e6*s['t_loop']:.0f} us"),
        ('CPI (coherent integration)', lambda s: f"{1e3*s['t_cpi']:.2f} ms"),
        ('frame period / duty cycle', lambda s: f"{1e3*s['period']:.0f} ms / {100*s['duty']:.0f} %"),
        ('carrier wavelength', lambda s: f"{1e3*s['lam']:.3f} mm"),
        ('unambiguous velocity v_max', lambda s: f"{s['v_max']:.3f} m/s"),
        ('Doppler bin dv', lambda s: f"{s['dv']:.4f} m/s"),
        ('  -> quantization dv/sqrt(12)', lambda s: f"{s['quant']:.4f} m/s"),
        ('valid bandwidth / range res', lambda s: f"{s['b_valid']:.0f} MHz / {s['dr']:.3f} m"),
        ('static clutter removal', lambda s: 'ON' if s['clutter'] else 'off'),
    ]
    labels = list(S)
    w = max(len(x) for x in labels) + 2
    print('RADAR MEASUREMENT SPECIFICATION, derived from the uploaded chirp config')
    print('=' * (34 + w * len(labels)))
    print(f"{'quantity':<34}" + ''.join(f'{x:>{w}}' for x in labels))
    print('-' * (34 + w * len(labels)))
    for name, fn in rows:
        print(f'{name:<34}' + ''.join(f'{fn(S[x]):>{w}}' for x in labels))

    print()
    print('Independent checks against the data and against bags.yaml:')
    bv = S.get('best_velocity (headline flights)')
    if bv:
        print(f"  Doppler bin      config {bv['dv']:.4f}  vs measured 0.049 m/s "
              f"(unique reported velocities)")
        print(f"  range resolution config {bv['dr']:.3f}   vs measured 0.214 m "
              f"(unique reported ranges)")
        print(f"  v_max            config {bv['v_max']:.3f}  vs bags.yaml 3.136 m/s")
    old = S.get('3d (old-firmware flights)')
    if old:
        print(f"  old-fw v_max     config {old['v_max']:.2f}   vs bags.yaml 4.99 m/s")
        print(f"  old-fw dv        config {old['dv']:.3f}  vs bags.yaml 0.63 m/s")

    if bv and old:
        print()
        print('Can the old-firmware flights show a speed law at all?')
        print('The answer depends on which mechanism you assume, and that is worth')
        print('being explicit about, because the two disagree:')
        print()
        q_new = 0.0104                       # RETRACTED frame-level fit (2026-08-06):
        q_old = q_new * old['t_cpi'] / bv['t_cpi']   # kept only to price the
                                                     # retracted smear mechanism
        # A bearing error has NO T_CPI dependence, so the same per-return law
        # applies to both configs.  dphi_0 = 2.1 deg rms at boresight, and the
        # per-return factor rms(sin(theta)/cos(phi)) is about 1.2 on these
        # flights (a flat 3.5 deg x rms(sin) 0.75 was the pre-2026-08-06 number;
        # the product barely moves, which is why this comparison is unchanged).
        a_bear = math.radians(2.1) / math.sqrt(2) * 1.2
        print(f"{'|v|':>5} {'smear (RETRACTED, ~T_CPI)':>26} {'bearing (adopted)':>19}"
              f" {'old-config floor':>18} {'bearing inflates total by':>26}")
        for v in (3.0, 4.0, 5.0):
            tot = (old['quant'] ** 2 + (a_bear * v) ** 2) ** 0.5
            print(f'{v:>5.0f} {q_old*v*v:>26.4f} {a_bear*v:>19.3f} '
                  f"{old['quant']:>18.3f} {100*(tot/old['quant']-1):>25.0f}%")
        print()
        print('  Under the RETRACTED coherent-smear mechanism the old config\'s term')
        print('  scales with its 8x shorter CPI and is buried 13x by quantization,')
        print('  which is where the claim "the old bags cannot show the speed law"')
        print('  came from.  The mechanism that survived the per-point test (a')
        print('  bearing error, notebook 8b) has NO T_CPI dependence, so the old')
        print('  config carries the same per-point law and the term is NOT buried:')
        print('  it inflates the total sigma by a detectable amount across 3-5 m/s.')
        print('  So the old-firmware null is CONSISTENT WITH, but not a sharp test')
        print('  of, the surviving mechanism.  Treat it as an open question rather')
        print('  than as a confirmed prediction.')
        print(f"  (The headline config's floor is only {bv['quant']:.3f} m/s, well "
              f"under its 0.082 m/s fitted noise floor.)")
        print()
        print(f"The retracted 'T_CPI ~ 4 ms' argument was right for the WRONG "
              f"config:\n  the old profile's CPI is {1e3*old['t_cpi']:.2f} ms; "
              f"the headline profile's is {1e3*bv['t_cpi']:.2f} ms.")


if __name__ == '__main__':
    main()
