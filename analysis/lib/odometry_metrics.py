"""
Odometry evaluation metrics: path length, drift %, KITTI-style translational RPE.

All functions operate on pre-aligned position arrays (world-frame, metres).
No dependency on the solver — pure NumPy, reusable from figure scripts or
validate_live_solver.py summary prints.

Usage example::

    from odometry_metrics import path_length, drift_percent, translational_rpe
    L   = path_length(mocap_pos)          # total GT path in metres
    pct = drift_percent(pos_rmse, L)       # e.g. 0.70 %
    rpe = translational_rpe(est, gt, seg_lengths=[5,10,20,30])
    # rpe['trans_pct'][i] = mean translational error (%) for seg_lengths[i]
"""
import numpy as np


# ---------------------------------------------------------------------------
# Scalar helpers
# ---------------------------------------------------------------------------

def path_length(positions: np.ndarray) -> float:
    """Total path length along a trajectory.

    Parameters
    ----------
    positions : (N, 3) array of world-frame positions.

    Returns
    -------
    float : total path length in metres.
    """
    return float(np.sum(np.linalg.norm(np.diff(positions, axis=0), axis=1)))


def drift_percent(pos_rmse: float, path_len: float) -> float:
    """Position RMSE as a percentage of distance traveled.

    Parameters
    ----------
    pos_rmse : position RMSE in metres.
    path_len : total path length in metres.

    Returns
    -------
    float : drift percentage (100 * pos_rmse / path_len).
    """
    return 100.0 * pos_rmse / path_len


# ---------------------------------------------------------------------------
# KITTI-style distance-based translational RPE
# ---------------------------------------------------------------------------

def translational_rpe(
    pos_est: np.ndarray,
    pos_gt: np.ndarray,
    seg_lengths: "list[float]",
) -> dict:
    """KITTI-style distance-based translational relative pose error (RPE).

    For each target segment length *L*, all sub-segments of that length are
    enumerated along the ground-truth trajectory (by accumulated GT distance).
    Each sub-segment is implicitly aligned at its start point (the translation
    and rotation relative to the sub-segment origin cancel).  The end-point
    translational displacement error is measured and normalised by *L*.

    This metric is alignment-independent: because both estimate and ground
    truth use the same global alignment (R_align, t_align fixed at frame 0),
    the alignment offsets cancel in every relative displacement, so the result
    is the same whether positions are reported in the aligned or raw frame.

    Parameters
    ----------
    pos_est : (N, 3) estimated positions, pre-aligned to GT world frame.
    pos_gt  : (N, 3) ground-truth positions.
    seg_lengths : list of target segment lengths in metres,
                  e.g. [5, 10, 20, 30, 40].

    Returns
    -------
    dict with keys:
        'seg_lengths'   : list of target lengths (metres) — same as input.
        'trans_pct'     : (K,) mean translational error (% of seg length).
        'trans_m'       : (K,) mean translational error (metres).
        'trans_std_pct' : (K,) std-dev of translational error (%).
        'n_segs'        : (K,) number of valid sub-segments per length.
    """
    N = len(pos_gt)
    step_norms = np.linalg.norm(np.diff(pos_gt, axis=0), axis=1)
    d_cum = np.concatenate([[0.0], np.cumsum(step_norms)])

    trans_pct_out  = []
    trans_m_out    = []
    trans_std_out  = []
    n_segs_out     = []

    for L in seg_lengths:
        errors_m = []
        for i in range(N):
            target = d_cum[i] + L
            if target > d_cum[-1]:
                break
            j = int(np.searchsorted(d_cum, target))
            if j >= N:
                break
            # Relative displacement error — alignment cancels
            err_m = float(np.linalg.norm(
                (pos_est[j] - pos_est[i]) - (pos_gt[j] - pos_gt[i])
            ))
            errors_m.append(err_m)

        if errors_m:
            errs     = np.array(errors_m)
            mean_m   = float(np.mean(errs))
            std_m    = float(np.std(errs))
            trans_pct_out.append(100.0 * mean_m / L)
            trans_m_out.append(mean_m)
            trans_std_out.append(100.0 * std_m / L)
            n_segs_out.append(len(errs))
        else:
            trans_pct_out.append(np.nan)
            trans_m_out.append(np.nan)
            trans_std_out.append(np.nan)
            n_segs_out.append(0)

    return {
        'seg_lengths':    seg_lengths,
        'trans_pct':      np.array(trans_pct_out),
        'trans_m':        np.array(trans_m_out),
        'trans_std_pct':  np.array(trans_std_out),
        'n_segs':         np.array(n_segs_out),
    }


# ---------------------------------------------------------------------------
# Time-based drift (cross-check)
# ---------------------------------------------------------------------------

def time_based_rpe(
    t: np.ndarray,
    pos_est: np.ndarray,
    pos_gt: np.ndarray,
    durations: "list[float]",
) -> dict:
    """Time-based relative pose error (cross-check for distance-based RPE).

    Same as :func:`translational_rpe` but sub-segments are defined by elapsed
    time rather than accumulated GT distance.  Useful for checking whether the
    distance-based result is driven by a few fast/slow sections.

    Parameters
    ----------
    t         : (N,) timestamps in seconds (relative, starting at 0).
    pos_est   : (N, 3) estimated positions.
    pos_gt    : (N, 3) ground-truth positions.
    durations : list of target sub-segment durations in seconds.

    Returns
    -------
    dict with keys:
        'seg_durations' : list of target durations (seconds).
        'trans_m'       : (K,) mean translational error (metres).
        'trans_pct'     : (K,) mean error as % of mean sub-segment path length.
        'n_segs'        : (K,) number of valid sub-segments per duration.
    """
    N = len(t)
    trans_m_out   = []
    trans_pct_out = []
    n_segs_out    = []

    for dt in durations:
        errors_m  = []
        seg_paths = []
        for i in range(N):
            target_t = t[i] + dt
            if target_t > t[-1]:
                break
            j = int(np.searchsorted(t, target_t))
            if j >= N:
                break
            err_m    = float(np.linalg.norm(
                (pos_est[j] - pos_est[i]) - (pos_gt[j] - pos_gt[i])
            ))
            seg_path = float(np.linalg.norm(pos_gt[j] - pos_gt[i]))
            errors_m.append(err_m)
            seg_paths.append(seg_path)

        if errors_m:
            errs       = np.array(errors_m)
            mean_path  = float(np.mean(seg_paths))
            mean_m     = float(np.mean(errs))
            mean_pct   = 100.0 * mean_m / mean_path if mean_path > 0.01 else np.nan
            trans_m_out.append(mean_m)
            trans_pct_out.append(mean_pct)
            n_segs_out.append(len(errs))
        else:
            trans_m_out.append(np.nan)
            trans_pct_out.append(np.nan)
            n_segs_out.append(0)

    return {
        'seg_durations': durations,
        'trans_m':       np.array(trans_m_out),
        'trans_pct':     np.array(trans_pct_out),
        'n_segs':        np.array(n_segs_out),
    }


# ---------------------------------------------------------------------------
# Convenience: print a formatted summary table
# ---------------------------------------------------------------------------

def print_summary_table(
    bag_label: str,
    pos_gt: np.ndarray,
    pos_est: np.ndarray,
    pos_rmse: float,
    vel_rmse: float,
    ori_rmse: float,
    mode: str = "batch",
) -> None:
    """Print a one-line summary plus a brief RPE table to stdout."""
    plen  = path_length(pos_gt)
    drift = drift_percent(pos_rmse, plen)
    segs  = [5, 10, 20, 30]
    rpe   = translational_rpe(pos_est, pos_gt, segs)
    print(f"\n{'='*64}")
    print(f"  {bag_label}  [{mode}]")
    print(f"  Path length : {plen:.1f} m")
    print(f"  Pos RMSE    : {pos_rmse:.3f} m  ({drift:.2f}% of path)")
    print(f"  Vel RMSE    : {vel_rmse:.3f} m/s")
    print(f"  Ori RMSE    : {ori_rmse:.3f} deg")
    print(f"  KITTI translational RPE:")
    for i, L in enumerate(segs):
        pct = rpe['trans_pct'][i]
        n   = rpe['n_segs'][i]
        print(f"    {L:>4} m  :  {pct:5.2f}%  (n={n})")
    print(f"{'='*64}")
