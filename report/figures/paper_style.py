"""
Shared plotting style for the paper figures (Fig. 2 trajectories, Fig. 3
error-over-time).  Import and call apply() at the top of each figure script so
line widths, grid, ticks, fonts, and the colour palette are identical across
figures.

All figures are authored at their true on-page width (\textwidth = 7.16 in for
figure*), so the rcParams point sizes below are the sizes the reader sees.
"""
import matplotlib

# --- shared colour palette (Okabe-Ito, colourblind-safe) -------------------
C_GT   = '#0072B2'   # blue        : MoCap ground truth
C_LIVE = '#D55E00'   # vermillion  : SW live edge (the deployed estimate)
C_START = '#0072B2'  # start marker matches ground truth

# --- shared line/marker weights --------------------------------------------
LW_MAIN   = 1.0      # trajectory / error traces  (thin: clean line crossings)
LW_LEGEND = 1.6      # legend swatch (a touch heavier so it reads at small size)
MS_START  = 4.5      # start marker size

# Geometry constant: IEEEtran two-column text width in inches.
TEXTWIDTH_IN = 7.16

def apply():
    """Set global rcParams shared by every paper figure."""
    matplotlib.rcParams['pdf.fonttype'] = 42   # TrueType (IEEE PDF eXpress)
    matplotlib.rcParams['ps.fonttype']  = 42
    matplotlib.rcParams.update({
        # role-mapped size ladder (base / annotation / tick), true on-page pt
        'font.size':       8,
        'axes.titlesize':  8,
        'axes.labelsize':  8,
        'legend.fontsize': 7,
        'xtick.labelsize': 6.5,
        'ytick.labelsize': 6.5,
        # lines
        'lines.linewidth': LW_MAIN,
        'lines.solid_capstyle': 'round',
        # axes frame: light, uniform
        'axes.linewidth':  0.6,
        'axes.edgecolor':  '#444444',
        'axes.titlepad':   3.0,
        'axes.labelpad':   1.5,
        # grid: uniform dotted, subtle
        'axes.grid':       True,
        'grid.color':      '#B0B0B0',
        'grid.linewidth':  0.4,
        'grid.linestyle':  ':',
        'grid.alpha':      0.7,
        # ticks: short, outward, uniform
        'xtick.direction': 'out',
        'ytick.direction': 'out',
        'xtick.major.size': 2.5,
        'ytick.major.size': 2.5,
        'xtick.major.width': 0.6,
        'ytick.major.width': 0.6,
        'xtick.major.pad': 1.5,
        'ytick.major.pad': 1.5,
        # figure
        'figure.dpi':      150,
        'savefig.dpi':     300,
    })
