"""
Generate a placeholder PDF for figures that require solver re-runs.
Run once to allow LaTeX to compile before the real figures are generated.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

OUT_DIR = Path(__file__).parent

placeholders = {
    'error_over_time': (
        'Error-over-time figure\n\n'
        'Regenerate with:\n'
        '  cd analysis/\n'
        '  python validate_live_solver.py slow_racing_best_velocity \\\n'
        '    --mocap-yaw --cpp --save-arrays --no-plot\n'
        '  python validate_live_solver.py fast_racing_best_velocity \\\n'
        '    --mocap-yaw --cpp --save-arrays --no-plot\n'
        '  python validate_live_solver.py slow_racing_best_velocity \\\n'
        '    --mocap-yaw --cpp --sliding-window --save-arrays --no-plot\n'
        '  python validate_live_solver.py fast_racing_best_velocity \\\n'
        '    --mocap-yaw --cpp --sliding-window --save-arrays --no-plot\n'
        '  python ../report/figures/gen_error_time.py'
    ),
}

for name, msg in placeholders.items():
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.text(0.5, 0.5, msg, ha='center', va='center',
            fontsize=10, family='monospace', transform=ax.transAxes,
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    ax.axis('off')
    out = OUT_DIR / f'{name}.pdf'
    fig.savefig(out, bbox_inches='tight')
    print(f'Saved placeholder: {out}')
    plt.close(fig)
