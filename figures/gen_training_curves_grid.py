"""Generate 2x3 grid of training curves: 3B/7B × 3 benchmarks."""
import json
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

matplotlib.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 9,
    'axes.titlesize': 10,
    'axes.labelsize': 9,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'figure.dpi': 300,
})

with open('/mnt/shared-storage-user/tanzelin-p/PSRO4math/figures/all_runs_metrics.json') as f:
    data = json.load(f)

# Config
benchmarks = [
    ('val-core/olympiad/acc/mean@4', 'OlympiadBench'),
    ('val-core/math__math/acc/mean@4', 'MATH-500'),
    ('val-core/codegen__humaneval/acc/mean@4', 'HumanEval'),
]

scales = [
    ('3B', {'ORM': '3B-ORM', 'PA-GRPO': '3B-PA-GRPO'}),
    ('7B', {'ORM': 'ORM', 'PA-GRPO': 'PA-GRPO'}),
]

colors = {'ORM': '#2196F3', 'PA-GRPO': '#4CAF50'}
labels = {'ORM': 'ORM', 'PA-GRPO': 'PA-GRPO'}

def extract_curve(run_key, metric_key):
    """Extract (steps, values) for a metric from a run."""
    run_data = data[run_key]
    steps, values = [], []
    for step_str in sorted(run_data.keys(), key=lambda x: int(x)):
        step = int(step_str)
        if metric_key in run_data[step_str]:
            val = run_data[step_str][metric_key]
            if val is not None:
                steps.append(step)
                values.append(val * 100)  # Convert to percentage
    return np.array(steps), np.array(values)

def smooth(values, window=5):
    """Simple moving average smoothing."""
    if len(values) <= window:
        return values
    kernel = np.ones(window) / window
    # Pad to avoid edge effects
    padded = np.concatenate([np.full(window//2, values[0]), values, np.full(window//2, values[-1])])
    smoothed = np.convolve(padded, kernel, mode='valid')
    return smoothed[:len(values)]

fig, axes = plt.subplots(2, 3, figsize=(11, 5.5), constrained_layout=True)

for row_idx, (scale_name, run_map) in enumerate(scales):
    for col_idx, (metric_key, bench_name) in enumerate(benchmarks):
        ax = axes[row_idx, col_idx]

        for method_name, run_key in run_map.items():
            steps, values = extract_curve(run_key, metric_key)
            if len(steps) == 0:
                continue

            # Light raw data
            ax.plot(steps, values, color=colors[method_name], alpha=0.15, linewidth=0.5)

            # Smoothed curve
            # Use different window sizes for 3B (fewer points) vs 7B
            win = 3 if '3B' in run_key else 15
            smoothed = smooth(values, window=win)
            ax.plot(steps, smoothed, color=colors[method_name], linewidth=1.8,
                    label=labels[method_name])

        # Title: benchmark name on top row only
        if row_idx == 0:
            ax.set_title(bench_name, fontweight='bold')

        # Y-axis label: scale name on left column only
        if col_idx == 0:
            ax.set_ylabel(f'Qwen2.5-{scale_name}\nAccuracy (%)')

        # X-axis label: only on bottom row
        if row_idx == 1:
            ax.set_xlabel('Training Step')

        ax.grid(True, alpha=0.3, linewidth=0.5)
        ax.set_xlim(0, 1100)

        # Legend: only on first subplot
        if row_idx == 0 and col_idx == 0:
            ax.legend(loc='lower right', framealpha=0.9)

output_path = '/mnt/shared-storage-user/tanzelin-p/PSRO4math/paper/figures/fig_training_curves_grid.pdf'
fig.savefig(output_path, bbox_inches='tight', dpi=300)
print(f'Saved to {output_path}')

# Also save PNG for preview
png_path = '/tmp/fig_training_curves_grid.png'
fig.savefig(png_path, bbox_inches='tight', dpi=150)
print(f'Preview: {png_path}')
