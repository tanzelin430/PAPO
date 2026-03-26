#!/usr/bin/env python3
"""Generate response length + accuracy comparison figure for the paper."""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# Style
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 11,
    'legend.fontsize': 9,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
})

COLORS = {
    'ORM': '#2196F3',
    'PRM': '#F44336',
    'PA-GRPO': '#4CAF50',
}
DISPLAY = {
    'ORM': 'ORM',
    'PRM': 'PRM',
    'PA-GRPO': 'PAPO',
}
LINESTYLES = {
    'ORM': '-',
    'PRM': '--',
    'PA-GRPO': '-',
}

def extract(data, run, metric, smooth=5):
    run_data = data[run]
    steps = sorted([int(s) for s in run_data.keys()])
    vals = []
    valid_steps = []
    for s in steps:
        v = run_data[str(s)].get(metric)
        if v is not None:
            valid_steps.append(s)
            vals.append(float(v))
    steps = np.array(valid_steps)
    vals = np.array(vals)
    if smooth > 1 and len(vals) > smooth:
        kernel = np.ones(smooth) / smooth
        vals = np.convolve(vals, kernel, mode='valid')
        steps = steps[:len(vals)]
    return steps, vals

data = json.load(open(Path(__file__).parent / 'all_runs_metrics.json'))
runs = ['ORM', 'PRM', 'PA-GRPO']

fig, axes = plt.subplots(1, 2, figsize=(10, 3.2))

# Panel (a): Response Length
ax = axes[0]
for run in runs:
    steps, vals = extract(data, run, 'response_length/mean', smooth=7)
    ax.plot(steps, vals, color=COLORS[run], linestyle=LINESTYLES[run],
            label=DISPLAY[run], linewidth=1.8)
ax.set_xlabel('Training Step')
ax.set_ylabel('Response Length (tokens)')
ax.set_title('(a) Response Length')
ax.legend(loc='upper left')
ax.set_xlim(0, 1100)
ax.set_ylim(0, 8200)
ax.axhline(y=8192, color='gray', linestyle=':', linewidth=0.8, alpha=0.5)
ax.text(50, 8192+100, 'max length cap', fontsize=7, color='gray', alpha=0.7)
ax.grid(True, alpha=0.2)

# Panel (b): OlympiadBench Accuracy
ax = axes[1]
for run in runs:
    steps, vals = extract(data, run, 'val-core/olympiad/acc/mean@4', smooth=1)
    ax.plot(steps, vals * 100, color=COLORS[run], linestyle=LINESTYLES[run],
            label=DISPLAY[run], linewidth=1.8)
ax.set_xlabel('Training Step')
ax.set_ylabel('OlympiadBench Accuracy (%)')
ax.set_title('(b) OlympiadBench avg@4')
ax.legend(loc='upper left')
ax.set_xlim(0, 1100)
ax.grid(True, alpha=0.2)

plt.tight_layout()
out = Path(__file__).parent.parent / 'paper' / 'figures' / 'fig_response_length.pdf'
plt.savefig(out)
print(f'Saved to {out}')

# Also save to figures/ for reference
out2 = Path(__file__).parent / 'fig_response_length.pdf'
plt.savefig(out2)
print(f'Also saved to {out2}')
