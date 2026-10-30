"""Report figures from the saved run histories (no training).

Usage:
    python make_figures.py      # -> figures/knn_during_pretraining.png, figures/label_efficiency.png
"""
import json
import statistics
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Categorical palette (validated for colour-vision deficiency); text stays in neutral ink.
BLUE, ORANGE, AQUA, YELLOW = '#2a78d6', '#eb6834', '#1baf7a', '#eda100'
INK, MUTED, GRID = '#0b0b0b', '#52514e', '#e4e3df'
OUT = Path('figures')


def history(name):
    return json.loads((Path('runs') / name / 'history.json').read_text())


def style(ax):
    ax.spines[['top', 'right']].set_visible(False)
    ax.spines[['left', 'bottom']].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(axis='y', color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def knn_figure():
    """kNN validation accuracy of the frozen encoder during label-free pretraining."""
    runs = [('simclr_strong', 'SimCLR, strong views', BLUE),
            ('simclr_mild', 'SimCLR, mild views', ORANGE),
            ('mae_vit', 'MAE (ViT-Tiny)', AQUA),
            ('dino_bn', 'DINO (stopped at 100)', YELLOW)]
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    for name, label, color in runs:
        path = Path('runs') / name / 'history.json'
        if name == 'dino_bn':
            # The stopped DINO run has no history.json; its kNN values are in the log.
            epochs, accs = [], []
            for line in Path('runs/dino_bn.log').read_text().splitlines():
                if 'kNN val acc' in line:
                    epochs.append(int(line.split('/')[0].split()[-1]))
                    accs.append(float(line.split('kNN val acc')[1].split('|')[0]))
        elif path.exists():
            h = history(name)
            epochs, accs = h['knn_epoch'], h['knn_acc']
        else:
            continue
        accs = [100 * a for a in accs]
        ax.plot(epochs, accs, color=color, linewidth=2, marker='o', markersize=3)
        # DINO's label goes above its short line so it does not cross the MAE line.
        offset, anchor = ((0, 9), (epochs[0], max(accs))) if name == 'dino_bn' else ((6, 0), (epochs[-1], accs[-1]))
        ax.annotate(label, anchor, xytext=offset, textcoords='offset points',
                    va='center', fontsize=8.5, color=INK)
    ax.axhline(79.2, color=MUTED, linewidth=1, linestyle='--')
    ax.text(10, 80.2, 'supervised ResNet-18, fine-tuned with all labels (79.2%)', fontsize=8, color=MUTED)
    ax.set_xlabel('pretraining epoch (no labels)', color=INK, fontsize=9)
    ax.set_ylabel('kNN val accuracy (%)', color=INK, fontsize=9)
    ax.set_ylim(20, 85)
    ax.set_xlim(0, 1080)
    style(ax)
    fig.tight_layout()
    fig.savefig(OUT / 'knn_during_pretraining.png', dpi=200)
    plt.close(fig)


def best(name):
    return 100 * max(history(name)['val_acc'])


def label_efficiency_figure():
    """Validation accuracy vs number of labelled images, supervised vs SimCLR fine-tuning."""
    sup100 = [best(n) for n in ('sup_resnet18', 'sup_resnet18_s1', 'sup_resnet18_s2')]
    ssl100 = [best(n) for n in ('ft_simclr_strong', 'ft_simclr_strong_s1', 'ft_simclr_strong_s2')]
    x = [192, 479, 1920]
    sup = [best('sup_10pct'), best('sup_25pct'), statistics.mean(sup100)]
    ssl = [best('ft_simclr_10pct'), best('ft_simclr_25pct'), statistics.mean(ssl100)]
    fig, ax = plt.subplots(figsize=(4.2, 3.2))
    ax.plot(x, sup, color=ORANGE, linewidth=2, marker='o', markersize=5)
    ax.plot(x, ssl, color=BLUE, linewidth=2, marker='o', markersize=5)
    ax.errorbar([1920], [sup[2]], yerr=[statistics.stdev(sup100)], color=ORANGE, capsize=3, linewidth=1)
    ax.errorbar([1920], [ssl[2]], yerr=[statistics.stdev(ssl100)], color=BLUE, capsize=3, linewidth=1)
    for xi, s, p in zip(x, sup, ssl):
        # Gap from the rounded values, so it matches the numbers quoted in the text.
        ax.annotate(f'+{round(p, 1) - round(s, 1):.1f}', (xi, (s + p) / 2), xytext=(8, 0),
                    textcoords='offset points', va='center', fontsize=8.5, color=INK)
    ax.annotate('SimCLR + fine-tune', (x[0], ssl[0]), xytext=(-4, 10), textcoords='offset points',
                fontsize=8.5, color=INK)
    ax.annotate('supervised', (x[0], sup[0]), xytext=(10, -4), textcoords='offset points',
                va='center', fontsize=8.5, color=INK)
    ax.set_xscale('log')
    ax.set_xticks(x)
    ax.set_xticklabels(['192\n(10%)', '479\n(25%)', '1,920\n(100%)'])
    ax.minorticks_off()
    ax.set_xlim(150, 3000)
    ax.set_ylim(48, 86)
    ax.set_xlabel('labelled training images', color=INK, fontsize=9)
    ax.set_ylabel('val accuracy (%)', color=INK, fontsize=9)
    style(ax)
    fig.tight_layout()
    fig.savefig(OUT / 'label_efficiency.png', dpi=200)
    plt.close(fig)


def main():
    OUT.mkdir(exist_ok=True)
    knn_figure()
    label_efficiency_figure()
    print(f'Saved figures to {OUT}/')


if __name__ == '__main__':
    main()
