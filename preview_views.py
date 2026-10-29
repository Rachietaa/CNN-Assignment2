"""Show SSL positive pairs: each row is one training image and several random views of it.
Use before pretraining to check that the views are plausible scenes (no artifacts) yet clearly different.

Usage:
    python preview_views.py --config configs/simclr_strong.yaml
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import yaml

from src.data import eval_transform, image_folder, split_indices
from src.ssl import ssl_view_transform


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--rows', type=int, default=6)
    parser.add_argument('--views', type=int, default=5)
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    data_cfg = cfg['data']
    torch.manual_seed(0)
    base = image_folder(data_cfg['root'], 'train', None)
    train_idx, _ = split_indices(len(base))
    view_tf = ssl_view_transform(data_cfg['img_size'], data_cfg['ssl_augment'])
    orig_tf = eval_transform(data_cfg['img_size'])

    # Spread the rows over different classes.
    picks = [train_idx[i * len(train_idx) // args.rows] for i in range(args.rows)]
    fig, axes = plt.subplots(args.rows, args.views + 1, figsize=(2 * (args.views + 1), 2 * args.rows))
    for r, idx in enumerate(picks):
        img, label = base[idx]
        panels = [orig_tf(img)] + [view_tf(img) for _ in range(args.views)]
        for c, x in enumerate(panels):
            ax = axes[r, c]
            ax.imshow(x[0] * 0.5 + 0.5, cmap='gray', vmin=0, vmax=1)
            ax.axis('off')
            if c == 0:
                ax.set_title(base.classes[label], fontsize=9)
            elif r == 0:
                ax.set_title(f'view {c}', fontsize=9)
    fig.tight_layout()
    out = Path('runs') / cfg['name'] / 'views_preview.png'
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=100)
    print(f'Saved {out}')


if __name__ == '__main__':
    main()
