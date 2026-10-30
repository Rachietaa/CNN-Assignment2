"""Inference: predict the scene class of every image in a folder with one checkpoint or an ensemble.

Usage:
    python predict.py --images data/test2 --out predictions_test2.csv \
        --checkpoints runs/ft_simclr_enclr_mix/best.pt runs/ft_simclr_enclr_mix_s1/best.pt runs/ft_simclr_enclr_mix_s2/best.pt

Writes a CSV with one row per image: file name, predicted class, and its averaged probability.
"""
import argparse
import csv
from pathlib import Path

import torch

from src.data import eval_transform, load_gray
from src.engine import get_device
from src.models import build_classifier

EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp'}


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--images', required=True, help='folder of images (searched recursively)')
    parser.add_argument('--checkpoints', nargs='+', required=True)
    parser.add_argument('--out', default='predictions.csv')
    parser.add_argument('--batch-size', type=int, default=64)
    args = parser.parse_args()

    device = get_device()
    paths = sorted(p for p in Path(args.images).rglob('*') if p.suffix.lower() in EXTENSIONS)
    models, classes, transform = [], None, None
    for path in args.checkpoints:
        ckpt = torch.load(path, map_location='cpu')
        model = build_classifier(ckpt['config'], num_classes=len(ckpt['classes']))
        model.load_state_dict(ckpt['state_dict'])
        models.append(model.to(device).eval())
        classes = ckpt['classes']
        transform = eval_transform(ckpt['config']['data']['img_size'])

    rows = []
    for start in range(0, len(paths), args.batch_size):
        batch = paths[start:start + args.batch_size]
        x = torch.stack([transform(load_gray(p)) for p in batch]).to(device)
        probs = sum(m(x).softmax(dim=1) for m in models) / len(models)
        conf, pred = probs.max(dim=1)
        rows += [(p.name, classes[i], f'{c:.4f}') for p, i, c in zip(batch, pred.tolist(), conf.tolist())]

    with open(args.out, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['image', 'predicted_class', 'probability'])
        writer.writerows(rows)
    print(f'Predicted {len(rows)} images with {len(models)} model(s) -> {args.out}')


if __name__ == '__main__':
    main()
