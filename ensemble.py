"""Evaluate an ensemble: average the predicted class probabilities of several classifier checkpoints.

Usage:
    python ensemble.py --split val --out runs/ens_ft_simclr \
        --checkpoints runs/ft_simclr_strong/best.pt runs/ft_simclr_strong_s1/best.pt runs/ft_simclr_strong_s2/best.pt
"""
import argparse
import json
from pathlib import Path

import torch

from src.data import eval_transform, labelled_loaders
from src.engine import get_device
from src.models import build_classifier


@torch.inference_mode()
def predict_probs(model, loader, device):
    model.eval()
    probs, labels = [], []
    for images, y in loader:
        probs.append(model(images.to(device)).softmax(dim=1).cpu())
        labels.append(y)
    return torch.cat(probs), torch.cat(labels)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoints', nargs='+', required=True)
    parser.add_argument('--split', choices=['val', 'test'], default='val')
    parser.add_argument('--out', required=True, help='output folder for eval_<split>.json')
    parser.add_argument('--data-root', help='override data.root stored in the checkpoints')
    args = parser.parse_args()

    device = get_device()
    total, labels, classes = None, None, None
    for path in args.checkpoints:
        ckpt = torch.load(path, map_location='cpu')
        cfg = ckpt['config']
        if args.data_root:
            cfg['data']['root'] = args.data_root
        loaders, classes = labelled_loaders(cfg, device, eval_transform(cfg['data']['img_size']))
        model = build_classifier(cfg, num_classes=len(classes))
        model.load_state_dict(ckpt['state_dict'])
        probs, y = predict_probs(model.to(device), loaders[args.split], device)
        if labels is not None:
            assert torch.equal(labels, y), 'checkpoints were evaluated on different images'
        labels = y
        acc = (probs.argmax(dim=1) == y).float().mean().item()
        print(f'{path}: {args.split} accuracy {acc:.4f}')
        total = probs if total is None else total + probs

    preds = total.argmax(dim=1)
    acc = (preds == labels).float().mean().item()
    confusion = torch.zeros(len(classes), len(classes), dtype=torch.long)
    for t, p in zip(labels, preds):
        confusion[t, p] += 1
    per_class = (confusion.diag().float() / confusion.sum(dim=1).clamp(min=1).float()).tolist()
    print(f'Ensemble of {len(args.checkpoints)}: {args.split} accuracy {acc:.4f}')

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / f'eval_{args.split}.json').write_text(json.dumps(
        {'split': args.split, 'acc': acc, 'checkpoints': args.checkpoints,
         'per_class': dict(zip(classes, per_class)), 'confusion': confusion.tolist()}, indent=2))
    print(f'Saved {out}/eval_{args.split}.json')


if __name__ == '__main__':
    main()
