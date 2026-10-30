"""Train a classifier from a YAML config.

Usage:
    python train_classifier.py --config configs/sup_resnet18.yaml
    python train_classifier.py --config configs/sup_resnet18.yaml --seed 1   # -> runs/<name>_s1/
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import yaml

from src.data import labelled_loaders, supervised_transform
from src.engine import cosine_with_warmup, get_device, set_seed, train_classifier
from src.models import build_classifier


def plot_history(history, out_path):
    epochs = range(1, len(history['train_loss']) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ax1.plot(epochs, history['train_loss'], label='train')
    ax1.plot(epochs, history['val_loss'], label='val')
    ax1.set_xlabel('epoch'); ax1.set_ylabel('loss'); ax1.legend()
    ax2.plot(epochs, history['train_acc'], label='train')
    ax2.plot(epochs, history['val_acc'], label='val')
    ax2.set_xlabel('epoch'); ax2.set_ylabel('accuracy'); ax2.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--data-root', help='override data.root from the config')
    parser.add_argument('--seed', type=int, help='override the training seed (the train/val split stays fixed)')
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    if args.data_root:
        cfg['data']['root'] = args.data_root
    if args.seed is not None:
        cfg['seed'] = args.seed
        cfg['name'] = f"{cfg['name']}_s{args.seed}"

    set_seed(cfg['seed'])
    device = get_device()
    print('Device:', device)

    train_tf = supervised_transform(cfg['data']['img_size'], cfg['data'].get('augment'))
    loaders, classes = labelled_loaders(cfg, device, train_tf)
    print(f"Train {len(loaders['train'].dataset)} | Val {len(loaders['val'].dataset)} | "
          f"Test {len(loaders['test'].dataset)} | {len(classes)} classes")

    model = build_classifier(cfg, num_classes=len(classes))
    tr = cfg['train']
    init = (cfg.get('init') or {}).get('encoder')
    if init:
        # Start from an SSL-pretrained encoder; the classification layer is new.
        model.encoder.load_state_dict(torch.load(init, map_location='cpu')['encoder'])
        print(f'Loaded encoder from {init}')
    freeze = tr.get('freeze_encoder', False)
    if freeze:
        # Linear probe: only the classification layer is trained on top of fixed features.
        for p in model.encoder.parameters():
            p.requires_grad = False
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Trainable parameters: {n_params:,}')

    if tr.get('encoder_lr') is not None and not freeze:
        # Lower learning rate for the pretrained encoder than for the new classification layer, so the
        # first large updates (driven by the random new layer) do not overwrite the SSL features.
        groups = [{'params': model.encoder.parameters(), 'lr': tr['encoder_lr']},
                  {'params': model.fc.parameters(), 'lr': tr['lr']}]
        print(f"Learning rates: encoder {tr['encoder_lr']}, classification layer {tr['lr']}")
    else:
        groups = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(groups, lr=tr['lr'], weight_decay=tr['weight_decay'])
    steps = len(loaders['train'])
    scheduler = cosine_with_warmup(optimizer, tr['epochs'] * steps, tr.get('warmup_epochs', 0) * steps)
    model, history, best_val_acc, best_epoch, seconds = train_classifier(
        model, loaders, optimizer, scheduler, tr['epochs'], device,
        label_smoothing=tr.get('label_smoothing', 0.0), freeze_encoder=freeze,
        mix_cfg=tr.get('mix'), num_classes=len(classes))

    out_dir = Path('runs') / cfg['name']
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({'state_dict': model.state_dict(), 'config': cfg, 'classes': classes,
                'best_val_acc': best_val_acc, 'best_epoch': best_epoch}, out_dir / 'best.pt')
    (out_dir / 'history.json').write_text(json.dumps(history, indent=2))
    summary = {'name': cfg['name'], 'best_val_acc': best_val_acc, 'best_epoch': best_epoch,
               'final_train_acc': history['train_acc'][-1], 'params': n_params,
               'train_seconds': round(seconds, 1), 'device': str(device), 'torch': torch.__version__}
    (out_dir / 'summary.json').write_text(json.dumps(summary, indent=2))
    plot_history(history, out_dir / 'curves.png')
    print(f'Saved checkpoint, history and curves to {out_dir}/')


if __name__ == '__main__':
    main()
