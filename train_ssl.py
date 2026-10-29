"""Self-supervised pretraining of the encoder (no labels) from a YAML config.

Usage:
    python train_ssl.py --config configs/simclr_strong.yaml
    python train_ssl.py --config configs/simclr_strong.yaml --resume   # continue from runs/<name>/last.pt

Only the 1,920 training images of the fixed split are used; their labels are ignored. Every
`knn_every` epochs the frozen features are scored with a kNN classifier on the validation split, to
monitor representation quality (labels are used for that measurement only).
Outputs: runs/<name>/{encoder.pt, last.pt, history.json, summary.json, curves.png}
"""
import argparse
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import yaml
from torch.utils.data import DataLoader, Subset

from src.data import eval_transform, image_folder, split_indices
from src.engine import cosine_with_warmup, get_device, set_seed
from src.models import build_encoder
from src.ssl import ProjectionHead, TwoViews, knn_accuracy, nt_xent, ssl_view_transform


def plot_history(history, out_path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ax1.plot(range(1, len(history['loss']) + 1), history['loss'])
    ax1.set_xlabel('epoch'); ax1.set_ylabel('SSL loss')
    ax2.plot(history['knn_epoch'], history['knn_acc'], marker='o')
    ax2.set_xlabel('epoch'); ax2.set_ylabel('kNN val accuracy')
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--data-root', help='override data.root from the config')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    if args.data_root:
        cfg['data']['root'] = args.data_root
    data_cfg, tr = cfg['data'], cfg['train']
    set_seed(cfg['seed'])
    device = get_device()
    print('Device:', device)

    # Unlabelled pretraining set: the training part of the fixed split, two random views per image.
    views = TwoViews(ssl_view_transform(data_cfg['img_size'], data_cfg['ssl_augment']))
    train_idx, val_idx = split_indices(len(image_folder(data_cfg['root'], 'train', None)))
    ssl_set = Subset(image_folder(data_cfg['root'], 'train', views), train_idx)
    workers = data_cfg.get('num_workers', 8)
    loader = DataLoader(ssl_set, batch_size=tr['batch_size'], shuffle=True, drop_last=True,
                        num_workers=workers, persistent_workers=workers > 0,
                        pin_memory=device.type == 'cuda')

    # kNN monitor: un-augmented train and val images.
    eval_tf = eval_transform(data_cfg['img_size'])
    knn_kw = dict(batch_size=256, shuffle=False, num_workers=2, persistent_workers=True)
    knn_train = DataLoader(Subset(image_folder(data_cfg['root'], 'train', eval_tf), train_idx), **knn_kw)
    knn_val = DataLoader(Subset(image_folder(data_cfg['root'], 'train', eval_tf), val_idx), **knn_kw)

    encoder, feat_dim = build_encoder(cfg)
    head = ProjectionHead(feat_dim, cfg['model']['proj_hidden'], cfg['model']['proj_dim'])
    encoder, head = encoder.to(device), head.to(device)
    params = list(encoder.parameters()) + list(head.parameters())
    optimizer = torch.optim.AdamW(params, lr=tr['lr'], weight_decay=tr['weight_decay'])
    steps = len(loader)
    scheduler = cosine_with_warmup(optimizer, tr['epochs'] * steps, tr.get('warmup_epochs', 0) * steps)
    temperature = cfg['ssl']['temperature']
    print(f'SSL images: {len(ssl_set)} | batch {tr["batch_size"]} | {steps} steps/epoch | '
          f'{2 * tr["batch_size"] - 2} negatives per view')

    out_dir = Path('runs') / cfg['name']
    out_dir.mkdir(parents=True, exist_ok=True)
    history = {'loss': [], 'lr': [], 'knn_epoch': [], 'knn_acc': []}
    start_epoch, elapsed = 1, 0.0
    if args.resume and (out_dir / 'last.pt').exists():
        state = torch.load(out_dir / 'last.pt', map_location='cpu')
        encoder.load_state_dict(state['encoder']); head.load_state_dict(state['head'])
        optimizer.load_state_dict(state['optimizer']); scheduler.load_state_dict(state['scheduler'])
        history, start_epoch, elapsed = state['history'], state['epoch'] + 1, state['elapsed']
        print(f'Resumed from epoch {state["epoch"]}')

    start = time.time() - elapsed
    for epoch in range(start_epoch, tr['epochs'] + 1):
        encoder.train(); head.train()
        history['lr'].append(optimizer.param_groups[0]['lr'])
        loss_sum = 0.0
        for (x1, x2), _ in loader:  # labels are discarded
            x1, x2 = x1.to(device), x2.to(device)
            # One forward pass over both views so BatchNorm sees the full 2N batch.
            z = head(encoder(torch.cat([x1, x2])))
            loss = nt_xent(*z.chunk(2), temperature)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()
            loss_sum += loss.item()
        history['loss'].append(loss_sum / steps)

        line = f'Epoch {epoch:03d}/{tr["epochs"]} | SSL loss {loss_sum / steps:.4f}'
        if epoch % tr['knn_every'] == 0 or epoch == tr['epochs']:
            acc = knn_accuracy(encoder, knn_train, knn_val, device)
            history['knn_epoch'].append(epoch); history['knn_acc'].append(acc)
            line += f' | kNN val acc {acc:.4f}'
        print(f'{line} | {time.time() - start:.0f}s', flush=True)

        if epoch % tr.get('save_every', 50) == 0 or epoch == tr['epochs']:
            torch.save({'encoder': encoder.state_dict(), 'head': head.state_dict(),
                        'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),
                        'history': history, 'epoch': epoch, 'elapsed': time.time() - start},
                       out_dir / 'last.pt')

    # The final encoder is the last epoch (no label-based selection of SSL checkpoints).
    torch.save({'encoder': encoder.state_dict(), 'config': cfg}, out_dir / 'encoder.pt')
    (out_dir / 'history.json').write_text(json.dumps(history, indent=2))
    summary = {'name': cfg['name'], 'epochs': tr['epochs'], 'final_loss': history['loss'][-1],
               'final_knn_acc': history['knn_acc'][-1], 'best_knn_acc': max(history['knn_acc']),
               'train_seconds': round(time.time() - start, 1), 'device': str(device),
               'torch': torch.__version__}
    (out_dir / 'summary.json').write_text(json.dumps(summary, indent=2))
    plot_history(history, out_dir / 'curves.png')
    print(f'Saved encoder, history and curves to {out_dir}/')


if __name__ == '__main__':
    main()
