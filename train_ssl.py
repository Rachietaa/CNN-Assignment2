"""Self-supervised pretraining of the encoder (no labels) from a YAML config.

Usage:
    python train_ssl.py --config configs/simclr_strong.yaml
    python train_ssl.py --config configs/dino.yaml
    python train_ssl.py --config configs/simclr_strong.yaml --resume   # continue from runs/<name>/last.pt

Methods (`ssl.method` in the config):
    simclr  contrastive: two views of an image are pulled together, all other images pushed apart
    dino    self-distillation: a student matches an EMA teacher's output across views, no negatives
    mae     masked image modeling (ViT only): hide most patches, reconstruct their pixels

Only the 1,920 training images of the fixed split are used; their labels are ignored. Every
`knn_every` epochs the frozen features are scored with a kNN classifier on the validation split, to
monitor representation quality (labels are used for that measurement only).
Outputs: runs/<name>/{encoder.pt, last.pt, history.json, summary.json, curves.png}
"""
import argparse
import copy
import json
import math
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
from src.vit import MAEDecoder, mae_loss
from src.ssl import (DINOHead, DINOLoss, MultiCropViews, ProjectionHead, TwoViews, collapse_stats,
                     ema_update, knn_accuracy, nt_xent, ssl_view_transform)


def plot_history(history, out_path):
    panels = 3 if history.get('teacher_entropy') else 2
    fig, axes = plt.subplots(1, panels, figsize=(5.5 * panels, 4))
    axes[0].plot(range(1, len(history['loss']) + 1), history['loss'])
    axes[0].set_xlabel('epoch'); axes[0].set_ylabel('SSL loss')
    axes[1].plot(history['knn_epoch'], history['knn_acc'], marker='o')
    axes[1].set_xlabel('epoch'); axes[1].set_ylabel('kNN val accuracy')
    if panels == 3:
        epochs = range(1, len(history['teacher_entropy']) + 1)
        axes[2].plot(epochs, history['teacher_entropy'], label='per image')
        axes[2].plot(epochs, history['batch_entropy'], label='batch mean')
        axes[2].set_xlabel('epoch'); axes[2].set_ylabel('teacher target entropy'); axes[2].legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def build_views(data_cfg, ssl_cfg):
    aug = data_cfg['ssl_augment']
    global_tf = ssl_view_transform(data_cfg['img_size'], aug)
    if ssl_cfg['method'] == 'simclr':
        return TwoViews(global_tf)
    if ssl_cfg['method'] == 'mae':
        return global_tf  # one view per image; the masking is the "augmentation"
    local = ssl_cfg['local_crops']
    local_aug = {**aug, 'crop_scale': local['scale'], 'blur_sigma': local['blur_sigma']}
    return MultiCropViews(global_tf, ssl_view_transform(local['size'], local_aug), local['n'])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--data-root', help='override data.root from the config')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    if args.data_root:
        cfg['data']['root'] = args.data_root
    data_cfg, ssl_cfg, tr = cfg['data'], cfg['ssl'], cfg['train']
    method = ssl_cfg['method']
    set_seed(cfg['seed'])
    device = get_device()
    print('Device:', device, '| method:', method)

    # Unlabelled pretraining set: the training part of the fixed split, several random views per image.
    views = build_views(data_cfg, ssl_cfg)
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
    m = cfg['model']
    if method == 'simclr':
        head = ProjectionHead(feat_dim, m['proj_hidden'], m['proj_dim'])
    elif method == 'mae':
        head = MAEDecoder(feat_dim, m['dec_dim'], m['dec_depth'], m['dec_heads'], encoder.grid, m['patch_size'])
    else:
        head = DINOHead(feat_dim, m['head_hidden'], m['head_bottleneck'], m['head_out'],
                        use_bn=m.get('head_bn', False), norm_last_layer=m.get('norm_last_layer', True))
    encoder, head = encoder.to(device), head.to(device)
    params = list(encoder.parameters()) + list(head.parameters())
    optimizer = torch.optim.AdamW(params, lr=tr['lr'], weight_decay=tr['weight_decay'],
                                  betas=tuple(tr.get('betas', (0.9, 0.999))))
    steps = len(loader)
    total_steps = tr['epochs'] * steps
    scheduler = cosine_with_warmup(optimizer, total_steps, tr.get('warmup_epochs', 0) * steps)

    if method == 'dino':
        # Teacher: same architecture, starts as a copy of the student, never receives gradients.
        t_encoder, t_head = copy.deepcopy(encoder), copy.deepcopy(head)
        for p in list(t_encoder.parameters()) + list(t_head.parameters()):
            p.requires_grad = False
        dino_loss = DINOLoss(m['head_out'], ssl_cfg['student_temp'], ssl_cfg['teacher_temp'],
                             ssl_cfg['center_momentum']).to(device)
        print(f'SSL images: {len(ssl_set)} | batch {tr["batch_size"]} | {steps} steps/epoch | '
              f'2 global + {ssl_cfg["local_crops"]["n"]} local views')
    elif method == 'mae':
        print(f'SSL images: {len(ssl_set)} | batch {tr["batch_size"]} | {steps} steps/epoch | '
              f'{encoder.grid ** 2} patches, {ssl_cfg["mask_ratio"]:.0%} hidden')
    else:
        temperature = ssl_cfg['temperature']
        print(f'SSL images: {len(ssl_set)} | batch {tr["batch_size"]} | {steps} steps/epoch | '
              f'{2 * tr["batch_size"] - 2} negatives per view')

    out_dir = Path('runs') / cfg['name']
    out_dir.mkdir(parents=True, exist_ok=True)
    history = {'loss': [], 'lr': [], 'knn_epoch': [], 'knn_acc': []}
    if method == 'dino':
        history.update(teacher_entropy=[], batch_entropy=[], momentum=[])
    start_epoch, elapsed = 1, 0.0
    if args.resume and (out_dir / 'last.pt').exists():
        state = torch.load(out_dir / 'last.pt', map_location='cpu')
        encoder.load_state_dict(state['encoder']); head.load_state_dict(state['head'])
        optimizer.load_state_dict(state['optimizer']); scheduler.load_state_dict(state['scheduler'])
        if method == 'dino':
            t_encoder.load_state_dict(state['t_encoder']); t_head.load_state_dict(state['t_head'])
            dino_loss.load_state_dict(state['dino_loss'])
        history, start_epoch, elapsed = state['history'], state['epoch'] + 1, state['elapsed']
        print(f'Resumed from epoch {state["epoch"]}')

    # For DINO the teacher is the model that is evaluated and kept (it is the smoother average).
    eval_encoder = t_encoder if method == 'dino' else encoder
    start = time.time() - elapsed
    for epoch in range(start_epoch, tr['epochs'] + 1):
        encoder.train(); head.train()
        history['lr'].append(optimizer.param_groups[0]['lr'])
        loss_sum, ent_sum, batch_ent_sum = 0.0, 0.0, 0.0
        for batch_views, _ in loader:  # labels are discarded
            if method == 'simclr':
                x1, x2 = batch_views[0].to(device), batch_views[1].to(device)
                # One forward pass over both views so BatchNorm sees the full 2N batch.
                z = head(encoder(torch.cat([x1, x2])))
                loss = nt_xent(*z.chunk(2), temperature)
            elif method == 'mae':
                x = batch_views.to(device)
                latent, mask, ids_restore = encoder.forward_masked(x, ssl_cfg['mask_ratio'])
                loss = mae_loss(head(latent, ids_restore), x, mask, m['patch_size'], ssl_cfg.get('norm_pix', True))
            else:
                views = [v.to(device) for v in batch_views]
                globals_ = torch.cat(views[:2])
                with torch.no_grad():
                    # Teacher in train mode: BatchNorm uses batch statistics (its running stats are
                    # not updated by the EMA).
                    t_out = t_head(t_encoder(globals_)).chunk(2)
                # Student: globals and locals have different sizes -> two forward passes.
                s_out = list(head(encoder(globals_)).chunk(2))
                if len(views) > 2:
                    s_out += list(head(encoder(torch.cat(views[2:]))).chunk(len(views) - 2))
                loss, targets = dino_loss(s_out, t_out)
                ent, batch_ent = collapse_stats(targets)
                ent_sum += ent; batch_ent_sum += batch_ent
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if tr.get('clip_grad'):
                torch.nn.utils.clip_grad_norm_(params, tr['clip_grad'])
            optimizer.step()
            scheduler.step()
            if method == 'dino':
                # Teacher momentum rises from m0 to 1 along a cosine over training.
                step = scheduler.last_epoch
                momentum = 1 - (1 - ssl_cfg['teacher_momentum']) * (math.cos(math.pi * step / total_steps) + 1) / 2
                ema_update(t_encoder, encoder, momentum)
                ema_update(t_head, head, momentum)
            loss_sum += loss.item()
        history['loss'].append(loss_sum / steps)

        line = f'Epoch {epoch:03d}/{tr["epochs"]} | SSL loss {loss_sum / steps:.4f}'
        if method == 'dino':
            history['teacher_entropy'].append(ent_sum / steps)
            history['batch_entropy'].append(batch_ent_sum / steps)
            history['momentum'].append(momentum)
            line += f' | H(target) {ent_sum / steps:.2f} H(mean) {batch_ent_sum / steps:.2f}'
        if epoch % tr['knn_every'] == 0 or epoch == tr['epochs']:
            acc = knn_accuracy(eval_encoder, knn_train, knn_val, device)
            history['knn_epoch'].append(epoch); history['knn_acc'].append(acc)
            line += f' | kNN val acc {acc:.4f}'
        print(f'{line} | {time.time() - start:.0f}s', flush=True)

        # Collapse guard: if the teacher targets are still almost uniform after `check_epoch` epochs, the
        # student has nothing to learn; stop instead of wasting the remaining compute.
        guard = ssl_cfg.get('collapse_guard')
        if method == 'dino' and guard and epoch == guard['check_epoch']:
            limit = guard['max_entropy_frac'] * math.log(m['head_out'])
            if history['teacher_entropy'][-1] > limit:
                (out_dir / 'history.json').write_text(json.dumps(history, indent=2))
                plot_history(history, out_dir / 'curves.png')
                raise SystemExit(f'COLLAPSED: H(target) {history["teacher_entropy"][-1]:.2f} > {limit:.2f} '
                                 f'at epoch {epoch}; stopping (history saved to {out_dir}/)')

        if epoch % tr.get('save_every', 50) == 0 or epoch == tr['epochs']:
            state = {'encoder': encoder.state_dict(), 'head': head.state_dict(),
                     'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),
                     'history': history, 'epoch': epoch, 'elapsed': time.time() - start}
            if method == 'dino':
                state.update(t_encoder=t_encoder.state_dict(), t_head=t_head.state_dict(),
                             dino_loss=dino_loss.state_dict())
            torch.save(state, out_dir / 'last.pt')

    # The final encoder is the last epoch (no label-based selection of SSL checkpoints).
    torch.save({'encoder': eval_encoder.state_dict(), 'config': cfg}, out_dir / 'encoder.pt')
    (out_dir / 'history.json').write_text(json.dumps(history, indent=2))
    summary = {'name': cfg['name'], 'method': method, 'epochs': tr['epochs'],
               'final_loss': history['loss'][-1], 'final_knn_acc': history['knn_acc'][-1],
               'best_knn_acc': max(history['knn_acc']), 'train_seconds': round(time.time() - start, 1),
               'device': str(device), 'torch': torch.__version__}
    (out_dir / 'summary.json').write_text(json.dumps(summary, indent=2))
    plot_history(history, out_dir / 'curves.png')
    print(f'Saved encoder, history and curves to {out_dir}/')


if __name__ == '__main__':
    main()
