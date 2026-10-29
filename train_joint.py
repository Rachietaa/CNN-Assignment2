"""Joint training from random init: classification loss + λ · SimCLR loss, optimized together.

Usage:
    python train_joint.py --config configs/joint_simclr.yaml

Each training image gives three views: one mildly augmented view for the classifier (the same
augmentation as the supervised baseline) and two strongly augmented views for the contrastive loss.
All three go through the shared encoder in one forward pass. The classification layer and the
projection head sit on top of the same 512-d features.
Outputs: runs/<name>/{best.pt, history.json, summary.json, curves.png}; best.pt works with evaluate.py.
"""
import argparse
import copy
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader, Subset

from src.data import eval_transform, image_folder, split_indices, supervised_transform
from src.engine import cosine_with_warmup, evaluate, get_device, set_seed
from src.models import build_classifier
from src.ssl import ProjectionHead, nt_xent, ssl_view_transform
from train_classifier import plot_history


class JointViews:
    def __init__(self, sup_transform, ssl_transform):
        self.sup_transform, self.ssl_transform = sup_transform, ssl_transform

    def __call__(self, img):
        return self.sup_transform(img), self.ssl_transform(img), self.ssl_transform(img)


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
    data_cfg, tr = cfg['data'], cfg['train']
    set_seed(cfg['seed'])
    device = get_device()
    print('Device:', device)

    size = data_cfg['img_size']
    views = JointViews(supervised_transform(size, data_cfg['augment']),
                       ssl_view_transform(size, data_cfg['ssl_augment']))
    train_view = image_folder(data_cfg['root'], 'train', views)
    classes = train_view.classes
    train_idx, val_idx = split_indices(len(train_view))
    workers = data_cfg.get('num_workers', 8)
    train_loader = DataLoader(Subset(train_view, train_idx), batch_size=tr['batch_size'], shuffle=True,
                              drop_last=True, num_workers=workers, persistent_workers=workers > 0,
                              pin_memory=device.type == 'cuda')
    val_loader = DataLoader(Subset(image_folder(data_cfg['root'], 'train', eval_transform(size)), val_idx),
                            batch_size=256, shuffle=False, num_workers=2, persistent_workers=True)

    model = build_classifier(cfg, num_classes=len(classes)).to(device)
    m = cfg['model']
    head = ProjectionHead(model.fc.in_features, m['proj_hidden'], m['proj_dim']).to(device)
    params = list(model.parameters()) + list(head.parameters())
    optimizer = torch.optim.AdamW(params, lr=tr['lr'], weight_decay=tr['weight_decay'])
    steps = len(train_loader)
    scheduler = cosine_with_warmup(optimizer, tr['epochs'] * steps, tr.get('warmup_epochs', 0) * steps)
    criterion = nn.CrossEntropyLoss()
    lam, temperature = cfg['joint']['ssl_weight'], cfg['joint']['temperature']
    print(f'Train {len(train_idx)} | Val {len(val_idx)} | λ = {lam} | {steps} steps/epoch')

    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'lr': [],
               'ce_loss': [], 'ssl_loss': []}
    best_state, best_val_acc, best_epoch = copy.deepcopy(model.state_dict()), 0.0, 0
    start = time.time()
    for epoch in range(1, tr['epochs'] + 1):
        model.train(); head.train()
        history['lr'].append(optimizer.param_groups[0]['lr'])
        ce_sum, ssl_sum, correct, seen = 0.0, 0.0, 0, 0
        for (x_sup, x1, x2), labels in train_loader:
            x_sup, x1, x2, labels = x_sup.to(device), x1.to(device), x2.to(device), labels.to(device)
            n = x_sup.size(0)
            feats = model.encoder(torch.cat([x_sup, x1, x2]))
            logits = model.fc(model.dropout(feats[:n]))
            ce = criterion(logits, labels)
            ssl = nt_xent(*head(feats[n:]).chunk(2), temperature)
            loss = ce + lam * ssl
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()
            ce_sum += ce.item() * n; ssl_sum += ssl.item() * n
            correct += (logits.argmax(dim=1) == labels).sum().item(); seen += n

        val = evaluate(model, val_loader, device)
        history['train_loss'].append(ce_sum / seen); history['ce_loss'].append(ce_sum / seen)
        history['ssl_loss'].append(ssl_sum / seen); history['train_acc'].append(correct / seen)
        history['val_loss'].append(val['loss']); history['val_acc'].append(val['acc'])
        if val['acc'] > best_val_acc:
            best_val_acc, best_epoch = val['acc'], epoch
            best_state = copy.deepcopy(model.state_dict())
        print(f'Epoch {epoch:03d}/{tr["epochs"]} | CE {ce_sum / seen:.4f} SSL {ssl_sum / seen:.4f} '
              f'acc {correct / seen:.4f} | val loss {val["loss"]:.4f} acc {val["acc"]:.4f} | '
              f'{time.time() - start:.0f}s', flush=True)

    print(f'Best validation accuracy: {best_val_acc:.4f} (epoch {best_epoch})')
    model.load_state_dict(best_state)
    out_dir = Path('runs') / cfg['name']
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({'state_dict': model.state_dict(), 'config': cfg, 'classes': classes,
                'best_val_acc': best_val_acc, 'best_epoch': best_epoch}, out_dir / 'best.pt')
    (out_dir / 'history.json').write_text(json.dumps(history, indent=2))
    summary = {'name': cfg['name'], 'best_val_acc': best_val_acc, 'best_epoch': best_epoch,
               'final_train_acc': history['train_acc'][-1], 'train_seconds': round(time.time() - start, 1),
               'device': str(device), 'torch': torch.__version__}
    (out_dir / 'summary.json').write_text(json.dumps(summary, indent=2))
    plot_history(history, out_dir / 'curves.png')
    print(f'Saved checkpoint, history and curves to {out_dir}/')


if __name__ == '__main__':
    main()
