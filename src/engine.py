"""Shared utilities: seeding, device, LR schedule, classifier training and evaluation."""
import copy
import math
import random
import time

import numpy as np
import torch
import torch.nn as nn


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def cosine_with_warmup(optimizer, total_steps, warmup_steps):
    """Linear warm-up from ~0 to the full LR, then cosine decay to 0. Stepped once per batch."""
    def factor(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


@torch.inference_mode()
def evaluate(model, loader, device, num_classes=None):
    """Return loss, accuracy and (optionally) a confusion matrix [true, pred]."""
    model.eval()
    loss_sum, correct, total = 0.0, 0, 0
    confusion = torch.zeros(num_classes, num_classes, dtype=torch.long) if num_classes else None
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss_sum += nn.functional.cross_entropy(logits, labels, reduction='sum').item()
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        if confusion is not None:
            for t, p in zip(labels.cpu(), preds.cpu()):
                confusion[t, p] += 1
    return {'loss': loss_sum / total, 'acc': correct / total, 'confusion': confusion}


def build_mixer(mix_cfg, num_classes):
    """Mixup or CutMix, chosen at random per mixed batch. Returns None if not configured."""
    if not mix_cfg:
        return None
    from torchvision.transforms import v2
    return v2.RandomChoice([v2.MixUp(alpha=mix_cfg['mixup_alpha'], num_classes=num_classes),
                            v2.CutMix(alpha=mix_cfg['cutmix_alpha'], num_classes=num_classes)])


def train_classifier(model, loaders, optimizer, scheduler, epochs, device, label_smoothing=0.0,
                     freeze_encoder=False, mix_cfg=None, num_classes=16):
    """Train with cross-entropy and keep the weights from the epoch with the best validation accuracy.
    The test set is never touched here.

    freeze_encoder: linear probe; the encoder also stays in eval mode so its BatchNorm statistics
    remain exactly those learned during pretraining.
    mix_cfg: {'p', 'mixup_alpha', 'cutmix_alpha'}: with probability p a batch is mixed with Mixup (blend
    two images and their labels) or CutMix (paste a patch of one image into another); labels become soft.
    Train accuracy is then measured against the original labels, so it is approximate."""
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    mixer = build_mixer(mix_cfg, num_classes)
    model = model.to(device)
    best_state, best_val_acc, best_epoch = copy.deepcopy(model.state_dict()), 0.0, 0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'lr': []}
    start = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        if freeze_encoder:
            model.encoder.eval()
        history['lr'].append(optimizer.param_groups[0]['lr'])
        loss_sum, correct, seen = 0.0, 0, 0
        for images, labels in loaders['train']:
            targets = labels
            if mixer is not None and random.random() < mix_cfg['p']:
                images, targets = mixer(images, labels)
            images, labels, targets = images.to(device), labels.to(device), targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            scheduler.step()
            loss_sum += loss.item() * images.size(0)
            correct += (logits.argmax(dim=1) == labels).sum().item()
            seen += labels.size(0)

        val = evaluate(model, loaders['val'], device)
        history['train_loss'].append(loss_sum / seen)
        history['train_acc'].append(correct / seen)
        history['val_loss'].append(val['loss'])
        history['val_acc'].append(val['acc'])
        if val['acc'] > best_val_acc:
            best_val_acc, best_epoch = val['acc'], epoch
            best_state = copy.deepcopy(model.state_dict())

        print(f'Epoch {epoch:03d}/{epochs} | train loss {loss_sum / seen:.4f} acc {correct / seen:.4f} | '
              f'val loss {val["loss"]:.4f} acc {val["acc"]:.4f} | {time.time() - start:.0f}s', flush=True)

    model.load_state_dict(best_state)
    print(f'Best validation accuracy: {best_val_acc:.4f} (epoch {best_epoch})')
    return model, history, best_val_acc, best_epoch, time.time() - start
