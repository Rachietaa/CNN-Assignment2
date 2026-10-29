"""Datasets, the fixed train/val split, and image transforms."""
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

# Same split as Assignment 1 (seeded randperm, 1,920 train / 480 val), so results stay comparable.
SPLIT_SEED = 0
VAL_FRACTION = 0.20

# Grayscale images in [0, 1] are normalized to [-1, 1].
MEAN, STD = [0.5], [0.5]


def load_gray(path):
    """Open an image as single-channel grayscale. Only the Flower class is in color, so color would be
    a one-class shortcut (A1 Step 3a)."""
    with open(path, 'rb') as f:
        return Image.open(f).convert('L')


def split_indices(n, val_fraction=VAL_FRACTION, seed=SPLIT_SEED):
    val_size = int(round(n * val_fraction))
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(seed)).tolist()
    return perm[:n - val_size], perm[n - val_size:]


def eval_transform(img_size):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])


def supervised_transform(img_size, aug):
    """Training augmentation for the classifier. `aug` is the `data.augment` config section."""
    if not aug:
        return eval_transform(img_size)
    t = []
    if aug.get('rotation'):
        # Rotate at full resolution with bilinear interpolation and mid-gray corners (A1: rotating
        # after downscaling produced staircase artifacts).
        t.append(transforms.RandomRotation(aug['rotation'],
                                           interpolation=transforms.InterpolationMode.BILINEAR,
                                           fill=128))
    if 'random_resized_crop' in aug:
        t.append(transforms.RandomResizedCrop(img_size, scale=tuple(aug['random_resized_crop'])))
    else:
        t.append(transforms.Resize((img_size, img_size)))
    if aug.get('hflip'):
        t.append(transforms.RandomHorizontalFlip(aug['hflip']))
    if aug.get('brightness') or aug.get('contrast'):
        t.append(transforms.ColorJitter(brightness=aug.get('brightness', 0),
                                        contrast=aug.get('contrast', 0)))
    t += [transforms.ToTensor(), transforms.Normalize(MEAN, STD)]
    return transforms.Compose(t)


def image_folder(root, split, transform):
    return datasets.ImageFolder(Path(root) / split, transform=transform, loader=load_gray)


def labelled_loaders(cfg, device, train_transform):
    """Train / val / test loaders for classification. Train and val are two views of the same folder
    so they can use different transforms."""
    data_cfg = cfg['data']
    eval_tf = eval_transform(data_cfg['img_size'])
    train_view = image_folder(data_cfg['root'], 'train', train_transform)
    val_view = image_folder(data_cfg['root'], 'train', eval_tf)
    test_set = image_folder(data_cfg['root'], 'test', eval_tf)

    train_idx, val_idx = split_indices(len(train_view))
    workers = data_cfg.get('num_workers', 4)
    kw = dict(batch_size=cfg['train']['batch_size'], num_workers=workers,
              pin_memory=device.type == 'cuda', persistent_workers=workers > 0)
    loaders = {
        'train': DataLoader(Subset(train_view, train_idx), shuffle=True, drop_last=True, **kw),
        'val': DataLoader(Subset(val_view, val_idx), shuffle=False, **kw),
        'test': DataLoader(test_set, shuffle=False, **kw),
    }
    return loaders, train_view.classes
