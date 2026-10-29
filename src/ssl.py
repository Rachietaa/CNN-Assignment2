"""Self-supervised learning components: multi-view augmentation, projection head, SimCLR loss, kNN monitor."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

from .data import MEAN, STD


def ssl_view_transform(img_size, aug):
    """One random "view" of an image. `aug` is the `data.ssl_augment` config section.

    The images are grayscale, so SimCLR's most important augmentation (color distortion) is not
    available; random cropping, brightness/contrast and blur have to provide the variation instead.
    No vertical flip and only small rotations: scenes have a fixed "up" (A1 Steps 2d-i/ii).
    """
    t = []
    if aug.get('rotation'):
        t.append(transforms.RandomRotation(aug['rotation'],
                                           interpolation=transforms.InterpolationMode.BILINEAR,
                                           fill=128))
    t.append(transforms.RandomResizedCrop(img_size, scale=tuple(aug['crop_scale'])))
    if aug.get('hflip'):
        t.append(transforms.RandomHorizontalFlip(aug['hflip']))
    if aug.get('brightness') or aug.get('contrast'):
        jitter = transforms.ColorJitter(brightness=aug.get('brightness', 0), contrast=aug.get('contrast', 0))
        t.append(transforms.RandomApply([jitter], p=aug.get('jitter_p', 1.0)))
    if aug.get('blur_p'):
        # Kernel ≈ 10% of the image side, as in SimCLR. SimCLR's sigma range (0.1, 2.0) is in pixels at
        # 224px; at smaller sizes it blurs more relative to the content, so it is set per config.
        kernel = int(0.1 * img_size) // 2 * 2 + 1
        sigma = tuple(aug.get('blur_sigma', (0.1, 2.0)))
        t.append(transforms.RandomApply([transforms.GaussianBlur(kernel, sigma=sigma)], p=aug['blur_p']))
    t += [transforms.ToTensor(), transforms.Normalize(MEAN, STD)]
    return transforms.Compose(t)


class TwoViews:
    """Apply the same random transform twice to one image -> a positive pair."""

    def __init__(self, transform):
        self.transform = transform

    def __call__(self, img):
        return self.transform(img), self.transform(img)


class ProjectionHead(nn.Module):
    """MLP g(·) that maps encoder features h to the space z where the SSL loss is applied. It is thrown
    away after pretraining: SimCLR showed h (before the head) transfers better than z."""

    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.BatchNorm1d(hidden_dim),
                                 nn.ReLU(inplace=True), nn.Linear(hidden_dim, out_dim))

    def forward(self, x):
        return self.net(x)


def nt_xent(z1, z2, temperature):
    """SimCLR loss (normalized temperature-scaled cross-entropy, InfoNCE).

    For a batch of N images there are 2N views. For each view, its partner view (same image) is the
    positive and the other 2N-2 views are negatives. The loss is a (2N-1)-way classification: find
    the partner among all other views using cosine similarity / temperature.
    """
    n = z1.size(0)
    z = F.normalize(torch.cat([z1, z2]), dim=1)
    sim = z @ z.t() / temperature
    sim.fill_diagonal_(float('-inf'))  # a view is not its own positive or negative
    targets = torch.cat([torch.arange(n, 2 * n), torch.arange(0, n)]).to(z.device)
    return F.cross_entropy(sim, targets)


@torch.inference_mode()
def extract_features(encoder, loader, device):
    encoder.eval()
    feats, labels = [], []
    for images, y in loader:
        feats.append(F.normalize(encoder(images.to(device)), dim=1).cpu())
        labels.append(y)
    return torch.cat(feats), torch.cat(labels)


@torch.inference_mode()
def knn_accuracy(encoder, train_loader, val_loader, device, k=20, temperature=0.07, num_classes=16):
    """Monitor only: weighted k-nearest-neighbour accuracy of frozen features (cosine similarity).
    Labels are used here to *measure* representation quality, never to train the encoder."""
    train_x, train_y = extract_features(encoder, train_loader, device)
    val_x, val_y = extract_features(encoder, val_loader, device)
    sim = val_x @ train_x.t()
    top_sim, top_idx = sim.topk(k, dim=1)
    weights = (top_sim / temperature).exp()
    votes = torch.zeros(val_x.size(0), num_classes).scatter_add_(1, train_y[top_idx], weights)
    return (votes.argmax(dim=1) == val_y).float().mean().item()
