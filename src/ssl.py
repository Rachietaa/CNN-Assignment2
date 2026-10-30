"""Self-supervised learning components: view augmentation, SimCLR (projection head, NT-Xent loss),
DINO (multi-crop, head, self-distillation loss, EMA teacher) and a kNN monitor."""
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


class MultiCropViews:
    """DINO multi-crop: 2 global views (large crops, full resolution) + n local views (small crops, low
    resolution). The teacher only sees global views; the student sees all of them, so it must predict
    the teacher's output for the whole scene from a small part of it ("local-to-global")."""

    def __init__(self, global_transform, local_transform, n_local):
        self.global_transform, self.local_transform, self.n_local = global_transform, local_transform, n_local

    def __call__(self, img):
        views = [self.global_transform(img), self.global_transform(img)]
        return views + [self.local_transform(img) for _ in range(self.n_local)]


class DINOHead(nn.Module):
    """MLP -> L2-normalized bottleneck -> cosine similarity to K learned prototypes.
    Normalizing both the features and the prototype weights equals DINO's weight-normalized last layer
    with the norm fixed to 1."""

    def __init__(self, in_dim, hidden_dim, bottleneck_dim, out_dim, use_bn=False, norm_last_layer=True):
        super().__init__()
        # use_bn: BatchNorm after each hidden layer (DINO's `use_bn_in_head` option). At random init a CNN
        # maps all images to nearly the same feature vector; BatchNorm rescales the differences between
        # images in a batch to unit size, so the centered teacher output is not near-uniform.
        def block(i, o):
            return [nn.Linear(i, o)] + ([nn.BatchNorm1d(o)] if use_bn else []) + [nn.GELU()]
        self.mlp = nn.Sequential(*block(in_dim, hidden_dim), *block(hidden_dim, hidden_dim),
                                 nn.Linear(hidden_dim, bottleneck_dim))
        self.prototypes = nn.Parameter(torch.randn(out_dim, bottleneck_dim) * 0.02)
        # norm_last_layer=False: each prototype also gets a learnable length g (init 1), so the logits are
        # no longer bounded to [-1, 1] and the outputs can become sharp (DINO's `norm_last_layer` option).
        self.scale = None if norm_last_layer else nn.Parameter(torch.ones(out_dim))

    def forward(self, x):
        logits = F.normalize(self.mlp(x), dim=1) @ F.normalize(self.prototypes, dim=1).t()
        return logits if self.scale is None else logits * self.scale


class DINOLoss(nn.Module):
    """Self-distillation without labels.

    The teacher's output is turned into a target distribution over K prototypes: *centered* (subtract
    a running mean, so no single prototype can win for every image) and *sharpened* (low temperature,
    so the target is not uniform). Centering alone would collapse to uniform, sharpening alone to one
    prototype; together they avoid collapse without negative pairs. The student is trained with
    cross-entropy to match the teacher's distribution for every *other* view of the same image.
    """

    def __init__(self, out_dim, student_temp, teacher_temp, center_momentum):
        super().__init__()
        self.student_temp, self.teacher_temp, self.center_momentum = student_temp, teacher_temp, center_momentum
        self.register_buffer('center', torch.zeros(1, out_dim))

    def forward(self, student_out, teacher_out):
        """student_out: list of [B, K] logits for every view; teacher_out: list for the 2 global views.
        Returns the loss and the teacher target probabilities (for collapse monitoring)."""
        targets = [F.softmax((t - self.center) / self.teacher_temp, dim=1).detach() for t in teacher_out]
        log_preds = [F.log_softmax(s / self.student_temp, dim=1) for s in student_out]
        total, n_terms = 0.0, 0
        for i, target in enumerate(targets):
            for j, log_pred in enumerate(log_preds):
                if i == j:  # same global view: nothing to predict
                    continue
                total = total + (-target * log_pred).sum(dim=1).mean()
                n_terms += 1
        self.update_center(torch.cat(teacher_out))
        return total / n_terms, torch.cat(targets)

    @torch.no_grad()
    def update_center(self, teacher_logits):
        self.center.mul_(self.center_momentum).add_(teacher_logits.mean(dim=0, keepdim=True),
                                                    alpha=1 - self.center_momentum)


@torch.no_grad()
def ema_update(teacher, student, momentum):
    """Teacher weights = exponential moving average of student weights (the teacher gets no gradients)."""
    for pt, ps in zip(teacher.parameters(), student.parameters()):
        pt.mul_(momentum).add_(ps.detach(), alpha=1 - momentum)


def collapse_stats(probs):
    """Per-image entropy of the teacher targets (low = confident) and entropy of their batch mean
    (low = every image maps to the same few prototypes = collapse). Max for both is log(K)."""
    eps = 1e-12
    per_image = -(probs * (probs + eps).log()).sum(dim=1).mean().item()
    mean = probs.mean(dim=0)
    batch = -(mean * (mean + eps).log()).sum().item()
    return per_image, batch


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
