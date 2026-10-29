"""Model definitions. Every network is randomly initialized (no pretrained weights anywhere)."""
import torch.nn as nn
from torchvision.models import resnet18

from .vit import ViT


def build_encoder(cfg):
    """Backbone that maps a 1×H×W grayscale image to a feature vector. Returns (encoder, feat_dim)."""
    name = cfg['model']['name']
    if name == 'resnet18':
        net = resnet18(weights=None)
        # Grayscale input: 1 input channel instead of 3.
        net.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        feat_dim = net.fc.in_features
        net.fc = nn.Identity()
        return net, feat_dim
    if name == 'vit':
        m = cfg['model']
        net = ViT(cfg['data']['img_size'], m['patch_size'], m['dim'], m['depth'], m['heads'])
        return net, m['dim']
    raise ValueError(f'Unknown model: {name}')


class Classifier(nn.Module):
    """Encoder + linear classification layer. Encoder weights live under `encoder.` so that an
    SSL-pretrained encoder can be loaded into it directly."""

    def __init__(self, encoder, feat_dim, num_classes, dropout=0.0):
        super().__init__()
        self.encoder = encoder
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(feat_dim, num_classes)

    def forward(self, x):
        return self.fc(self.dropout(self.encoder(x)))


def build_classifier(cfg, num_classes):
    encoder, feat_dim = build_encoder(cfg)
    return Classifier(encoder, feat_dim, num_classes, dropout=cfg['model'].get('dropout', 0.0))
