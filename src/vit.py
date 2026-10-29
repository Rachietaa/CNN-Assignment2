"""Vision Transformer encoder and MAE decoder, written from scratch (random init, no pretrained weights)."""
import torch
import torch.nn as nn
import torch.nn.functional as F


def sincos_pos_embed(dim, grid):
    """Fixed 2-D sine-cosine position embedding, shape [1, grid*grid, dim] (as in MAE)."""
    ys, xs = torch.meshgrid(torch.arange(grid, dtype=torch.float32),
                            torch.arange(grid, dtype=torch.float32), indexing='ij')
    omega = 1.0 / 10000 ** (torch.arange(dim // 4, dtype=torch.float32) / (dim // 4))
    parts = []
    for pos in (ys.flatten(), xs.flatten()):
        angles = pos[:, None] * omega[None, :]
        parts += [angles.sin(), angles.cos()]
    return torch.cat(parts, dim=1)[None]


class Block(nn.Module):
    """Pre-norm transformer block: self-attention + MLP, each with a residual connection."""

    def __init__(self, dim, heads, mlp_ratio=4.0):
        super().__init__()
        self.heads = heads
        self.norm1 = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))

    def forward(self, x):
        b, n, d = x.shape
        q, k, v = self.qkv(self.norm1(x)).reshape(b, n, 3, self.heads, d // self.heads).permute(2, 0, 3, 1, 4)
        attn = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(b, n, d)
        x = x + self.proj(attn)
        return x + self.mlp(self.norm2(x))


def init_weights(module):
    if isinstance(module, nn.Linear):
        nn.init.trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.LayerNorm):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)


class ViT(nn.Module):
    """ViT encoder for 1-channel images. forward() returns the mean of the patch tokens (no class token),
    so it plugs into the same Classifier as the ResNet encoder."""

    def __init__(self, img_size, patch_size, dim, depth, heads, in_channels=1):
        super().__init__()
        self.patch_size, self.grid = patch_size, img_size // patch_size
        self.patch_embed = nn.Conv2d(in_channels, dim, kernel_size=patch_size, stride=patch_size)
        self.register_buffer('pos', sincos_pos_embed(dim, self.grid), persistent=False)
        self.blocks = nn.ModuleList([Block(dim, heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.apply(init_weights)
        w = self.patch_embed.weight.data
        nn.init.xavier_uniform_(w.view(w.shape[0], -1))

    def tokens(self, x):
        return self.patch_embed(x).flatten(2).transpose(1, 2) + self.pos

    def forward(self, x):
        t = self.tokens(x)
        for block in self.blocks:
            t = block(t)
        return self.norm(t).mean(dim=1)

    def forward_masked(self, x, mask_ratio):
        """MAE encoder pass: keep a random (1 - mask_ratio) subset of patches per image and encode only
        those. Returns the visible-token features, the binary mask (1 = hidden) in the original patch
        order, and the permutation needed to put tokens back in order."""
        t = self.tokens(x)
        b, n, d = t.shape
        keep = int(n * (1 - mask_ratio))
        ids_shuffle = torch.rand(b, n, device=x.device).argsort(dim=1)
        ids_restore = ids_shuffle.argsort(dim=1)
        t = torch.gather(t, 1, ids_shuffle[:, :keep, None].expand(-1, -1, d))
        for block in self.blocks:
            t = block(t)
        mask = torch.ones(b, n, device=x.device)
        mask[:, :keep] = 0
        mask = torch.gather(mask, 1, ids_restore)
        return self.norm(t), mask, ids_restore


class MAEDecoder(nn.Module):
    """Light transformer that receives the visible-patch features plus a shared learned "mask token" at
    every hidden position, and predicts the pixels of every patch. Thrown away after pretraining."""

    def __init__(self, enc_dim, dec_dim, depth, heads, grid, patch_size, in_channels=1):
        super().__init__()
        self.embed = nn.Linear(enc_dim, dec_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, dec_dim))
        self.register_buffer('pos', sincos_pos_embed(dec_dim, grid), persistent=False)
        self.blocks = nn.ModuleList([Block(dec_dim, heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(dec_dim)
        self.pred = nn.Linear(dec_dim, patch_size * patch_size * in_channels)
        self.apply(init_weights)
        nn.init.normal_(self.mask_token, std=0.02)

    def forward(self, latent, ids_restore):
        x = self.embed(latent)
        b, keep, d = x.shape
        n = ids_restore.shape[1]
        x = torch.cat([x, self.mask_token.expand(b, n - keep, d)], dim=1)
        x = torch.gather(x, 1, ids_restore[:, :, None].expand(-1, -1, d)) + self.pos
        for block in self.blocks:
            x = block(x)
        return self.pred(self.norm(x))


def patchify(images, patch_size):
    """[B, C, H, W] -> [B, num_patches, patch_size*patch_size*C], row-major patch order."""
    b, c, h, w = images.shape
    g = h // patch_size
    x = images.reshape(b, c, g, patch_size, g, patch_size)
    return x.permute(0, 2, 4, 3, 5, 1).reshape(b, g * g, patch_size * patch_size * c)


def mae_loss(pred, images, mask, patch_size, norm_pix=True):
    """Mean squared error on the hidden patches only. norm_pix: each target patch is normalized to zero
    mean / unit variance, so the model must predict structure (edges, layout) rather than brightness."""
    target = patchify(images, patch_size)
    if norm_pix:
        mean, var = target.mean(dim=-1, keepdim=True), target.var(dim=-1, keepdim=True)
        target = (target - mean) / (var + 1e-6).sqrt()
    per_patch = ((pred - target) ** 2).mean(dim=-1)
    return (per_patch * mask).sum() / mask.sum()
