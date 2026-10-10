"""Joint MAE + contrastive (InfoNCE) pretraining model around a timm ViT-S/16 encoder.

- MAE branch (view 1): random per-sample masking (ratio 0.75); the encoder sees only the
  visible patch tokens (+cls); a light decoder (4 blocks, dim 256, 8 heads, fixed 2D sin-cos
  positions) predicts per-patch normalised pixels; the loss is averaged over masked patches only.
- Contrastive branch (both views): full, unmasked encoder passes; representation = mean of the
  patch tokens (cls excluded); projection MLP; symmetric NT-Xent with in-batch negatives.
- L = w_mae * L_mae + w_con * L_con with fixed weights (lambda_mae, lambda_con) or Kendall et al.
  (2018) uncertainty weighting: L = sum_i exp(-s_i) * L_i + s_i with learnable s_i = log(sigma_i^2).
A branch whose lambda is 0 is not computed at all.

Only the encoder is exported (`encoder_state_dict`); it is a plain timm VisionTransformer state
dict (num_classes=0) that loads with strict=True into `build_encoder`.
"""
from __future__ import annotations

from functools import partial
from typing import Any

import numpy as np
import timm
import torch
import torch.nn.functional as F
from timm.models.vision_transformer import Block
from torch import nn

WEIGHTINGS = ("fixed", "uncertainty")


def build_encoder(name: str = "vit_small_patch16_224", drop_path_rate: float = 0.0,
                  pretrained: bool = False) -> nn.Module:
    """timm ViT without head; dynamic_img_size so position embeddings interpolate for DETR inputs."""
    return timm.create_model(name, pretrained=pretrained, num_classes=0, dynamic_img_size=True,
                             drop_path_rate=drop_path_rate)


def sincos_pos_embed(dim: int, gh: int, gw: int) -> torch.Tensor:
    """Fixed 2D sin-cos position embedding (gh*gw, dim), as in MAE."""
    def one_d(d: int, pos: np.ndarray) -> np.ndarray:
        omega = 1.0 / 10000 ** (np.arange(d // 2, dtype=np.float64) / (d / 2))
        out = np.einsum("m,d->md", pos.reshape(-1), omega)
        return np.concatenate([np.sin(out), np.cos(out)], axis=1)
    gy, gx = np.meshgrid(np.arange(gh, dtype=np.float64), np.arange(gw, dtype=np.float64), indexing="ij")
    emb = np.concatenate([one_d(dim // 2, gy), one_d(dim // 2, gx)], axis=1)
    return torch.from_numpy(emb).float()


def info_nce(z1: torch.Tensor, z2: torch.Tensor, temperature: float) -> torch.Tensor:
    """Symmetric NT-Xent: positives are (i, i+B); all other 2B-2 samples are negatives."""
    b = z1.shape[0]
    z = F.normalize(torch.cat([z1, z2]).float(), dim=1)
    sim = z @ z.t() / temperature
    sim = sim.masked_fill(torch.eye(2 * b, dtype=torch.bool, device=z.device), float("-inf"))
    targets = torch.cat([torch.arange(b, 2 * b), torch.arange(0, b)]).to(z.device)
    return F.cross_entropy(sim, targets)


class SSLModel(nn.Module):
    def __init__(self, encoder: nn.Module, cfg: dict[str, Any]):
        super().__init__()
        self.encoder = encoder
        self.cfg = cfg
        self.patch = encoder.patch_embed.patch_size[0]
        self.prefix = encoder.num_prefix_tokens
        dim = encoder.embed_dim
        lc = cfg["loss"]
        self.lambda_mae, self.lambda_con = float(lc["lambda_mae"]), float(lc["lambda_con"])
        if self.lambda_mae < 0 or self.lambda_con < 0 or self.lambda_mae + self.lambda_con == 0:
            raise ValueError("lambdas must be >= 0 and not both 0")
        self.weighting = lc["loss_weighting"]
        if self.weighting not in WEIGHTINGS:
            raise ValueError(f"loss_weighting must be one of {WEIGHTINGS}")
        if self.weighting == "uncertainty":
            if not (self.use_mae and self.use_con):
                raise ValueError("uncertainty weighting needs both branches (lambdas > 0)")
            self.log_vars = nn.Parameter(torch.zeros(2))  # [mae, con]
        self.mask_ratio = cfg["mae"]["mask_ratio"]
        self.temperature = cfg["contrastive"]["temperature"]

        if self.use_mae:
            d = cfg["mae"]
            self.decoder_embed = nn.Linear(dim, d["decoder_dim"])
            self.mask_token = nn.Parameter(torch.zeros(1, 1, d["decoder_dim"]))
            self.decoder_blocks = nn.ModuleList(
                Block(d["decoder_dim"], d["decoder_heads"], mlp_ratio=4.0, qkv_bias=True,
                      norm_layer=partial(nn.LayerNorm, eps=1e-6)) for _ in range(d["decoder_depth"]))
            self.decoder_norm = nn.LayerNorm(d["decoder_dim"], eps=1e-6)
            self.decoder_pred = nn.Linear(d["decoder_dim"], self.patch ** 2 * 3)
            nn.init.normal_(self.mask_token, std=0.02)
        if self.use_con:
            c = cfg["contrastive"]
            self.projector = nn.Sequential(nn.Linear(dim, c["proj_hidden"]), nn.BatchNorm1d(c["proj_hidden"]),
                                           nn.GELU(), nn.Linear(c["proj_hidden"], c["proj_dim"]))

    @property
    def use_mae(self) -> bool:
        return self.lambda_mae > 0

    @property
    def use_con(self) -> bool:
        return self.lambda_con > 0

    # -- encoder -------------------------------------------------------------------------
    def encode_full(self, x: torch.Tensor) -> torch.Tensor:
        """Mean of the final-normed patch tokens (cls excluded), (B, dim)."""
        return self.encoder.forward_features(x)[:, self.prefix:].mean(1)

    def encode_masked(self, x: torch.Tensor, mask_ratio: float, noise: torch.Tensor | None = None):
        """Encode only a random subset of patches. mask: (B, N), 1 = masked (removed)."""
        vit = self.encoder
        t = vit._pos_embed(vit.patch_embed(x))            # (B, prefix + N, C), positions added
        prefix, t = t[:, :self.prefix], t[:, self.prefix:]
        b, n, c = t.shape
        keep = int(round(n * (1 - mask_ratio)))
        if noise is None:
            noise = torch.rand(b, n, device=x.device)
        ids_shuffle = noise.argsort(dim=1)
        ids_restore = ids_shuffle.argsort(dim=1)
        ids_keep = ids_shuffle[:, :keep]
        t = torch.gather(t, 1, ids_keep.unsqueeze(-1).expand(-1, -1, c))
        mask = torch.ones(b, n, device=x.device)
        mask[:, :keep] = 0
        mask = torch.gather(mask, 1, ids_restore)
        t = vit.norm_pre(torch.cat([prefix, t], dim=1))
        for blk in vit.blocks:
            t = blk(t)
        return vit.norm(t), mask, ids_restore

    # -- MAE -----------------------------------------------------------------------------
    def patchify(self, x: torch.Tensor) -> torch.Tensor:
        """(B, 3, H, W) -> (B, N, p*p*3), patch-major order matching the encoder tokens."""
        p = self.patch
        b, c, h, w = x.shape
        x = x.reshape(b, c, h // p, p, w // p, p)
        return torch.einsum("bchpwq->bhwpqc", x).reshape(b, (h // p) * (w // p), p * p * c)

    def unpatchify(self, t: torch.Tensor, h: int, w: int) -> torch.Tensor:
        p = self.patch
        b = t.shape[0]
        t = t.reshape(b, h // p, w // p, p, p, 3)
        return torch.einsum("bhwpqc->bchpwq", t).reshape(b, 3, h, w)

    def decode(self, latent: torch.Tensor, ids_restore: torch.Tensor, grid: tuple[int, int]) -> torch.Tensor:
        t = self.decoder_embed(latent)
        prefix, vis = t[:, :self.prefix], t[:, self.prefix:]
        b, n = ids_restore.shape
        t = torch.cat([vis, self.mask_token.expand(b, n - vis.shape[1], -1).to(vis.dtype)], dim=1)
        t = torch.gather(t, 1, ids_restore.unsqueeze(-1).expand(-1, -1, t.shape[2]))
        pos = sincos_pos_embed(t.shape[2], *grid).to(t.device, t.dtype)
        t = torch.cat([prefix, t + pos], dim=1)
        for blk in self.decoder_blocks:
            t = blk(t)
        return self.decoder_pred(self.decoder_norm(t))[:, self.prefix:]

    def mae_target(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Per-patch normalised pixel targets plus the per-patch mean / std used."""
        t = self.patchify(x.float())
        mean, var = t.mean(-1, keepdim=True), t.var(-1, keepdim=True)
        std = (var + 1e-6).sqrt()
        return (t - mean) / std, mean, std

    @staticmethod
    def mae_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        per_patch = ((pred.float() - target) ** 2).mean(-1)
        return (per_patch * mask).sum() / mask.sum()

    def mae_forward(self, x: torch.Tensor, noise: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        latent, mask, ids_restore = self.encode_masked(x, self.mask_ratio, noise)
        grid = (x.shape[2] // self.patch, x.shape[3] // self.patch)
        pred = self.decode(latent, ids_restore, grid)
        target, mean, std = self.mae_target(x)
        return {"loss": self.mae_loss(pred, target, mask), "pred": pred, "mask": mask, "mean": mean, "std": std}

    @torch.no_grad()
    def reconstruct(self, x: torch.Tensor, noise: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """(masked input, reconstruction with visible patches pasted back), in the input's space."""
        out = self.mae_forward(x, noise)
        h, w = x.shape[2:]
        m = out["mask"].unsqueeze(-1)
        patches = self.patchify(x.float())
        rec = out["pred"].float() * out["std"] + out["mean"]
        masked = self.unpatchify(patches * (1 - m), h, w)
        recon = self.unpatchify(patches * (1 - m) + rec * m, h, w)
        return masked, recon

    # -- joint ---------------------------------------------------------------------------
    def weights(self) -> dict[str, float]:
        """Effective loss weights (for logging)."""
        if self.weighting == "uncertainty":
            w = torch.exp(-self.log_vars.detach()).tolist()
            return {"w_mae": w[0], "w_con": w[1]}
        return {"w_mae": self.lambda_mae, "w_con": self.lambda_con}

    def forward(self, v1: torch.Tensor, v2: torch.Tensor, noise: torch.Tensor | None = None) -> dict[str, Any]:
        out: dict[str, Any] = {}
        losses = {}
        if self.use_mae:
            losses["mae"] = self.mae_forward(v1, noise)["loss"]
        if self.use_con:
            h1, h2 = self.encode_full(v1), self.encode_full(v2)
            z1, z2 = self.projector(h1), self.projector(h2)
            losses["con"] = info_nce(z1, z2, self.temperature)
            out.update(h1=h1.detach(), h2=h2.detach(), z1=z1.detach(), z2=z2.detach())
        if self.weighting == "uncertainty":
            s = self.log_vars
            total = torch.exp(-s[0]) * losses["mae"] + s[0] + torch.exp(-s[1]) * losses["con"] + s[1]
        else:
            total = sum(lam * losses[k] for k, lam in (("mae", self.lambda_mae), ("con", self.lambda_con)) if k in losses)
        out["loss"] = total
        out["loss_mae"] = losses.get("mae")
        out["loss_con"] = losses.get("con")
        return out

    def encoder_state_dict(self) -> dict[str, torch.Tensor]:
        return {k: v.detach().cpu().clone() for k, v in self.encoder.state_dict().items()}


def param_groups(model: nn.Module, weight_decay: float) -> list[dict[str, Any]]:
    """AdamW groups: no weight decay on biases, norms, tokens, position embeddings, log-variances."""
    no_decay_names = ("pos_embed", "cls_token", "mask_token", "log_vars")
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (no_decay if p.ndim <= 1 or any(k in name for k in no_decay_names) else decay).append(p)
    return [{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}]
