"""Joint MAE + contrastive (InfoNCE) pretraining model around a timm ViT-S/16 encoder.

- MAE branch (view 1): random per-sample masking (ratio 0.75); the encoder sees only the
  visible patch tokens (+cls); a light decoder (4 blocks, dim 256, 8 heads, fixed 2D sin-cos
  positions) predicts per-patch normalised pixels; the loss is averaged over masked patches only.
- Contrastive branch (both views): full, unmasked encoder passes; representation = mean of the
  patch tokens (cls excluded); projection MLP; symmetric NT-Xent with in-batch negatives.
- L = w_mae * L_mae + w_con * L_con with fixed weights (lambda_mae, lambda_con) or Kendall et al.
  (2018) uncertainty weighting: L = sum_i exp(-s_i) * L_i + s_i with learnable s_i = log(sigma_i^2).
A branch whose lambda is 0 is not computed at all.

Phase 7 (SSL v2) optional branches, each switched on by its own config block (None = off):
- `dino` (self-distillation, Caron et al. 2021): EMA teacher (encoder + head) of the student; student views =
  the two global views + `local_crops.n` small local crops, teacher views = the two global views; DINO head on the
  mean of the patch tokens (the feature the probe / DETR use), prototypes L2-normalised (norm_last_layer);
  cross-entropy between the sharpened, centred teacher softmax and the student softmax over all pairs of different
  views. During `head_warmup_epochs` the DINO loss does not reach the student encoder (fresh random head).
  The teacher encoder is the exported encoder.
- `stain_consistency`: view 1 under a strong stain / resolution / JPEG shift (`twin`, same geometry);
  loss = mean over patches of 1 - cos(twin token, stop-grad clean token).

Only the encoder is exported (`encoder_state_dict`); it is a plain timm VisionTransformer state
dict (num_classes=0) that loads with strict=True into `build_encoder`.
"""
from __future__ import annotations

import copy
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


class DINOHead(nn.Module):
    """DINO projection head: MLP (GELU) -> L2-normalised bottleneck -> cosine to L2-normalised prototypes."""

    def __init__(self, dim: int, hidden: int, bottleneck: int, n_prototypes: int, n_layers: int = 3):
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(dim, hidden), nn.GELU()]
        for _ in range(n_layers - 2):
            layers += [nn.Linear(hidden, hidden), nn.GELU()]
        layers.append(nn.Linear(hidden, bottleneck))
        self.mlp = nn.Sequential(*layers)
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                nn.init.zeros_(m.bias)
        self.prototypes = nn.Parameter(torch.empty(n_prototypes, bottleneck))
        nn.init.trunc_normal_(self.prototypes, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = F.normalize(self.mlp(x).float(), dim=-1)
        return z @ F.normalize(self.prototypes.float(), dim=-1).t()


def dino_loss(student: list[torch.Tensor], teacher_probs: list[torch.Tensor], student_temp: float) -> torch.Tensor:
    """Mean over (teacher view i, student view v != i) of CE(teacher_probs[i], softmax(student[v] / T_s)).
    The first len(teacher_probs) student views are the same global views as the teacher's."""
    terms = []
    for i, q in enumerate(teacher_probs):
        for v, s in enumerate(student):
            if v != i:
                terms.append(torch.sum(-q * F.log_softmax(s.float() / student_temp, dim=-1), dim=-1).mean())
    return torch.stack(terms).mean()


def stain_consistency_loss(twin_tokens: torch.Tensor, clean_tokens: torch.Tensor) -> torch.Tensor:
    """Mean over patches of 1 - cos(twin token, stop-grad clean token); tokens (B, N, C) of the same geometry."""
    return (1 - F.cosine_similarity(twin_tokens.float(), clean_tokens.detach().float(), dim=-1)).mean()


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
        self.dino_cfg, self.stain_cfg = cfg.get("dino"), cfg.get("stain_consistency")
        self.lambda_dino = float(self.dino_cfg["lambda"]) if self.dino_cfg else 0.0
        self.lambda_stain = float(self.stain_cfg["lambda"]) if self.stain_cfg else 0.0
        lams = (self.lambda_mae, self.lambda_con, self.lambda_dino, self.lambda_stain)
        if min(lams) < 0 or sum(lams) == 0:
            raise ValueError("lambdas must be >= 0 and not all 0")
        self.weighting = lc["loss_weighting"]
        if self.weighting not in WEIGHTINGS:
            raise ValueError(f"loss_weighting must be one of {WEIGHTINGS}")
        if self.weighting == "uncertainty":
            if not (self.use_mae and self.use_con) or self.use_dino or self.use_stain:
                raise ValueError("uncertainty weighting needs exactly the MAE and InfoNCE branches")
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
        self.dino_backbone_grad = True   # False during the DINO head warm-up (set by the trainer)
        if self.use_dino:
            d = self.dino_cfg
            self.dino_head = DINOHead(dim, d["head_hidden"], d["head_bottleneck"], d["n_prototypes"])
            self.teacher_encoder = copy.deepcopy(encoder)
            self.teacher_head = copy.deepcopy(self.dino_head)
            for p in (*self.teacher_encoder.parameters(), *self.teacher_head.parameters()):
                p.requires_grad = False
            self.register_buffer("center", torch.zeros(1, d["n_prototypes"]))

    @property
    def use_mae(self) -> bool:
        return self.lambda_mae > 0

    @property
    def use_con(self) -> bool:
        return self.lambda_con > 0

    @property
    def use_dino(self) -> bool:
        return self.lambda_dino > 0

    @property
    def use_stain(self) -> bool:
        return self.lambda_stain > 0

    # -- encoder -------------------------------------------------------------------------
    def encode_full(self, x: torch.Tensor) -> torch.Tensor:
        """Mean of the final-normed patch tokens (cls excluded), (B, dim)."""
        return self.encode_tokens(x).mean(1)

    def encode_tokens(self, x: torch.Tensor, encoder: nn.Module | None = None) -> torch.Tensor:
        """Final-normed patch tokens (cls excluded), (B, N, dim)."""
        return (encoder or self.encoder).forward_features(x)[:, self.prefix:]

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
        w = {"w_mae": self.lambda_mae, "w_con": self.lambda_con}
        if self.use_dino:
            w["w_dino"] = self.lambda_dino
        if self.use_stain:
            w["w_stain"] = self.lambda_stain
        return w

    def dino_branch(self, f1: torch.Tensor, f2: torch.Tensor, v1: torch.Tensor, v2: torch.Tensor,
                    local: torch.Tensor | None, out: dict[str, Any]) -> torch.Tensor:
        """DINO loss from the student's pooled global features f1, f2 (+ local crops (B, n, 3, h, w))."""
        d = self.dino_cfg
        feats = [f1, f2]
        if local is not None and local.shape[1] > 0:
            b, n = local.shape[:2]
            fl = self.encode_tokens(local.flatten(0, 1)).mean(1)
            feats += list(fl.view(b, n, -1).transpose(0, 1))
        if not self.dino_backbone_grad:
            feats = [f.detach() for f in feats]
        student = [self.dino_head(f) for f in feats]
        with torch.no_grad():
            t = [self.teacher_head(self.encode_tokens(v, self.teacher_encoder).mean(1)) for v in (v1, v2)]
            probs = [F.softmax((x - self.center) / d["teacher_temp"], dim=-1) for x in t]
            if self.training:
                batch_center = torch.cat(t).mean(0, keepdim=True)
                self.center.mul_(d["center_momentum"]).add_(batch_center, alpha=1 - d["center_momentum"])
            p = torch.cat(probs)
            out["dino_teacher_entropy"] = -(p * p.clamp_min(1e-12).log()).sum(-1).mean()
            pm = p.mean(0)
            out["dino_teacher_mean_entropy"] = -(pm * pm.clamp_min(1e-12).log()).sum()
        return dino_loss(student, probs, d["student_temp"])

    def forward(self, v1: torch.Tensor, v2: torch.Tensor, noise: torch.Tensor | None = None,
                local: torch.Tensor | None = None, twin: torch.Tensor | None = None) -> dict[str, Any]:
        out: dict[str, Any] = {}
        losses = {}
        if self.use_mae:
            losses["mae"] = self.mae_forward(v1, noise)["loss"]
        if self.use_con or self.use_dino or self.use_stain:
            tok1 = self.encode_tokens(v1)
            h1, h2 = tok1.mean(1), self.encode_full(v2)
            if self.use_con:
                z1, z2 = self.projector(h1), self.projector(h2)
                losses["con"] = info_nce(z1, z2, self.temperature)
                out.update(z1=z1.detach(), z2=z2.detach())
            if self.use_dino:
                losses["dino"] = self.dino_branch(h1, h2, v1, v2, local, out)
            if self.use_stain:
                if twin is None:
                    raise ValueError("stain consistency needs the twin view")
                losses["stain"] = stain_consistency_loss(self.encode_tokens(twin), tok1)
            out.update(h1=h1.detach(), h2=h2.detach())
        if self.weighting == "uncertainty":
            s = self.log_vars
            total = torch.exp(-s[0]) * losses["mae"] + s[0] + torch.exp(-s[1]) * losses["con"] + s[1]
        else:
            lams = {"mae": self.lambda_mae, "con": self.lambda_con, "dino": self.lambda_dino, "stain": self.lambda_stain}
            total = sum(lams[k] * v for k, v in losses.items())
        out["loss"] = total
        for k in ("mae", "con", "dino", "stain"):
            out[f"loss_{k}"] = losses.get(k)
        return out

    @torch.no_grad()
    def update_teacher(self, momentum: float) -> None:
        """EMA of the student encoder + DINO head into the teacher (no-op without the DINO branch)."""
        if not self.use_dino:
            return
        for src, dst in ((self.encoder, self.teacher_encoder), (self.dino_head, self.teacher_head)):
            for ps, pt in zip(src.parameters(), dst.parameters()):
                pt.mul_(momentum).add_(ps.detach(), alpha=1 - momentum)

    def export_encoder(self) -> nn.Module:
        """The encoder that is exported / probed: the EMA teacher with the DINO branch, else the student."""
        return self.teacher_encoder if self.use_dino else self.encoder

    def encoder_state_dict(self, student: bool = False) -> dict[str, torch.Tensor]:
        enc = self.encoder if student else self.export_encoder()
        return {k: v.detach().cpu().clone() for k, v in enc.state_dict().items()}


def param_groups(model: nn.Module, weight_decay: float) -> list[dict[str, Any]]:
    """AdamW groups: no weight decay on biases, norms, tokens, position embeddings, log-variances."""
    no_decay_names = ("pos_embed", "cls_token", "mask_token", "log_vars", "prototypes")
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (no_decay if p.ndim <= 1 or any(k in name for k in no_decay_names) else decay).append(p)
    return [{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}]
