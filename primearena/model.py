from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
from torch import nn

from .config import ModelConfig


class ResidualMLPBlock(nn.Module):
    """Small residual block for stable policy/value training."""

    def __init__(self, dim: int, dropout: float = 0.0, layer_norm: bool = True):
        super().__init__()
        self.norm = nn.LayerNorm(dim) if layer_norm else nn.Identity()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(dim, dim),
        )
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.net(self.norm(x)))


class ConvResidualBlock(nn.Module):
    def __init__(self, dim: int, kernel_size: int = 5, dropout: float = 0.0, layer_norm: bool = True):
        super().__init__()
        pad = kernel_size // 2
        self.norm = nn.LayerNorm(dim) if layer_norm else nn.Identity()
        self.conv = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size=kernel_size, padding=pad),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Conv1d(dim, dim, kernel_size=kernel_size, padding=pad),
        )
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.norm(x)
        y = y.transpose(1, 2)
        y = self.conv(y).transpose(1, 2)
        return self.act(x + y)


class PolicyValueNet(nn.Module):
    """Policy/value net with three interchangeable backbones.

    All variants emit logits in the PrimeArena fixed action layout:
      filters | candidate tests | candidate guesses | expand.

    `candidate_transformer` and `candidate_conv` are the important v0.3 additions:
    they preserve per-candidate structure instead of flattening the entire window,
    making both learning and interpretability cleaner.
    """

    def __init__(self, observation_dim: int, action_count: int, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.observation_dim = int(observation_dim)
        self.action_count = int(action_count)
        self.num_filters, self.window_size, self.candidate_feature_dim, self.global_feature_dim = infer_primearena_shape(
            observation_dim, action_count
        )
        arch = (cfg.architecture or "residual_mlp").lower()
        self.architecture = arch

        if arch == "residual_mlp":
            self.input = nn.Linear(observation_dim, cfg.hidden_dim)
            self.input_norm = nn.LayerNorm(cfg.hidden_dim) if cfg.layer_norm else nn.Identity()
            self.input_act = nn.GELU()
            if cfg.residual:
                self.backbone = nn.Sequential(
                    *[ResidualMLPBlock(cfg.hidden_dim, cfg.dropout, cfg.layer_norm) for _ in range(cfg.layers)]
                )
            else:
                layers = []
                for _ in range(cfg.layers):
                    layers.append(nn.Linear(cfg.hidden_dim, cfg.hidden_dim))
                    layers.append(nn.GELU())
                    if cfg.dropout > 0:
                        layers.append(nn.Dropout(cfg.dropout))
                    if cfg.layer_norm:
                        layers.append(nn.LayerNorm(cfg.hidden_dim))
                self.backbone = nn.Sequential(*layers)
            self.head_norm = nn.LayerNorm(cfg.hidden_dim) if cfg.layer_norm else nn.Identity()
            self.policy_head = nn.Linear(cfg.hidden_dim, action_count)
            self.value_head = nn.Sequential(
                nn.Linear(cfg.hidden_dim, cfg.hidden_dim), nn.GELU(), nn.Linear(cfg.hidden_dim, 1), nn.Tanh()
            )
            return

        self.candidate_in = nn.Linear(self.candidate_feature_dim, cfg.hidden_dim)
        self.global_in = nn.Linear(self.global_feature_dim, cfg.hidden_dim)
        self.candidate_norm = nn.LayerNorm(cfg.hidden_dim) if cfg.layer_norm else nn.Identity()
        self.global_norm = nn.LayerNorm(cfg.hidden_dim) if cfg.layer_norm else nn.Identity()
        self.pos = (
            nn.Parameter(torch.zeros(1, self.window_size, cfg.hidden_dim))
            if cfg.use_positional_embeddings
            else None
        )

        if arch == "candidate_transformer":
            enc_layer = nn.TransformerEncoderLayer(
                d_model=cfg.hidden_dim,
                nhead=max(1, cfg.n_heads),
                dim_feedforward=cfg.hidden_dim * max(1, cfg.ff_mult),
                dropout=cfg.dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.token_backbone = nn.TransformerEncoder(enc_layer, num_layers=max(1, cfg.layers))
        elif arch == "candidate_conv":
            self.token_backbone = nn.Sequential(
                *[
                    ConvResidualBlock(cfg.hidden_dim, cfg.conv_kernel, cfg.dropout, cfg.layer_norm)
                    for _ in range(max(1, cfg.layers))
                ]
            )
        else:
            raise ValueError(f"Unknown model architecture: {cfg.architecture}")

        self.pooled_norm = nn.LayerNorm(cfg.hidden_dim) if cfg.layer_norm else nn.Identity()
        self.filter_head = nn.Linear(cfg.hidden_dim, self.num_filters)
        self.test_head = nn.Linear(cfg.hidden_dim, 1)
        self.guess_head = nn.Linear(cfg.hidden_dim, 1)
        self.expand_head = nn.Linear(cfg.hidden_dim, 1)
        self.value_head = nn.Sequential(
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim), nn.GELU(), nn.Linear(cfg.hidden_dim, 1), nn.Tanh()
        )

    def _split_observation(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        cand_len = self.window_size * self.candidate_feature_dim
        cand = obs[:, :cand_len].reshape(obs.shape[0], self.window_size, self.candidate_feature_dim)
        glob = obs[:, cand_len : cand_len + self.global_feature_dim]
        return cand, glob

    def forward(
        self,
        obs: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
        return_activations: bool = False,
    ):
        activations: Dict[str, torch.Tensor] = {}
        if self.architecture == "residual_mlp":
            h = self.input_act(self.input_norm(self.input(obs)))
            if return_activations:
                activations["layer_0_pooled"] = h
            if isinstance(self.backbone, nn.Sequential):
                for i, layer in enumerate(self.backbone, start=1):
                    h = layer(h)
                    if return_activations:
                        activations[f"layer_{i}_pooled"] = h
            else:
                h = self.backbone(h)
                if return_activations:
                    activations["layer_1_pooled"] = h
            h = self.head_norm(h)
            activations["pooled"] = h
            logits = self.policy_head(h)
            value = self.value_head(h).squeeze(-1)
        else:
            cand, glob = self._split_observation(obs)
            cand_h = self.candidate_norm(self.candidate_in(cand))
            if self.pos is not None:
                cand_h = cand_h + self.pos
            global_token = self.global_norm(self.global_in(glob)).unsqueeze(1)
            if self.architecture == "candidate_transformer":
                tokens = torch.cat([global_token, cand_h], dim=1)
                if return_activations:
                    activations["layer_0_pooled"] = tokens[:, 0, :]
                    activations["layer_0_candidate_tokens"] = tokens[:, 1:, :]
                out = tokens
                for i, layer in enumerate(self.token_backbone.layers, start=1):
                    out = layer(out)
                    if return_activations:
                        activations[f"layer_{i}_pooled"] = out[:, 0, :]
                        activations[f"layer_{i}_candidate_tokens"] = out[:, 1:, :]
                norm = getattr(self.token_backbone, "norm", None)
                if norm is not None:
                    out = norm(out)
                pooled = out[:, 0, :]
                cand_out = out[:, 1:, :]
            else:
                # Conv backbone only operates over candidates; global context is added as bias.
                cand_out = cand_h + global_token
                if return_activations:
                    activations["layer_0_candidate_tokens"] = cand_out
                    activations["layer_0_pooled"] = cand_out.mean(dim=1) + global_token.squeeze(1)
                if isinstance(self.token_backbone, nn.Sequential):
                    for i, layer in enumerate(self.token_backbone, start=1):
                        cand_out = layer(cand_out)
                        if return_activations:
                            activations[f"layer_{i}_candidate_tokens"] = cand_out
                            activations[f"layer_{i}_pooled"] = cand_out.mean(dim=1) + global_token.squeeze(1)
                else:
                    cand_out = self.token_backbone(cand_out)
                    if return_activations:
                        activations["layer_1_candidate_tokens"] = cand_out
                        activations["layer_1_pooled"] = cand_out.mean(dim=1) + global_token.squeeze(1)
                pooled = cand_out.mean(dim=1) + global_token.squeeze(1)
            pooled = self.pooled_norm(pooled)
            activations["candidate_tokens"] = cand_out
            activations["pooled"] = pooled
            filter_logits = self.filter_head(pooled)
            test_logits = self.test_head(cand_out).squeeze(-1)
            guess_logits = self.guess_head(cand_out).squeeze(-1)
            expand_logit = self.expand_head(pooled)
            logits = torch.cat([filter_logits, test_logits, guess_logits, expand_logit], dim=-1)
            value = self.value_head(pooled).squeeze(-1)

        if action_mask is not None:
            mask_value = torch.finfo(logits.dtype).min
            logits = logits.masked_fill(action_mask <= 0, mask_value)
        if return_activations:
            return logits, value, activations
        return logits, value


def infer_primearena_shape(observation_dim: int, action_count: int) -> Tuple[int, int, int, int]:
    """Infer (num_filters, window_size, candidate_feature_dim, global_feature_dim).

    The equations can have multiple integer solutions for very small windows.
    PrimeArena always has a modest number of filter actions and a larger
    candidate window, so choose the valid solution with the smallest
    num_filters / largest window.
    """
    global_dim = 7
    candidates: list[Tuple[int, int, int, int]] = []
    for window_size in range(1, max(2, action_count // 2 + 1)):
        num_filters = action_count - 2 * window_size - 1
        if num_filters < 0:
            continue
        candidate_feature_dim = 6 + 2 * num_filters
        if window_size * candidate_feature_dim + global_dim == observation_dim:
            candidates.append((num_filters, window_size, candidate_feature_dim, global_dim))
    if candidates:
        candidates.sort(key=lambda x: (x[0], -x[1]))
        return candidates[0]
    raise ValueError(
        f"Could not infer PrimeArena shape from observation_dim={observation_dim}, action_count={action_count}."
    )


def choose_device(device: str = "auto") -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def save_checkpoint(
    path: str,
    model: PolicyValueNet,
    optimizer: Optional[torch.optim.Optimizer],
    step: int,
    extra: Optional[dict] = None,
) -> None:
    payload = {
        "model_state": model.state_dict(),
        "step": step,
        "extra": extra or {},
    }
    if optimizer is not None:
        payload["optimizer_state"] = optimizer.state_dict()
    torch.save(payload, path)


def load_checkpoint(
    path: str,
    model: PolicyValueNet,
    optimizer: Optional[torch.optim.Optimizer] = None,
    map_location: str | torch.device = "cpu",
) -> dict:
    payload = torch.load(path, map_location=map_location)
    model.load_state_dict(payload["model_state"])
    if optimizer is not None and "optimizer_state" in payload:
        optimizer.load_state_dict(payload["optimizer_state"])
    return payload
