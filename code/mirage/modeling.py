from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class KANLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.spline_left = nn.Linear(in_dim, out_dim)
        self.spline_right = nn.Linear(in_dim, out_dim)
        self.gate = nn.Sequential(nn.Linear(in_dim, out_dim), nn.Sigmoid())
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.gate(x)
        spline = gate * torch.tanh(self.spline_left(x)) + (1.0 - gate) * F.silu(self.spline_right(x))
        return self.norm(self.linear(x) + spline)


class KANNetwork(nn.Module):
    def __init__(self, dims: Sequence[int], dropout: float = 0.1, final_activation: bool = False) -> None:
        super().__init__()
        layers = []
        for idx in range(len(dims) - 1):
            layers.append(KANLayer(dims[idx], dims[idx + 1]))
            is_last = idx == len(dims) - 2
            if not is_last or final_activation:
                layers.append(nn.GELU())
                if dropout > 0.0:
                    layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TabularKANAutoencoder(nn.Module):
    def __init__(self, input_dim: int, embed_dim: int = 512, dropout: float = 0.1) -> None:
        super().__init__()
        hidden_dim = max(256, min(1024, input_dim * 2))
        bottleneck_dim = max(256, embed_dim)
        self.encoder = KANNetwork([input_dim, hidden_dim, bottleneck_dim, embed_dim], dropout=dropout)
        self.decoder = nn.Sequential(
            nn.Linear(embed_dim, bottleneck_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(bottleneck_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        embedding = self.encoder(x)
        reconstruction = self.decoder(embedding)
        return reconstruction, embedding

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


class HierarchicalCodeEmbedder(nn.Module):
    def __init__(self, input_dim: int, embed_dim: int = 512, dropout: float = 0.1) -> None:
        super().__init__()
        hidden_dim = max(256, min(1024, input_dim))
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.LayerNorm(embed_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        embedding = self.encoder(x)
        reconstruction = self.decoder(embedding)
        return reconstruction, embedding

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


class QualityGate(nn.Module):
    def __init__(self, n_mod: int = 6, hidden: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_mod, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_mod),
            nn.Sigmoid(),
        )

    def forward(self, quality: torch.Tensor) -> torch.Tensor:
        return self.net(quality)


class GraphVAEImputer(nn.Module):
    def __init__(
        self,
        n_modalities: int = 6,
        d_model: int = 512,
        latent_dim: int = 64,
        n_graph_layers: int = 2,
        beta: float = 0.5,
    ) -> None:
        super().__init__()
        self.beta = beta
        self.graph_layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(d_model, d_model),
                    nn.GELU(),
                    nn.LayerNorm(d_model),
                )
                for _ in range(n_graph_layers)
            ]
        )
        self.encoder = nn.Sequential(
            nn.Linear(d_model * 2, 256),
            nn.GELU(),
            nn.Linear(256, 128),
            nn.GELU(),
        )
        self.mu_head = nn.Linear(128, latent_dim)
        self.logvar_head = nn.Linear(128, latent_dim)
        self.decoders = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(latent_dim, 256),
                    nn.GELU(),
                    nn.Linear(256, d_model),
                )
                for _ in range(n_modalities)
            ]
        )
        self.uncertainty_heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(latent_dim, 32),
                    nn.GELU(),
                    nn.Linear(32, 1),
                    nn.Sigmoid(),
                )
                for _ in range(n_modalities)
            ]
        )

    @staticmethod
    def _masked_mean(emb: torch.Tensor, mask: torch.Tensor, quality: torch.Tensor) -> torch.Tensor:
        weights = mask.float() * (quality.float() + 1e-3)
        weighted = emb * weights.unsqueeze(-1)
        denom = weights.sum(dim=1, keepdim=True).clamp(min=1.0)
        return weighted.sum(dim=1) / denom

    def _message_passing(self, node_repr: torch.Tensor, adjacency: Optional[torch.Tensor]) -> torch.Tensor:
        if adjacency is None:
            return node_repr
        adjacency = adjacency.float()
        row_sums = adjacency.sum(dim=-1, keepdim=True).clamp(min=1e-6)
        propagated = torch.matmul(adjacency / row_sums, node_repr)
        for layer in self.graph_layers:
            propagated = layer(propagated)
        return propagated

    @staticmethod
    def _reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(
        self,
        emb: torch.Tensor,
        present_mask: torch.Tensor,
        quality: torch.Tensor,
        patient_graph: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        node_repr = self._masked_mean(emb, present_mask, quality)
        neighbor_repr = self._message_passing(node_repr, patient_graph)
        hidden = self.encoder(torch.cat([node_repr, neighbor_repr], dim=-1))
        mu = self.mu_head(hidden)
        logvar = torch.clamp(self.logvar_head(hidden), min=-5.0, max=5.0)
        latent = self._reparameterize(mu, logvar)

        decoded = torch.stack([decoder(latent) for decoder in self.decoders], dim=1)
        uncertainty = torch.stack(
            [head(latent).squeeze(-1) for head in self.uncertainty_heads],
            dim=1,
        )
        missing_mask = (present_mask == 0).unsqueeze(-1)
        imputed = torch.where(missing_mask, decoded, emb)

        observed_mask = present_mask.float().unsqueeze(-1)
        recon_error = ((decoded - emb) ** 2) * observed_mask
        recon_loss = recon_error.sum() / observed_mask.sum().clamp(min=1.0)
        kl_loss = (
            -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / emb.size(0)
        ) * self.beta
        return imputed, uncertainty, kl_loss, recon_loss


class RWKVBlock(nn.Module):
    def __init__(self, d_model: int, d_ffn: Optional[int] = None, dropout: float = 0.1) -> None:
        super().__init__()
        d_ffn = d_ffn or d_model * 4
        self.time_mix_k = nn.Parameter(torch.rand(1, 1, d_model))
        self.time_mix_v = nn.Parameter(torch.rand(1, 1, d_model))
        self.time_mix_r = nn.Parameter(torch.rand(1, 1, d_model))
        self.time_decay = nn.Parameter(torch.zeros(d_model))
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.receptance = nn.Linear(d_model, d_model)
        self.output = nn.Linear(d_model, d_model)
        self.channel = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_ffn),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ffn, d_model),
        )
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    @staticmethod
    def _time_shift(x: torch.Tensor) -> torch.Tensor:
        shifted = torch.zeros_like(x)
        shifted[:, 1:, :] = x[:, :-1, :]
        return shifted

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        prev = self._time_shift(x)
        xk = x * self.time_mix_k + prev * (1.0 - self.time_mix_k)
        xv = x * self.time_mix_v + prev * (1.0 - self.time_mix_v)
        xr = x * self.time_mix_r + prev * (1.0 - self.time_mix_r)

        key = torch.tanh(self.key(xk))
        value = self.value(xv)
        receptance = torch.sigmoid(self.receptance(xr))
        decay = torch.sigmoid(self.time_decay).view(1, 1, -1)

        states = []
        running = torch.zeros(x.size(0), x.size(-1), device=x.device, dtype=x.dtype)
        for step in range(x.size(1)):
            running = decay.squeeze(1) * running + (1.0 - decay.squeeze(1)) * (key[:, step, :] * value[:, step, :])
            states.append(running)
        mixed = torch.stack(states, dim=1)
        x = x + self.dropout(self.output(receptance * mixed))
        x = x + self.channel(self.norm(x))
        return x


class RWKVTemporalEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        d_model: int = 64,
        n_layers: int = 3,
        embed_dim: int = 512,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.blocks = nn.ModuleList([RWKVBlock(d_model, dropout=dropout) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.input_proj(x)
        for block in self.blocks:
            hidden = block(hidden)
        hidden = self.norm(hidden)
        return self.head(hidden.mean(dim=1))


class LSTMTemporalEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        n_layers: int = 2,
        embed_dim: int = 512,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden, _ = self.lstm(x)
        return self.head(hidden.mean(dim=1))


class PositionalMLP(nn.Module):
    def __init__(self, model_dim: int = 512) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2, model_dim),
            nn.GELU(),
            nn.Linear(model_dim, model_dim),
        )

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        return self.mlp(coords)


class PatchProjector(nn.Module):
    def __init__(self, in_dim: int = 1024, model_dim: int = 512) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_dim, model_dim),
            nn.GELU(),
            nn.LayerNorm(model_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class AttentionPooling(nn.Module):
    def __init__(self, d_model: int = 512, hidden_dim: int = 256) -> None:
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        scores = self.score(x).squeeze(-1)
        weights = torch.softmax(scores, dim=1)
        return torch.sum(weights.unsqueeze(-1) * x, dim=1), weights


class HierarchicalABMIL(nn.Module):
    def __init__(self, d_model: int = 512, n_heads: int = 8, n_layers: int = 2, dropout: float = 0.1) -> None:
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.cluster_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.intra_pool = AttentionPooling(d_model=d_model)
        self.slide_pool = AttentionPooling(d_model=d_model)
        self.fusion = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Linear(d_model, 2),
        )

    def encode_bag(self, patch_embeddings: torch.Tensor, cluster_ids: torch.Tensor) -> torch.Tensor:
        unique_ids = torch.unique(cluster_ids, sorted=True)
        cluster_embeddings = []
        for cluster_id in unique_ids:
            cluster_mask = cluster_ids == cluster_id
            cluster_tokens = patch_embeddings[cluster_mask].unsqueeze(0)
            pooled, _ = self.intra_pool(cluster_tokens)
            cluster_embeddings.append(pooled.squeeze(0))
        if not cluster_embeddings:
            return torch.zeros(patch_embeddings.size(-1), device=patch_embeddings.device, dtype=patch_embeddings.dtype)
        cluster_tensor = torch.stack(cluster_embeddings, dim=0).unsqueeze(0)
        refined = self.cluster_encoder(cluster_tensor)
        pooled, _ = self.slide_pool(refined)
        return pooled.squeeze(0)

    def forward(
        self,
        primary_embeddings: Optional[torch.Tensor],
        primary_clusters: Optional[torch.Tensor],
        lymph_embeddings: Optional[torch.Tensor],
        lymph_clusters: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if primary_embeddings is None and lymph_embeddings is None:
            return torch.zeros(1, 512, device=self.fusion[0].weight.device)

        primary_repr = None
        lymph_repr = None
        if primary_embeddings is not None and primary_clusters is not None:
            primary_repr = self.encode_bag(primary_embeddings, primary_clusters)
        if lymph_embeddings is not None and lymph_clusters is not None:
            lymph_repr = self.encode_bag(lymph_embeddings, lymph_clusters)

        if primary_repr is None:
            return lymph_repr.unsqueeze(0)
        if lymph_repr is None:
            return primary_repr.unsqueeze(0)

        weights = torch.softmax(self.fusion(torch.cat([primary_repr, lymph_repr], dim=-1)), dim=-1)
        fused = weights[0] * primary_repr + weights[1] * lymph_repr
        return fused.unsqueeze(0)


class MIRAGENet(nn.Module):
    def __init__(
        self,
        d_model: int = 512,
        n_modalities: int = 6,
        n_latents: int = 64,
        n_perceiver_layers: int = 4,
        n_heads: int = 8,
        dropout: float = 0.2,
        use_graph_vae: bool = True,
        use_kan_heads: bool = True,
    ) -> None:
        super().__init__()
        self.use_graph_vae = use_graph_vae
        self.graph_vae_imputer = GraphVAEImputer(n_modalities=n_modalities, d_model=d_model)
        self.quality_gate = QualityGate(n_mod=n_modalities, hidden=64)
        self.mod_pos_enc = nn.Parameter(torch.randn(n_modalities, d_model) * 0.02)
        self.latents = nn.Parameter(torch.randn(n_latents, d_model) * 0.02)
        self.group_cross_attn = nn.ModuleList(
            [
                nn.MultiheadAttention(d_model, n_heads, batch_first=True, dropout=dropout)
                for _ in range(4)
            ]
        )
        perceiver_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.perceiver_self_attn = nn.TransformerEncoder(perceiver_layer, num_layers=n_perceiver_layers)
        self.route_surv = nn.Linear(d_model, 4)
        self.route_rec = nn.Linear(d_model, 4)
        if use_kan_heads:
            self.head_surv = KANNetwork([d_model, 256, 128, 1], dropout=dropout)
            self.head_rec = KANNetwork([d_model, 256, 128, 1], dropout=dropout)
        else:
            self.head_surv = nn.Sequential(
                nn.Linear(d_model, 256),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, 1),
            )
            self.head_rec = nn.Sequential(
                nn.Linear(d_model, 256),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, 1),
            )
        self.contrast_proj = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
        )
        self.uncertainty_weight = nn.Parameter(torch.tensor(0.1))

    def forward(
        self,
        emb: torch.Tensor,
        quality: torch.Tensor,
        present_mask: torch.Tensor,
        patient_graph: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        if self.use_graph_vae:
            imputed, uncertainty, kl_loss, recon_loss = self.graph_vae_imputer(
                emb, present_mask, quality, patient_graph
            )
            missing = (present_mask == 0).unsqueeze(-1)
            emb = torch.where(missing, imputed, emb)
            quality = torch.where(
                present_mask == 0,
                quality * (1.0 - uncertainty * self.uncertainty_weight),
                quality,
            )
        else:
            kl_loss = torch.tensor(0.0, device=emb.device)
            recon_loss = torch.tensor(0.0, device=emb.device)

        gate = self.quality_gate(quality) * present_mask.float()
        x = emb + self.mod_pos_enc.unsqueeze(0)
        x = x * gate.unsqueeze(-1)

        latents = self.latents.unsqueeze(0).expand(emb.size(0), -1, -1)
        group_outputs = []
        for group_idx, modality_indices in enumerate(([0, 1, 4], [2], [3], [5])):
            group_tokens = x[:, modality_indices, :]
            attended, _ = self.group_cross_attn[group_idx](latents, group_tokens, group_tokens)
            latents = latents + attended
            group_outputs.append(latents.mean(dim=1))

        latents = self.perceiver_self_attn(latents)
        pooled = latents.mean(dim=1)

        group_stack = torch.stack(group_outputs, dim=1)
        weight_surv = torch.softmax(self.route_surv(pooled), dim=-1).unsqueeze(-1)
        weight_rec = torch.softmax(self.route_rec(pooled), dim=-1).unsqueeze(-1)
        rep_surv = torch.sum(weight_surv * group_stack, dim=1)
        rep_rec = torch.sum(weight_rec * group_stack, dim=1)

        return {
            "logit_surv": self.head_surv(rep_surv).squeeze(-1),
            "logit_rec": self.head_rec(rep_rec).squeeze(-1),
            "rep": pooled,
            "cproj": F.normalize(self.contrast_proj(pooled), dim=-1),
            "kl_loss": kl_loss,
            "imputation_loss": recon_loss,
            "group_reps": group_stack,
        }
