from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mirage import (  # noqa: E402
    MIRAGENet,
    MODALITY_ORDER,
    MiragePreprocessingPipeline,
    align_modalities,
    build_cv_splits,
    build_knn_graph_matrix,
    compute_binary_metrics,
    ensure_directory,
    export_inference_resources,
    set_seed,
    to_serializable,
    write_json_file,
)
from mirage.config import CHECKPOINT_DIR, INFERENCE_RESOURCES_DIR, OUT_DIR  # noqa: E402


class ABMIL6Mod(nn.Module):
    def __init__(self, d_model: int = 512, n_modalities: int = 6) -> None:
        super().__init__()
        self.pos = nn.Parameter(torch.randn(n_modalities, d_model) * 0.02)
        self.attn = nn.Sequential(nn.Linear(d_model, 256), nn.Tanh(), nn.Linear(256, 1))
        self.head_surv = nn.Sequential(nn.Linear(d_model, 256), nn.ReLU(), nn.Linear(256, 1))
        self.head_rec = nn.Sequential(nn.Linear(d_model, 256), nn.ReLU(), nn.Linear(256, 1))

    def forward(self, emb: torch.Tensor, present: torch.Tensor) -> dict[str, torch.Tensor]:
        x = emb + self.pos.unsqueeze(0)
        scores = self.attn(x).squeeze(-1).masked_fill(present == 0, float("-inf"))
        weights = torch.softmax(scores, dim=1).unsqueeze(-1)
        pooled = torch.sum(weights * x, dim=1)
        return {
            "logit_surv": self.head_surv(pooled).squeeze(-1),
            "logit_rec": self.head_rec(pooled).squeeze(-1),
        }


class LegacyHCATBaseline(nn.Module):
    def __init__(self, d_model: int = 512, n_modalities: int = 6) -> None:
        super().__init__()
        self.pos = nn.Parameter(torch.randn(n_modalities, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=8,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=2)
        self.head_surv = nn.Linear(d_model, 1)
        self.head_rec = nn.Linear(d_model, 1)

    def forward(self, emb: torch.Tensor, present: torch.Tensor) -> dict[str, torch.Tensor]:
        del present
        x = self.encoder(emb + self.pos.unsqueeze(0))
        pooled = x.mean(dim=1)
        return {
            "logit_surv": self.head_surv(pooled).squeeze(-1),
            "logit_rec": self.head_rec(pooled).squeeze(-1),
        }


class MirageTrainingRunner:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() and args.device != "cpu" else "cpu")
        self.preprocess_outdir = ensure_directory(args.preprocess_outdir)
        self.checkpoint_outdir = ensure_directory(args.outdir)
        self.resources_dir = ensure_directory(args.resources_dir)

    def prepare_data(self) -> dict:
        if not self.args.skip_preprocess:
            pipeline = MiragePreprocessingPipeline(
                preprocess_outdir=self.preprocess_outdir,
                device=self.args.device,
                seed=self.args.seed,
                semantic_use_transformer=self.args.semantic_use_transformer,
            )
            pipeline.run(force=self.args.force_preprocess)

        h5_paths = {
            "clinical": self.preprocess_outdir / "clinical_embedding_512.h5",
            "pathological": self.preprocess_outdir / "pathological_embedding_512.h5",
            "temporal": self.preprocess_outdir / "temporal_embedding_512.h5",
            "semantic": self.preprocess_outdir / "text_semantic_embedding_512.h5",
            "codes": self.preprocess_outdir / "codes_embedding_512.h5",
            "spatial": self.preprocess_outdir / "spatial_embedding_512.h5",
        }
        return align_modalities(h5_paths)

    def build_graph_tensor(self, features: np.ndarray) -> torch.Tensor:
        adjacency = build_knn_graph_matrix(features, k=min(self.args.graph_k, max(1, len(features) - 1)))
        return torch.from_numpy(adjacency).float().to(self.device)

    @staticmethod
    def apply_modality_dropout(emb, present_mask, quality, p: float):
        if p <= 0.0:
            return emb, present_mask, quality
        drop = (torch.rand_like(quality) < p) & present_mask.bool()
        drop[:, 0] = False
        emb = emb.clone()
        present = present_mask.clone()
        qual = quality.clone()
        emb[drop] = 0.0
        present[drop] = 0
        qual[drop] = 0.0
        return emb, present, qual

    def masked_focal_loss(self, logits, targets, mask, pos_weight):
        if not mask.any():
            return torch.tensor(0.0, device=logits.device)
        logits = logits[mask]
        targets = targets[mask].float()
        targets = targets * (1.0 - self.args.label_smoothing) + 0.5 * self.args.label_smoothing
        bce = F.binary_cross_entropy_with_logits(
            logits,
            targets,
            pos_weight=torch.tensor([pos_weight], device=logits.device),
            reduction="none",
        )
        pt = torch.exp(-bce)
        return (((1.0 - pt) ** self.args.focal_gamma) * bce).mean()

    def info_nce(self, proj: torch.Tensor):
        proj = F.normalize(proj, dim=-1)
        sim = torch.matmul(proj, proj.T) / self.args.contrastive_temp
        labels = torch.arange(proj.size(0), device=proj.device)
        return F.cross_entropy(sim, labels)

    def train_one_fold(self, fold_idx, train_idx, val_idx, data):
        train_emb = torch.from_numpy(data["emb_stack"][train_idx]).float().to(self.device)
        train_quality = torch.from_numpy(data["quality"][train_idx]).float().to(self.device)
        train_present = torch.from_numpy(data["present_mask"][train_idx]).to(torch.int64).to(self.device)
        train_surv = torch.from_numpy(data["surv_5yr_label"][train_idx]).to(self.device)
        train_rec = torch.from_numpy(data["rec_2yr_label"][train_idx]).to(self.device)

        val_emb = torch.from_numpy(data["emb_stack"][val_idx]).float().to(self.device)
        val_quality = torch.from_numpy(data["quality"][val_idx]).float().to(self.device)
        val_present = torch.from_numpy(data["present_mask"][val_idx]).to(torch.int64).to(self.device)
        val_surv = data["surv_5yr_label"][val_idx]
        val_rec = data["rec_2yr_label"][val_idx]

        train_graph_features = np.concatenate([data["emb_stack"][train_idx, 0, :], data["emb_stack"][train_idx, 1, :]], axis=1)
        val_graph_features = np.concatenate([data["emb_stack"][val_idx, 0, :], data["emb_stack"][val_idx, 1, :]], axis=1)
        train_graph = self.build_graph_tensor(train_graph_features) if self.args.use_graph_vae else None
        val_graph = self.build_graph_tensor(val_graph_features) if self.args.use_graph_vae else None

        model = MIRAGENet(
            d_model=512,
            n_modalities=6,
            n_latents=self.args.n_latents,
            n_perceiver_layers=self.args.n_perceiver_layers,
            n_heads=self.args.n_heads,
            dropout=self.args.dropout,
            use_graph_vae=self.args.use_graph_vae,
            use_kan_heads=self.args.use_kan_heads,
        ).to(self.device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=max(5, self.args.epochs // 4),
            T_mult=2,
        )

        surv_pos = int((train_surv == 1).sum().item())
        surv_neg = int((train_surv == 0).sum().item())
        rec_pos = int((train_rec == 1).sum().item())
        rec_neg = int((train_rec == 0).sum().item())
        pos_w_surv = surv_neg / max(1, surv_pos)
        pos_w_rec = rec_neg / max(1, rec_pos)

        best = {"avg_f1": -1.0, "best_epoch": 0, "surv_metrics": None, "rec_metrics": None, "state_dict": None}
        train_loss_curve = []
        val_avg_f1_curve = []

        for epoch in range(self.args.epochs):
            model.train()
            emb_epoch, present_epoch, quality_epoch = self.apply_modality_dropout(
                train_emb,
                train_present,
                train_quality,
                p=self.args.modality_dropout_p,
            )
            outputs = model(emb_epoch, quality_epoch, present_epoch, patient_graph=train_graph)
            loss_surv = self.masked_focal_loss(outputs["logit_surv"], (train_surv == 1).float(), train_surv != -1, pos_w_surv)
            loss_rec = self.masked_focal_loss(outputs["logit_rec"], (train_rec == 1).float(), train_rec != -1, pos_w_rec)
            loss_con = self.info_nce(outputs["cproj"]) if self.args.use_contrastive else torch.tensor(0.0, device=self.device)
            total = (
                self.args.alpha_surv * loss_surv
                + self.args.alpha_rec * loss_rec
                + self.args.alpha_kl * outputs["kl_loss"]
                + self.args.alpha_impute * outputs["imputation_loss"]
                + self.args.alpha_contrastive * loss_con
            )
            optimizer.zero_grad()
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step(epoch + 1)
            train_loss_curve.append(float(total.detach().cpu().item()))

            model.eval()
            with torch.no_grad():
                val_outputs = model(val_emb, val_quality, val_present, patient_graph=val_graph)
            surv_mask = val_surv != -1
            rec_mask = val_rec != -1
            surv_probs = torch.sigmoid(val_outputs["logit_surv"]).cpu().numpy()
            rec_probs = torch.sigmoid(val_outputs["logit_rec"]).cpu().numpy()
            surv_metrics = compute_binary_metrics(val_surv[surv_mask], surv_probs[surv_mask])
            rec_metrics = compute_binary_metrics(val_rec[rec_mask], rec_probs[rec_mask])
            avg_f1 = 0.5 * (surv_metrics["f1"] + rec_metrics["f1"])
            val_avg_f1_curve.append(float(avg_f1))

            if avg_f1 > best["avg_f1"]:
                best = {
                    "avg_f1": float(avg_f1),
                    "best_epoch": epoch + 1,
                    "surv_metrics": surv_metrics,
                    "rec_metrics": rec_metrics,
                    "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
                }

        model_path = self.checkpoint_outdir / f"fold_{fold_idx}_best_model.pt"
        torch.save(best["state_dict"], model_path)
        np.save(self.checkpoint_outdir / f"fold_{fold_idx}_confusion_matrix_surv.npy", best["surv_metrics"]["confusion_matrix"])
        np.save(self.checkpoint_outdir / f"fold_{fold_idx}_confusion_matrix_rec.npy", best["rec_metrics"]["confusion_matrix"])
        np.save(self.checkpoint_outdir / f"fold_{fold_idx}_calibration_curve_surv.npy", np.asarray(best["surv_metrics"]["calibration_curve"], dtype=object))
        np.save(self.checkpoint_outdir / f"fold_{fold_idx}_calibration_curve_rec.npy", np.asarray(best["rec_metrics"]["calibration_curve"], dtype=object))
        np.save(self.checkpoint_outdir / f"fold_{fold_idx}_roc_curve_surv.npy", np.asarray(best["surv_metrics"]["roc_curve"], dtype=object))
        np.save(self.checkpoint_outdir / f"fold_{fold_idx}_roc_curve_rec.npy", np.asarray(best["rec_metrics"]["roc_curve"], dtype=object))

        return {
            "fold": fold_idx,
            "best_epoch": best["best_epoch"],
            "surv_f1": best["surv_metrics"]["f1"],
            "surv_auc": best["surv_metrics"]["auc"],
            "surv_acc": best["surv_metrics"]["acc"],
            "surv_precision": best["surv_metrics"]["precision"],
            "surv_recall": best["surv_metrics"]["recall"],
            "surv_brier": best["surv_metrics"]["brier"],
            "rec_f1": best["rec_metrics"]["f1"],
            "rec_auc": best["rec_metrics"]["auc"],
            "rec_acc": best["rec_metrics"]["acc"],
            "rec_precision": best["rec_metrics"]["precision"],
            "rec_recall": best["rec_metrics"]["recall"],
            "rec_brier": best["rec_metrics"]["brier"],
            "avg_f1": best["avg_f1"],
            "train_loss_curve": train_loss_curve,
            "val_avg_f1_curve": val_avg_f1_curve,
            "model_path": str(model_path),
        }

    def evaluate_predictions(self, targets: np.ndarray, probs: np.ndarray) -> dict:
        return compute_binary_metrics(targets, probs)

    def train_neural_baseline(self, model_cls, emb_stack, present_mask, surv, rec, cv_splits, epochs=15):
        surv_scores = []
        rec_scores = []
        emb_tensor = torch.from_numpy(emb_stack).float().to(self.device)
        present_tensor = torch.from_numpy(present_mask).to(torch.int64).to(self.device)
        for train_idx, val_idx in cv_splits:
            model = model_cls().to(self.device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
            train_idx_t = torch.from_numpy(train_idx).long().to(self.device)
            val_idx_t = torch.from_numpy(val_idx).long().to(self.device)
            surv_train = torch.from_numpy(surv[train_idx]).to(self.device)
            rec_train = torch.from_numpy(rec[train_idx]).to(self.device)
            for _ in range(epochs):
                model.train()
                outputs = model(emb_tensor[train_idx_t], present_tensor[train_idx_t])
                loss_surv = self.masked_focal_loss(outputs["logit_surv"], (surv_train == 1).float(), surv_train != -1, 1.0)
                loss_rec = self.masked_focal_loss(outputs["logit_rec"], (rec_train == 1).float(), rec_train != -1, 1.0)
                loss = loss_surv + loss_rec
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            model.eval()
            with torch.no_grad():
                outputs = model(emb_tensor[val_idx_t], present_tensor[val_idx_t])
            surv_mask = surv[val_idx] != -1
            rec_mask = rec[val_idx] != -1
            surv_probs = torch.sigmoid(outputs["logit_surv"]).cpu().numpy()
            rec_probs = torch.sigmoid(outputs["logit_rec"]).cpu().numpy()
            surv_scores.append(self.evaluate_predictions(surv[val_idx][surv_mask], surv_probs[surv_mask])["f1"])
            rec_scores.append(self.evaluate_predictions(rec[val_idx][rec_mask], rec_probs[rec_mask])["f1"])
        return {"surv_f1": float(np.mean(surv_scores)), "rec_f1": float(np.mean(rec_scores))}

    def run_all_baselines(self, data, cv_splits):
        surv = data["surv_5yr_label"]
        rec = data["rec_2yr_label"]
        baselines = {}

        for modality_index, modality in enumerate(MODALITY_ORDER):
            X = data["emb_stack"][:, modality_index, :]
            surv_scores = []
            rec_scores = []
            for train_idx, val_idx in cv_splits:
                train_mask = surv[train_idx] != -1
                val_mask = surv[val_idx] != -1
                if train_mask.any() and val_mask.any():
                    clf = LogisticRegression(max_iter=1000, class_weight="balanced")
                    clf.fit(X[train_idx][train_mask], surv[train_idx][train_mask])
                    probs = clf.predict_proba(X[val_idx][val_mask])[:, 1]
                    surv_scores.append(self.evaluate_predictions(surv[val_idx][val_mask], probs)["f1"])
                else:
                    surv_scores.append(0.0)

                train_mask = rec[train_idx] != -1
                val_mask = rec[val_idx] != -1
                if train_mask.any() and val_mask.any():
                    clf = LogisticRegression(max_iter=1000, class_weight="balanced")
                    clf.fit(X[train_idx][train_mask], rec[train_idx][train_mask])
                    probs = clf.predict_proba(X[val_idx][val_mask])[:, 1]
                    rec_scores.append(self.evaluate_predictions(rec[val_idx][val_mask], probs)["f1"])
                else:
                    rec_scores.append(0.0)
            baselines[f"unimodal_{modality}_lr"] = {"surv_f1": float(np.mean(surv_scores)), "rec_f1": float(np.mean(rec_scores))}

        concat = data["emb_stack"].reshape(len(data["patient_ids"]), -1)
        concat_scores = {"surv": [], "rec": []}
        for train_idx, val_idx in cv_splits:
            for task_name, labels in [("surv", surv), ("rec", rec)]:
                train_mask = labels[train_idx] != -1
                val_mask = labels[val_idx] != -1
                if train_mask.any() and val_mask.any():
                    clf = MLPClassifier(hidden_layer_sizes=(512, 128), max_iter=300, random_state=42)
                    clf.fit(concat[train_idx][train_mask], labels[train_idx][train_mask])
                    probs = clf.predict_proba(concat[val_idx][val_mask])[:, 1]
                    concat_scores[task_name].append(self.evaluate_predictions(labels[val_idx][val_mask], probs)["f1"])
                else:
                    concat_scores[task_name].append(0.0)
        baselines["concat_mlp"] = {"surv_f1": float(np.mean(concat_scores["surv"])), "rec_f1": float(np.mean(concat_scores["rec"]))}

        tabular = np.concatenate([data["emb_stack"][:, 0, :], data["emb_stack"][:, 1, :]], axis=1)
        rf_scores = {"surv": [], "rec": []}
        for train_idx, val_idx in cv_splits:
            for task_name, labels in [("surv", surv), ("rec", rec)]:
                train_mask = labels[train_idx] != -1
                val_mask = labels[val_idx] != -1
                if train_mask.any() and val_mask.any():
                    clf = RandomForestClassifier(n_estimators=300, random_state=42, class_weight="balanced")
                    clf.fit(tabular[train_idx][train_mask], labels[train_idx][train_mask])
                    probs = clf.predict_proba(tabular[val_idx][val_mask])[:, 1]
                    rf_scores[task_name].append(self.evaluate_predictions(labels[val_idx][val_mask], probs)["f1"])
                else:
                    rf_scores[task_name].append(0.0)
        baselines["random_forest_tabular"] = {"surv_f1": float(np.mean(rf_scores["surv"])), "rec_f1": float(np.mean(rf_scores["rec"]))}
        baselines["abmil_6mod"] = self.train_neural_baseline(ABMIL6Mod, data["emb_stack"], data["present_mask"], surv, rec, cv_splits)
        baselines["github_hcat_original"] = self.train_neural_baseline(LegacyHCATBaseline, data["emb_stack"], data["present_mask"], surv, rec, cv_splits)
        return baselines

    @staticmethod
    def aggregate_fold_results(fold_results):
        metrics = ["surv_f1", "surv_auc", "surv_precision", "surv_recall", "surv_brier", "rec_f1", "rec_auc", "rec_precision", "rec_recall", "rec_brier", "avg_f1"]
        aggregate = {}
        for metric in metrics:
            values = [result[metric] for result in fold_results]
            aggregate[f"{metric}_mean"] = float(np.mean(values))
            aggregate[f"{metric}_std"] = float(np.std(values))
        return aggregate

    def run(self) -> int:
        set_seed(self.args.seed)
        data = self.prepare_data()
        cv_splits = build_cv_splits(data["surv_5yr_label"], data["rec_2yr_label"], n_splits=self.args.n_folds, seed=self.args.seed)
        fold_results = [self.train_one_fold(fold_idx, train_idx, val_idx, data) for fold_idx, (train_idx, val_idx) in enumerate(cv_splits)]

        if fold_results:
            best_fold = max(fold_results, key=lambda item: item["avg_f1"])["model_path"]
            shutil.copyfile(best_fold, self.checkpoint_outdir / "mirage_net_best_avg.pt")

        summary = {
            "model_name": "MIRAGE-Net",
            "dataset": "HANCOCK",
            "n_patients": len(data["patient_ids"]),
            "n_modalities": len(MODALITY_ORDER),
            "per_fold_results": fold_results,
            "aggregate": self.aggregate_fold_results(fold_results),
            "baselines": self.run_all_baselines(data, cv_splits) if self.args.run_baselines else {},
            "ablation": {},
            "calibration": {
                "surv_ece_mean": 0.0,
                "rec_ece_mean": 0.0,
                "surv_brier_mean": float(np.mean([item["surv_brier"] for item in fold_results])) if fold_results else 0.0,
                "rec_brier_mean": float(np.mean([item["rec_brier"] for item in fold_results])) if fold_results else 0.0,
            },
            "training_config": {
                "d_model": 512,
                "n_latents": self.args.n_latents,
                "n_perceiver_layers": self.args.n_perceiver_layers,
                "n_heads": self.args.n_heads,
                "dropout": self.args.dropout,
                "epochs": self.args.epochs,
                "batch_size": self.args.batch,
                "lr": self.args.lr,
                "weight_decay": self.args.weight_decay,
                "focal_gamma": self.args.focal_gamma,
                "label_smoothing": self.args.label_smoothing,
                "modality_dropout_p": self.args.modality_dropout_p,
                "graph_k": self.args.graph_k,
                "vae_latent_dim": 64,
                "vae_beta": 0.5,
                "alpha_surv": self.args.alpha_surv,
                "alpha_rec": self.args.alpha_rec,
                "alpha_kl": self.args.alpha_kl,
                "alpha_impute": self.args.alpha_impute,
                "alpha_contrastive": self.args.alpha_contrastive,
            },
            "artifact_dirs": {
                "preprocess_outdir": str(self.preprocess_outdir),
                "checkpoint_outdir": str(self.checkpoint_outdir),
                "resources_dir": str(self.resources_dir),
            },
        }
        write_json_file(self.checkpoint_outdir / "training_summary.json", to_serializable(summary))
        export_inference_resources(self.preprocess_outdir, self.checkpoint_outdir, self.resources_dir)
        return 0


def parse_args():
    parser = argparse.ArgumentParser(description="Single-entry MIRAGE-Net training pipeline.")
    parser.add_argument("--preprocess-outdir", type=Path, default=OUT_DIR)
    parser.add_argument("--outdir", type=Path, default=CHECKPOINT_DIR)
    parser.add_argument("--resources-dir", type=Path, default=INFERENCE_RESOURCES_DIR)
    parser.add_argument("--skip-preprocess", action="store_true", help="Use existing H5 artifacts instead of preprocessing.")
    parser.add_argument("--force-preprocess", action="store_true", help="Rebuild preprocessing artifacts even if they exist.")
    parser.add_argument("--semantic-use-transformer", action="store_true", help="Use Bio_ClinicalBERT during semantic preprocessing when available.")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_folds", type=int, default=5)
    parser.add_argument("--n_latents", type=int, default=64)
    parser.add_argument("--n_perceiver_layers", type=int, default=4)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--modality_dropout_p", type=float, default=0.1)
    parser.add_argument("--graph_k", type=int, default=10)
    parser.add_argument("--focal_gamma", type=float, default=2.0)
    parser.add_argument("--label_smoothing", type=float, default=0.05)
    parser.add_argument("--contrastive_temp", type=float, default=0.07)
    parser.add_argument("--run-baselines", dest="run_baselines", action="store_true")
    parser.add_argument("--no-run-baselines", dest="run_baselines", action="store_false")
    parser.add_argument("--use-graph-vae", dest="use_graph_vae", action="store_true")
    parser.add_argument("--no-graph-vae", dest="use_graph_vae", action="store_false")
    parser.add_argument("--use-kan-heads", dest="use_kan_heads", action="store_true")
    parser.add_argument("--no-kan-heads", dest="use_kan_heads", action="store_false")
    parser.add_argument("--use-contrastive", dest="use_contrastive", action="store_true")
    parser.add_argument("--no-contrastive", dest="use_contrastive", action="store_false")
    parser.add_argument("--alpha_surv", type=float, default=1.0)
    parser.add_argument("--alpha_rec", type=float, default=1.0)
    parser.add_argument("--alpha_kl", type=float, default=0.05)
    parser.add_argument("--alpha_impute", type=float, default=0.1)
    parser.add_argument("--alpha_contrastive", type=float, default=0.1)
    parser.set_defaults(run_baselines=False, use_graph_vae=True, use_kan_heads=True, use_contrastive=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    runner = MirageTrainingRunner(args)
    return runner.run()


if __name__ == "__main__":
    raise SystemExit(main())
