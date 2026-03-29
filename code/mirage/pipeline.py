from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Dict

import joblib
import numpy as np
import pandas as pd

from .common import build_knn_graph_matrix, ensure_directory, load_json_file, save_embedding_h5, save_preprocessed_h5, set_seed, write_json_file
from .config import (
    BLOOD_JSON,
    BLOOD_REF_JSON,
    CHECKPOINT_DIR,
    CLINICAL_JSON,
    HISTORIES_DIR,
    ICD_CODES_DIR,
    INFERENCE_RESOURCES_DIR,
    OPS_CODES_DIR,
    OUT_DIR,
    PATHOLOGICAL_JSON,
    REPORTS_DIR,
    SURGERY_DIR,
    WSI_CUP,
    WSI_HYPO,
    WSI_LARYNX,
    WSI_LYMPH,
    WSI_ORAL,
    WSI_OROPH1,
    WSI_OROPH2,
)

from preprocess import Temporal as temporal_stage
from preprocess import clinical as clinical_stage
from preprocess import codes as codes_stage
from preprocess import pathological as pathological_stage
from preprocess import semantic as semantic_stage
from preprocess import spatial as spatial_stage


@dataclass
class PreprocessArtifacts:
    h5_paths: Dict[str, Path]
    preprocess_outdir: Path


class BasePreprocessor:
    def __init__(self, outdir: Path, device: str = "auto", seed: int = 42) -> None:
        self.outdir = ensure_directory(outdir)
        self.device = device
        self.seed = seed


class ClinicalPreprocessor(BasePreprocessor):
    def __init__(self, input_json: Path = CLINICAL_JSON, **kwargs) -> None:
        super().__init__(**kwargs)
        self.input_json = Path(input_json)

    def run(self) -> Dict[str, Path]:
        set_seed(self.seed)
        df = pd.read_json(self.input_json)
        df["patient_id"] = df["patient_id"].apply(lambda value: str(int(value)).zfill(3))
        df = df.drop_duplicates(subset=["patient_id"]).set_index("patient_id").sort_index()

        surv_label, rec_label = clinical_stage.derive_labels(df.reset_index())
        features, base_features, _, frequency_maps, observed = clinical_stage.build_feature_frame(df)
        pipeline = clinical_stage.fit_tabular_pipeline(features)
        graph_adjacency = build_knn_graph_matrix(pipeline["scaled"], k=10)
        quality = (1.0 - observed.isna().mean(axis=1).values).astype(np.float32)

        model, embeddings = clinical_stage.train_embedding_model(
            pipeline["scaled"],
            SimpleNamespace(device=self.device, epochs=50, bs=128, lr=1e-3),
        )
        embeddings = clinical_stage.normalize_embedding_dim(embeddings)

        save_preprocessed_h5(
            self.outdir / "clinical_preprocessed.h5",
            df.index.tolist(),
            features.columns.tolist(),
            pipeline["raw"],
            pipeline["imputed"],
            pipeline["mask"],
            extra={
                "graph_adjacency": graph_adjacency,
                "surv_5yr_label": surv_label,
                "rec_2yr_label": rec_label,
                "quality_score": quality,
                "base_feature_names": base_features.columns.tolist(),
            },
        )
        save_embedding_h5(
            self.outdir / "clinical_embedding_512.h5",
            df.index.tolist(),
            embeddings,
            quality_scores=quality,
            extra={"surv_5yr_label": surv_label, "rec_2yr_label": rec_label},
        )
        joblib.dump(
            {
                "feature_names": features.columns.tolist(),
                "base_feature_names": base_features.columns.tolist(),
                "frequency_maps": frequency_maps,
                "cat_encoders": frequency_maps,
                "imputer": pipeline["imputer"],
                "median_imputer": pipeline["imputer"],
                "scaler": pipeline["scaler"],
                "graph_k": 10,
                "drop_cols": clinical_stage.DROP_COLS,
                "numeric_cols": clinical_stage.NUMERIC_COLS,
                "categorical_cols": clinical_stage.CATEGORICAL_COLS,
            },
            self.outdir / "clinical_preproc_objects.joblib",
        )
        if model is not None:
            import torch

            torch.save(
                {"state_dict": model.state_dict(), "input_dim": pipeline["scaled"].shape[1], "embed_dim": 512},
                self.outdir / "clinical_kan_encoder.pt",
            )
        return {"clinical": self.outdir / "clinical_embedding_512.h5"}


class PathologicalPreprocessor(BasePreprocessor):
    def __init__(self, input_json: Path = PATHOLOGICAL_JSON, **kwargs) -> None:
        super().__init__(**kwargs)
        self.input_json = Path(input_json)

    def run(self) -> Dict[str, Path]:
        set_seed(self.seed)
        df = pd.read_json(self.input_json)
        df["patient_id"] = df["patient_id"].apply(lambda value: str(int(value)).zfill(3))
        df = df.drop_duplicates(subset=["patient_id"]).set_index("patient_id").sort_index()

        features, base_features, _, frequency_maps, observed = pathological_stage.build_feature_frame(df)
        pipeline = pathological_stage.fit_tabular_pipeline(features)
        graph_adjacency = build_knn_graph_matrix(pipeline["scaled"], k=10)
        quality = (1.0 - observed.isna().mean(axis=1).values).astype(np.float32)

        model, embeddings = pathological_stage.train_embedding_model(
            pipeline["scaled"],
            SimpleNamespace(device=self.device, epochs=50, bs=128, lr=1e-3),
        )
        embeddings = pathological_stage.normalize_embedding_dim(embeddings)

        save_preprocessed_h5(
            self.outdir / "pathological_preprocessed.h5",
            df.index.tolist(),
            features.columns.tolist(),
            pipeline["raw"],
            pipeline["imputed"],
            pipeline["mask"],
            extra={
                "graph_adjacency": graph_adjacency,
                "quality_score": quality,
                "base_feature_names": base_features.columns.tolist(),
            },
        )
        save_embedding_h5(
            self.outdir / "pathological_embedding_512.h5",
            df.index.tolist(),
            embeddings,
            quality_scores=quality,
        )
        joblib.dump(
            {
                "feature_names": features.columns.tolist(),
                "base_feature_names": base_features.columns.tolist(),
                "frequency_maps": frequency_maps,
                "cat_encoders": frequency_maps,
                "imputer": pipeline["imputer"],
                "median_imputer": pipeline["imputer"],
                "scaler": pipeline["scaler"],
                "graph_k": 10,
                "binary_columns": pathological_stage.BINARY_COLUMNS,
                "categorical_columns": pathological_stage.CATEGORICAL_COLUMNS,
                "numeric_columns": pathological_stage.NUMERIC_COLUMNS,
            },
            self.outdir / "pathological_preproc_objects.joblib",
        )
        if model is not None:
            import torch

            torch.save(
                {"state_dict": model.state_dict(), "input_dim": pipeline["scaled"].shape[1], "embed_dim": 512},
                self.outdir / "pathological_kan_encoder.pt",
            )
        return {"pathological": self.outdir / "pathological_embedding_512.h5"}


class TemporalPreprocessor(BasePreprocessor):
    def __init__(self, blood_json: Path = BLOOD_JSON, ref_json: Path = BLOOD_REF_JSON, mode: str = "rwkv", **kwargs) -> None:
        super().__init__(**kwargs)
        self.blood_json = Path(blood_json)
        self.ref_json = Path(ref_json)
        self.mode = mode

    def run(self) -> Dict[str, Path]:
        set_seed(self.seed)
        records = load_json_file(self.blood_json, default=[])
        ref_records = load_json_file(self.ref_json, default=[])
        df = temporal_stage.normalize_blood_dataframe(records)
        reference_ranges = temporal_stage.load_reference_ranges(ref_records)
        patient_ids = sorted(df["patient_id"].unique().tolist())
        analyte_counts = df.groupby("analyte_name")["patient_id"].nunique().sort_values(ascending=False)
        analytes = analyte_counts[analyte_counts >= temporal_stage.MIN_ANALYTE_PATIENTS].index.tolist()
        for analyte in reference_ranges.keys():
            if analyte not in analytes:
                analytes.append(analyte)
        analytes = sorted(dict.fromkeys(analytes))

        edges, centers = temporal_stage.make_time_bins()
        matrices = np.full((len(patient_ids), len(analytes), temporal_stage.SEQ_LENGTH), np.nan, dtype=np.float32)
        for patient_index, patient_id in enumerate(patient_ids):
            patient_df = df[df["patient_id"] == patient_id]
            matrices[patient_index] = temporal_stage.patient_to_matrix(patient_df, analytes, edges, temporal_stage.SEQ_LENGTH)

        quality = ((~np.isnan(matrices)).sum(axis=(1, 2)) / float(len(analytes) * temporal_stage.SEQ_LENGTH)).astype(np.float32)
        filled = np.stack([temporal_stage.physiology_fill(matrix, analytes, reference_ranges) for matrix in matrices], axis=0)
        flattened = filled.reshape(len(patient_ids), -1)

        knn_imputer = temporal_stage.KNNImputer(n_neighbors=min(temporal_stage.KNN_NEIGHBORS, max(2, len(patient_ids) - 1)))
        imputed = knn_imputer.fit_transform(flattened)
        scaler = temporal_stage.StandardScaler()
        scaled = scaler.fit_transform(imputed).astype(np.float32)
        sequence_inputs = scaled.reshape(len(patient_ids), len(analytes), temporal_stage.SEQ_LENGTH).transpose(0, 2, 1)

        device = self.device if self.device != "auto" else ("cuda" if temporal_stage.TORCH_AVAILABLE and temporal_stage.torch.cuda.is_available() else "cpu")
        if temporal_stage.TORCH_AVAILABLE:
            model, embeddings = temporal_stage.train_encoder(sequence_inputs, self.mode, device, 50, 64, 1e-3, len(analytes))
            if model is not None:
                import torch

                model_name = "temporal_rwkv_encoder.pt" if self.mode == "rwkv" else "temporal_lstm_encoder.pt"
                torch.save({"state_dict": model.state_dict(), "mode": self.mode, "input_dim": len(analytes)}, self.outdir / model_name)
            else:
                embeddings = temporal_stage.fallback_temporal_embedding(sequence_inputs.reshape(len(patient_ids), -1))
        else:
            embeddings = temporal_stage.fallback_temporal_embedding(sequence_inputs.reshape(len(patient_ids), -1))

        save_embedding_h5(
            self.outdir / "temporal_embedding_512.h5",
            patient_ids,
            embeddings,
            quality_scores=quality,
            extra={"analytes": analytes, "time_centers": centers},
        )
        joblib.dump(
            {
                "analytes": analytes,
                "time_edges": edges,
                "time_centers": centers,
                "seq_len": temporal_stage.SEQ_LENGTH,
                "seq_length": temporal_stage.SEQ_LENGTH,
                "window_days": temporal_stage.TIME_WINDOW_DAYS,
                "mode": self.mode,
                "knn_imputer": knn_imputer,
                "scaler": scaler,
                "reference_ranges": reference_ranges,
            },
            self.outdir / "temporal_preproc_objects.joblib",
        )
        write_json_file(
            self.outdir / "temporal_preproc_summary.json",
            {"n_patients": len(patient_ids), "n_analytes": len(analytes), "seq_len": temporal_stage.SEQ_LENGTH, "mode": self.mode, "embedding_shape": list(embeddings.shape)},
        )
        return {"temporal": self.outdir / "temporal_embedding_512.h5"}


class SemanticPreprocessor(BasePreprocessor):
    def __init__(self, histories: Path = HISTORIES_DIR, reports: Path = REPORTS_DIR, surgery: Path = SURGERY_DIR, use_transformer: bool = False, **kwargs) -> None:
        super().__init__(**kwargs)
        self.histories = Path(histories)
        self.reports = Path(reports)
        self.surgery = Path(surgery)
        self.use_transformer = use_transformer

    def run(self) -> Dict[str, Path]:
        histories = semantic_stage.read_text_folder(self.histories)
        reports = semantic_stage.read_text_folder(self.reports)
        surgery = semantic_stage.read_text_folder(self.surgery)
        patient_ids, documents, quality_scores = semantic_stage.combine_modalities(histories, reports, surgery)
        if self.use_transformer and semantic_stage.HAS_TRANSFORMERS:
            device = self.device if self.device != "auto" else ("cuda" if semantic_stage.torch.cuda.is_available() else "cpu")
            embeddings, preproc_objects = semantic_stage.transformer_embeddings(documents, "emilyalsentzer/Bio_ClinicalBERT", 8, device)
        else:
            embeddings, preproc_objects = semantic_stage.tfidf_embeddings(documents)
            preproc_objects["transformer_model_path"] = "emilyalsentzer/Bio_ClinicalBERT"
        save_embedding_h5(self.outdir / "text_semantic_embedding_512.h5", patient_ids, embeddings, quality_scores=quality_scores, extra={"document": documents})
        joblib.dump(preproc_objects, self.outdir / "text_semantic_preproc_objects.joblib")
        return {"semantic": self.outdir / "text_semantic_embedding_512.h5"}


class CodesPreprocessor(BasePreprocessor):
    def __init__(self, icd_dir: Path = ICD_CODES_DIR, ops_dir: Path = OPS_CODES_DIR, **kwargs) -> None:
        super().__init__(**kwargs)
        self.icd_dir = Path(icd_dir)
        self.ops_dir = Path(ops_dir)

    def run(self) -> Dict[str, Path]:
        set_seed(self.seed)
        bags = codes_stage.build_token_bags(self.icd_dir, self.ops_dir)
        patient_ids, vocabulary, token_to_idx, matrix, max_tokens = codes_stage.build_token_matrix(bags)
        quality = (matrix.sum(axis=1) > 0).astype(np.float32)
        model, embeddings = codes_stage.train_embedding_model(matrix, SimpleNamespace(device=self.device, epochs=25, bs=64, lr=1e-3))
        save_embedding_h5(self.outdir / "codes_embedding_512.h5", patient_ids, embeddings, quality_scores=quality, extra={"token_count": matrix.sum(axis=1)})
        joblib.dump(
            {"vocabulary": vocabulary, "token_to_idx": token_to_idx, "max_tokens": max_tokens, "icd_pattern": codes_stage.ICD_PATTERN, "ops_pattern": codes_stage.OPS_PATTERN, "patient_ids": patient_ids},
            self.outdir / "codes_preproc_objects.joblib",
        )
        if model is not None:
            import torch

            torch.save({"state_dict": model.state_dict(), "input_dim": matrix.shape[1], "embed_dim": 512}, self.outdir / "codes_hierarchical_embedder.pt")
        return {"codes": self.outdir / "codes_embedding_512.h5"}


class SpatialPreprocessor(BasePreprocessor):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def run(self) -> Dict[str, Path]:
        if spatial_stage.torch is None:
            raise RuntimeError("spatial preprocessing requires PyTorch")
        files = spatial_stage.collect_wsi_h5_files([WSI_LYMPH, WSI_CUP, WSI_HYPO, WSI_LARYNX, WSI_ORAL, WSI_OROPH1, WSI_OROPH2])
        grouped = spatial_stage.defaultdict(lambda: {"primary": [], "lymph": []})
        for path in files:
            bucket = "lymph" if "LymphNode" in str(path) else "primary"
            grouped[spatial_stage.extract_pid_from_filename(path)][bucket].append(path)

        device = self.device if self.device != "auto" else ("cuda" if spatial_stage.torch.cuda.is_available() else "cpu")
        projector = spatial_stage.PatchProjector(in_dim=1024, model_dim=512).to(device)
        pos_mlp = spatial_stage.PositionalMLP(model_dim=512).to(device)
        habmil = spatial_stage.HierarchicalABMIL(d_model=512).to(device)
        fusion = spatial_stage.RegionFusion().to(device)

        patient_ids, embeddings, embedding_var, qualities, n_patches, tumor_site = [], [], [], [], [], []
        for patient_id, regions in sorted(grouped.items()):
            patient_ids.append(patient_id)
            samples = []
            primary_quality = 0.0
            lymph_quality = 0.0
            total_patches = 0
            for _ in range(8):
                primary_emb, primary_quality, primary_patches = spatial_stage.encode_region(regions["primary"], projector, pos_mlp, habmil, device)
                lymph_emb, lymph_quality, lymph_patches = spatial_stage.encode_region(regions["lymph"], projector, pos_mlp, habmil, device)
                total_patches = max(total_patches, primary_patches + lymph_patches)
                if primary_emb is None and lymph_emb is None:
                    combined = spatial_stage.torch.zeros(512, device=device)
                elif primary_emb is None:
                    combined = lymph_emb.to(device)
                elif lymph_emb is None:
                    combined = primary_emb.to(device)
                else:
                    combined = fusion(spatial_stage.torch.cat([primary_emb.to(device), lymph_emb.to(device)], dim=-1).unsqueeze(0)).squeeze(0)
                samples.append(combined.detach().cpu().numpy())
            sample_stack = np.stack(samples, axis=0)
            embeddings.append(sample_stack.mean(axis=0))
            embedding_var.append(sample_stack.var(axis=0))
            qualities.append(float(max(primary_quality, lymph_quality)))
            n_patches.append(total_patches)
            tumor_site.append("primary_and_lymph" if regions["primary"] and regions["lymph"] else "single_site")

        save_embedding_h5(
            self.outdir / "spatial_embedding_512.h5",
            patient_ids,
            np.asarray(embeddings, dtype=np.float32),
            quality_scores=np.asarray(qualities, dtype=np.float32),
            extra={"embedding_var_512": np.asarray(embedding_var, dtype=np.float32), "n_patches": np.asarray(n_patches, dtype=np.int32), "tumor_site": tumor_site},
        )
        spatial_stage.torch.save(projector.state_dict(), self.outdir / "patch_projector.pt")
        spatial_stage.torch.save(pos_mlp.state_dict(), self.outdir / "positional_mlp.pt")
        spatial_stage.torch.save(habmil.state_dict(), self.outdir / "habmil_model.pt")
        spatial_stage.torch.save(fusion.state_dict(), self.outdir / "spatial_fusion.pt")
        joblib.dump({"max_patches": spatial_stage.MAX_PATCHES}, self.outdir / "spatial_preproc_objects.joblib")
        return {"spatial": self.outdir / "spatial_embedding_512.h5"}


class MiragePreprocessingPipeline:
    def __init__(self, preprocess_outdir: Path = OUT_DIR, device: str = "auto", seed: int = 42, semantic_use_transformer: bool = False) -> None:
        self.preprocess_outdir = ensure_directory(preprocess_outdir)
        self.device = device
        self.seed = seed
        self.semantic_use_transformer = semantic_use_transformer

    def run(self, force: bool = False) -> PreprocessArtifacts:
        expected = {
            "clinical": self.preprocess_outdir / "clinical_embedding_512.h5",
            "pathological": self.preprocess_outdir / "pathological_embedding_512.h5",
            "temporal": self.preprocess_outdir / "temporal_embedding_512.h5",
            "semantic": self.preprocess_outdir / "text_semantic_embedding_512.h5",
            "codes": self.preprocess_outdir / "codes_embedding_512.h5",
            "spatial": self.preprocess_outdir / "spatial_embedding_512.h5",
        }
        if not force and all(path.exists() for path in expected.values()):
            return PreprocessArtifacts(h5_paths=expected, preprocess_outdir=self.preprocess_outdir)

        ClinicalPreprocessor(outdir=self.preprocess_outdir, device=self.device, seed=self.seed).run()
        PathologicalPreprocessor(outdir=self.preprocess_outdir, device=self.device, seed=self.seed).run()
        TemporalPreprocessor(outdir=self.preprocess_outdir, device=self.device, seed=self.seed).run()
        SemanticPreprocessor(outdir=self.preprocess_outdir, device=self.device, seed=self.seed, use_transformer=self.semantic_use_transformer).run()
        CodesPreprocessor(outdir=self.preprocess_outdir, device=self.device, seed=self.seed).run()
        SpatialPreprocessor(outdir=self.preprocess_outdir, device=self.device, seed=self.seed).run()
        return PreprocessArtifacts(h5_paths=expected, preprocess_outdir=self.preprocess_outdir)


def export_inference_resources(
    preprocess_outdir: Path = OUT_DIR,
    checkpoint_outdir: Path = CHECKPOINT_DIR,
    resources_dir: Path = INFERENCE_RESOURCES_DIR,
) -> Path:
    preprocess_outdir = Path(preprocess_outdir)
    checkpoint_outdir = Path(checkpoint_outdir)
    resources_dir = ensure_directory(resources_dir)
    required_files = [
        "clinical_preproc_objects.joblib",
        "clinical_kan_encoder.pt",
        "pathological_preproc_objects.joblib",
        "pathological_kan_encoder.pt",
        "temporal_preproc_objects.joblib",
        "temporal_rwkv_encoder.pt",
        "temporal_lstm_encoder.pt",
        "text_semantic_preproc_objects.joblib",
        "codes_preproc_objects.joblib",
        "codes_hierarchical_embedder.pt",
        "patch_projector.pt",
        "positional_mlp.pt",
        "habmil_model.pt",
        "spatial_fusion.pt",
    ]
    for name in required_files:
        source = preprocess_outdir / name
        if source.exists():
            shutil.copy2(source, resources_dir / name)
    if (checkpoint_outdir / "mirage_net_best_avg.pt").exists():
        shutil.copy2(checkpoint_outdir / "mirage_net_best_avg.pt", resources_dir / "mirage_net_best_avg.pt")
    for fold_path in checkpoint_outdir.glob("fold_*_best_model.pt"):
        shutil.copy2(fold_path, resources_dir / fold_path.name.replace("fold_", "mirage_net_fold"))
    return resources_dir
