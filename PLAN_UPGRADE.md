# MIRAGE-Net — Research-Grade Multimodal Cancer Prediction Architecture

> **Proposed Architecture Name:** MIRAGE-Net  
> **Full form:** **M**ultimodal **I**ntegrative **R**esearch **A**rchitecture with **G**raph-**E**nhanced Networks  
> **Replaces:** HCAT-FusionNet (GitHub version)

> **For LLM Readers:** This document is a complete, self-contained plan for building
> MIRAGE-Net from scratch on the HANCOCK dataset.  
> The architecture is fundamentally different from the GitHub HCAT code — not an incremental upgrade.  
> Read EVERY section before touching any code.  
> PATH PLACEHOLDERS are collected in Section 7. Fill them first.

---

## 0. Project Goal (Read This First)

**Task:** Predict two binary outcomes for head-and-neck cancer patients:
- `surv_5yr_label` — 5-year survival (0 = died from tumor, 1 = alive at 5 years, -1 = unknown/censored)
- `rec_2yr_label` — 2-year recurrence (0 = no recurrence within 2 years, 1 = recurrence, -1 = unknown)

**Dataset (HANCOCK):** 763 patients with up to 9 modalities each.

**Evaluation metric:** F1-score for both tasks (primary = average F1).  
**Split:** 5-fold cross-validation (single split was rejected by reviewers).

---

## 1. What the EDA Revealed — Critical Facts

### 1.1 Patients and Labels
| Label | Class 0 | Class 1 | Unknown (-1) | Usable |
|---|---|---|---|---|
| 5yr Survival | 165 (died) | 242 (alive) | 356 | 407 patients |
| 2yr Recurrence | 412 (no) | 133 (yes) | 218 | 545 patients |

- **Class imbalance** for recurrence: pos_weight ≈ 3.1x → must use weighted BCE or focal loss.
- **Censored labels** (-1) must be masked out of the loss — they are NOT negatives.

### 1.2 Modality Coverage
| Modality | Patients | Coverage |
|---|---|---|
| Clinical tabular | 763 | 100% |
| Pathological tabular | 763 | 100% |
| Blood (temporal) | 693 | 90.8% |
| WSI any | 747 | 97.9% |
| reports_english | 742 | 97.2% |
| surgery_descriptions_english | 763 | 100% |
| histories_english | 529 | 69.3% |
| icd_codes | 712 | 93.3% |
| ops_codes | 734 | 96.2% |

### 1.3 Bugs in the Current GitHub Code
1. **ICD/OPS codes are completely ignored** — they are free structured signal encoding tumor type and procedure.
2. **QualityGate hidden dim mismatch:** `train.py` uses `hidden=128`, `hcat_model.py` uses `hidden=64` — causes silent weight skip with `strict=False`.
3. **Multi-h5 patients:** Some patients have `_036` and `_036_a` WSI files — current spatial pipeline handles them incorrectly (duplicate PIDs).
4. **Single train/val split** — rejected by reviewers. Needs 5-fold CV.
5. **No baselines** — paper requires unimodal LR, concat-MLP, ABMIL, MCAT comparisons.
6. **Blood data path bug** (from EDA error): code looks for `StructuredData/clinical_data.json` but correct path is `StructuredData/StructuredData/clinical_data.json`.
7. **German text used** — `histories/` and `reports/` folders contain German. Only use `*_english` folders.

### 1.4 WSI Spatial Data Notes
- LymphNode: 369 files for 369 patients.
- PrimaryTumor is split across 6 sub-folders by tumor site. Must scan all.
- Patients with 2+ h5 files must be **aggregated by mean-pooling patch features before the spatial encoder**, not concatenated.
- Each `.h5` file contains `features` (n_patches × 1024) and `coords` (n_patches × 2).

### 1.5 Blood Temporal Notes
- 38 analytes, 16 analytes present in >80% of patients.
- Time window: 0–14 days before treatment.
- Most measurements at day 0 (31.8%).
- Use a 16-bin temporal grid covering 0–14 days.

---

## 2. Proposed New Architecture: MIRAGE-Net

### 2.1 What Is Wrong with the GitHub Architecture (HCAT)

The current GitHub HCAT has these fundamental design flaws:
1. **Stacked transformer on 5 modality tokens** — applying multi-head attention to only 5 tokens is wasteful; the model cannot learn meaningful cross-modal interactions at that small scale.
2. **No principled fusion** — modality tokens are simply concatenated and passed through a generic TransformerEncoder. There is no mechanism to route information between specific pairs of modalities (e.g., blood markers ↔ pathology, spatial WSI ↔ clinical stage).
3. **MLP encoders for tabular data** — standard MLPs have limited expressivity for structured medical tables. They learn axis-aligned representations.
4. **LSTM for temporal data** — LSTMs suffer from vanishing gradients for longer sequences and cannot parallelize.
5. **CLS-token Transformer for WSI** — a CLS token attending to 1000+ patch tokens is expensive and may not capture hierarchical tissue structure.
6. **No graph structure** — patient similarity (by tumor site, stage, demographics) is never exploited during imputation or fusion.
7. **QualityGate bug** — hidden dim mismatch between train and inference causes silent weight skip.

### 2.2 MIRAGE-Net: Core Design Decisions

| Module | GitHub HCAT | MIRAGE-Net | Why Changed |
|---|---|---|---|
| Tabular encoder | MLP | **KAN (Kolmogorov-Arnold Network)** | KANs learn spline-based functions per feature interaction, provably more expressive than MLPs for structured data, interpretable via activation plots |
| Temporal encoder | BiLSTM | **RWKV-style Linear Recurrence** | RWKV has O(L) complexity, no vanishing gradient, trains like Transformer but infers like RNN, handles the 16-bin blood sequence efficiently |
| WSI aggregator | Transformer CLS | **Hierarchical ABMIL (H-ABMIL) with Nyströmformer** | Two-level: patch→cluster attention then cluster→slide attention. Nyströmformer approximates attention in O(n) for thousands of patches |
| Cross-modal fusion | TransformerEncoder on 5 tokens | **Perceiver IO** | Fixed 64-latent array cross-attends to ALL modalities; latents are query-driven, forcing the model to learn what to extract from each modality. This is fundamentally richer than stacking 5 tokens. |
| Imputation | Joint VAE | **Graph-VAE with patient similarity prior** | Add patient similarity graph (GNN message passing) as a prior during VAE imputation — much more principled than the current graph-smoothing postprocessing |
| Code handling | Ignored | **Code2Vec-style hierarchical code embedding** | ICD codes have hierarchical structure (C32 → C32.0 → C32.0 R); embed at all levels and pool |
| Task routing | Softmax-weighted branch | **Uncertainty-aware attention routing** | Route through uncertainty heads; if a modality is imputed (uncertain), reduce its contribution to the final decision via learned uncertainty gate |

### 2.3 Architecture Overview — MIRAGE-Net

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STAGE 0: MODALITY-SPECIFIC PREPROCESSING (offline)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Clinical JSON    → Graph-VAE Imputer → KAN Encoder → 512d
  Pathological JSON→ Graph-VAE Imputer → KAN Encoder → 512d
  Blood JSON       → Binning → PhysioFill → RWKV Encoder → 512d
  Text (EN only)   → Bio_ClinicalBERT / TF-IDF → Projection → 512d
  ICD + OPS codes  → Hierarchical Code2Vec → 512d
  WSI h5 files     → H-ABMIL + Nyströmformer → 512d

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STAGE 1: QUALITY GATING (per modality)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Input:  [emb_stack (B×6×512), quality (B×6), present_mask (B×6)]
  QualityGate(n_mod=6, hidden=64):  Linear(6→64)→ReLU→Linear(64→6)→Sigmoid
  Gated embedding: emb_i = emb_i * gate_i * present_mask_i

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STAGE 2: CROSS-MODAL IMPUTATION (Graph-VAE, online during training)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Build patient similarity graph: k-NN on clinical+pathological features
  GraphVAE:
    - Per-modality encoder: GATv2Conv (graph attention) → μ, log-σ
    - Shared latent z: reparameterize
    - Per-modality decoder: MLP → reconstructed embedding
    - Fill missing modality embeddings with decoded output
    - Uncertainty = variance of decoded distribution → used in Stage 3

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STAGE 3: PERCEIVER IO CROSS-MODAL FUSION (the novel core)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Latent array:  L = 64 latents × 512d  (learned, shared across patients)

  Cross-attention (latents attend to modality tokens):
    Q = Latents (64×512)
    K, V = Modality embeddings (6×512)
    → Output: updated latents (64×512)
    → Repeat with each modality group separately:
        Group A: clinical + pathological + codes (tabular group)
        Group B: temporal (blood group)
        Group C: text (semantic group)
        Group D: spatial WSI (imaging group)
    → 4 cross-attention rounds, modality-specific

  Self-attention on latents:
    4 Transformer layers on the 64 latent vectors

  Output:
    Mean-pool latents → 512d patient representation

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STAGE 4: TASK HEADS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Uncertainty-gated routing:
    route_surv = softmax(Linear(512→4))  # 4 groups
    route_rec  = softmax(Linear(512→4))
    rep_surv = Σ weight_i * group_i_representation
    rep_rec  = Σ weight_i * group_i_representation

  head_surv: KAN(512→256→128→1)   ← KAN instead of MLP for interpretability
  head_rec:  KAN(512→256→128→1)

  contrast_proj: Linear(512→128) for InfoNCE loss

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TRAINING STRATEGY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  5-Fold Stratified CV
  Focal loss (γ=2) + Label smoothing (ε=0.05)
  Masked loss (ignore -1 labels)
  InfoNCE contrastive loss (temperature=0.07)
  Graph-VAE KL loss (β-VAE with β=0.5)
  AdamW lr=5e-5, weight_decay=1e-4
  CosineAnnealingWarmRestarts (T_0=15, T_mult=2)
  Gradient clip max_norm=1.0
  Modality dropout (p=0.1 per modality, clinical always kept)
```

### 2.4 What Makes MIRAGE-Net Genuinely Novel

1. **KAN tabular encoders** (2024, MIT) — No other multimodal cancer paper uses KAN for tabular modality encoding. KANs replace weight matrices with learnable 1D spline functions, making the encoder interpretable (each spline shows how a clinical variable contributes to the embedding).

2. **Perceiver IO cross-modal fusion** (2021, DeepMind; first use in head-neck cancer prediction) — Instead of stacking 6 modality tokens and applying self-attention, we use 64 learned latent queries that cross-attend to each modality group. This means the model explicitly queries: "What do I want to know from the blood data? What from the WSI?" — a fundamentally more principled fusion.

3. **Graph-VAE imputation** — The patient similarity graph is built from clinical+pathological features (well-observed modalities) and used as a GNN prior in the VAE imputer. Missing modality information is imputed by borrowing from similar patients through graph message passing.

4. **RWKV for blood time-series** — RWKV (2023, BlinkDL) achieves Transformer-level accuracy with O(L) memory and O(1) inference cost per token. Unlike LSTM, it parallelizes during training. Unlike standard Transformer, it does not require storing the full attention matrix.

5. **H-ABMIL with Nyströmformer** for WSI — Two-level MIL: first cluster patches into ~32 clusters using k-means, apply intra-cluster attention to get cluster embeddings, then apply inter-cluster attention across 32 clusters. Nyströmformer (O(n√n)) approximates full attention for large patch counts.

6. **Hierarchical Code2Vec for ICD/OPS** — ICD codes like `C32.0` have three levels: `C` (malignant neoplasm), `C32` (larynx), `C32.0` (glottis). Embed each level separately, sum embeddings. This gives a structured semantic representation instead of a flat bag-of-words.

### 2.5 Component Implementation References

| Component | Paper | Install | Fallback |
|---|---|---|---|
| KAN | "KAN: Kolmogorov-Arnold Networks" (Liu et al. 2024) | `pip install pykan` | MLP with GELU |
| RWKV | "RWKV: Reinventing RNNs for the Transformer Era" (Peng et al. 2023) | implement manually (simple) | BiLSTM |
| Perceiver IO | "Perceiver IO" (Jaegle et al. 2021, DeepMind) | implement manually | Transformer on stacked tokens |
| Nyströmformer | "Nyströmformer" (Xiong et al. 2021) | `pip install nystrom-attention` | Standard MHSA with max_seq=64 |
| GATv2 | "How Attentive are Graph Attention Networks?" (Brody et al. 2022) | `pip install torch-geometric` | Simple k-NN mean pooling |
| Code2Vec | "Code2Vec: Learning Distributed Representations of Code" (Alon et al. 2019) | implement manually | TF-IDF on codes |

---

## 3. Preprocessing Plan — Step by Step

> **⚠️ PATH UPDATE REQUIRED:** After reading this section, update all `PATH_TO_*` placeholders at the end of this document.

---

### STEP 1 — Clinical Tabular Preprocessing (`preprocess/clinical.py`)

**Input:**  
`PATH_TO_STRUCTURED_DATA/StructuredData/clinical_data.json`

**Expected shape:** 763 rows × 32 columns.

**Columns to DROP (they become labels or are direct leakage):**
```python
DROP_COLS = [
    "survival_status",               # becomes label — DO NOT use as feature
    "survival_status_with_cause",    # becomes label — DO NOT use as feature
    "days_to_last_information",      # used to compute label — then drop
    "days_to_recurrence",            # used to compute label — then drop
    "recurrence",                    # used to compute label — then drop
    "patient_id",                    # identifier, not a feature (keep separately)
]
```

**Columns to KEEP as numeric features (9 numeric):**
```python
NUMERIC_COLS = [
    "year_of_initial_diagnosis",
    "age_at_initial_diagnosis",
    "days_to_first_treatment",
    "days_to_progress_1",
    "days_to_progress_2",
    "days_to_metastasis_1",
    "days_to_metastasis_2",
    "days_to_metastasis_3",
    "days_to_metastasis_4",
]
```

**Columns to KEEP as categorical features (frequency-encode these):**
```python
CATEGORICAL_COLS = [
    "sex",
    "smoking_status",
    "primarily_metastasis",
    "first_treatment_intent",
    "first_treatment_modality",
    "adjuvant_treatment_intent",
    "adjuvant_radiotherapy",
    "adjuvant_radiotherapy_modality",
    "adjuvant_systemic_therapy",
    "adjuvant_systemic_therapy_modality",
    "adjuvant_radiochemotherapy",
    "progress_1",
    "progress_2",
    "metastasis_1_locations",
    "metastasis_2_locations",
    "metastasis_3_locations",
    "metastasis_4_locations",
]
```

**Step-by-step detailed logic:**

1. Load JSON → pandas DataFrame. Assert shape = (763, 32).
2. Assert `patient_id` column exists, has no duplicates.
3. Set `patient_id` as index (zero-padded 3-digit string: `str(int(x)).zfill(3)`).
4. **Derive labels (do this BEFORE dropping columns):**
   ```python
   # 5-year survival label
   died_tumor = df["survival_status_with_cause"].str.lower().str.contains("tumor", na=False)
   followup = pd.to_numeric(df["days_to_last_information"], errors="coerce")
   rec_flag = df["recurrence"].str.lower()
   days_rec = pd.to_numeric(df["days_to_recurrence"], errors="coerce")
   
   surv_label = np.full(len(df), -1, dtype=np.int8)
   surv_label[died_tumor & (followup <= 1825)] = 0   # died within 5yr from tumor
   surv_label[(~died_tumor) & (followup >= 1825)] = 1 # alive at 5yr
   # Ambiguous cases (alive but <5yr followup, or died non-tumor) remain -1
   
   # 2-year recurrence label
   rec_label = np.full(len(df), -1, dtype=np.int8)
   rec_label[(rec_flag == "yes") & (days_rec <= 730)] = 1   # recurred within 2yr
   rec_label[(rec_flag == "no") & (followup >= 730)] = 0    # no recurrence, sufficient followup
   ```
5. Drop `DROP_COLS` from the DataFrame.
6. For each numeric column: `pd.to_numeric(df[col], errors="coerce")` → NaN if unparseable.
7. For each categorical column: frequency-encode as `value_counts(normalize=True)` map.
   - Missing values → `"__MISSING__"` before encoding.
   - Store the encoding map in joblib for inference.
8. Create missingness indicator for each feature column: `miss__colname` = 1 if original was NaN, 0 otherwise. This adds ~26 more binary columns.
9. Stack all features into matrix `X_raw` of shape `(763, n_features)`.
10. Fit `SimpleImputer(strategy="median")` on `X_raw`. Save as `median_imputer`. Transform.
11. Fit `StandardScaler()` on median-imputed data. Save as `scaler`. Transform → `X_std`.
12. **Graph-VAE Imputation:**
    - Build patient similarity graph: k-NN (k=10) on first 9 numeric columns after scaling.
    - Use `torch_geometric` `GATv2Conv` to do 2 rounds of message passing.
    - VAE: encoder is GATv2 → μ (64d) + log-σ (64d). Decoder is MLP(64→128→n_features).
    - Train for 60 epochs with masked reconstruction loss (only reconstruct observed entries) + KL(β=0.5).
    - Sample M=5 imputations → stack into shape `(5, 763, n_features)`.
    - If `torch_geometric` unavailable: fall back to standard VAE without graph prior.
13. Aggregate: `features_mean = mean over M imputations`, `features_var = var over M`.
14. For OBSERVED entries: replace imputed values with original observed values.
15. Build AE input: `X_in = concat([features_mean, features_var, missing_mask_as_float], axis=1)`.
16. **KAN Encoder (replaces DenoisingAE):**
    ```python
    from kan import KAN
    # Architecture: [X_in.shape[1], 512, 512]
    # Each layer uses B-spline activation with k=3, grid=5
    # Input dim typically ~80-100 (features_mean + features_var + mask)
    # Train as denoising: randomly zero out 10% of input, reconstruct X_in
    # Extract bottleneck (512d) as the patient embedding
    ```
    - Train for 40 epochs, Adam lr=1e-3, batch=64.
    - If `pykan` unavailable: fall back to DenoisingAE (MLP).
17. Save outputs:
    - `clinical_preprocessed.h5`: `patient_id`, `features_mean`, `features_var`, `missing_mask`, `surv_5yr_label`, `rec_2yr_label`, `feature_names`
    - `clinical_embedding_512.h5`: `patient_id`, `embedding_512`, `surv_5yr_label`, `rec_2yr_label`
    - `clinical_preproc_objects.joblib`: `feature_names`, `median_imputer`, `scaler`, `cat_encoders`, `vae_model` reference
    - `clinical_kan_encoder.pt` (or `clinical_embedding_ae.pt` if fallback)

**Key constants:**
```python
DAYS_5Y = 1825        # 5 * 365
DAYS_2Y = 730         # 2 * 365
M_IMPUTATIONS = 5     # VAE draws
VAE_LATENT_DIM = 64   # VAE bottleneck
VAE_BETA = 0.5        # beta-VAE weighting for KL
GRAPH_K = 10          # k-NN for patient graph
KAN_GRID = 5          # B-spline grid points per activation
KAN_K = 3             # B-spline order
EMBED_DIM = 512
SEED = 42
```

**Validation:** After running, print:
```
clinical patients: 763
n_features (before missingness indicators): ~26
n_features (total including miss__ cols): ~52
label distribution surv: {0: 165, 1: 242, -1: 356}
label distribution rec: {0: 412, 1: 133, -1: 218}
embedding shape: (763, 512)
```

---

### STEP 2 — Pathological Tabular Preprocessing (`preprocess/pathological.py`)

**Input:**  
`PATH_TO_STRUCTURED_DATA/StructuredData/pathological_data.json`

**Expected shape:** 763 rows × 18 columns.

**Column-by-column cleaning rules:**

| Column | Type | Special Cleaning |
|---|---|---|
| `patient_id` | ID | zero-pad to 3 digits, set as index |
| `primary_tumor_site` | categorical | frequency encode; top values: Oropharynx, Larynx, Oral_Cavity |
| `pT_stage` | categorical | frequency encode; top: pT2, pT1, pT3 |
| `pN_stage` | categorical | frequency encode; top: pN0, NX, pN2b |
| `grading` | categorical | frequency encode; top: G3, G2 |
| `hpv_association_p16` | categorical | frequency encode; top: not_tested, negative, positive |
| `number_of_positive_lymph_nodes` | numeric | `pd.to_numeric(errors="coerce")`; 19.9% missing |
| `number_of_resected_lymph_nodes` | numeric | `pd.to_numeric(errors="coerce")` |
| `perinodal_invasion` | categorical | frequency encode; 47.8% missing |
| `lymphovascular_invasion_L` | categorical (binary) | map yes→1, no→0 |
| `vascular_invasion_V` | categorical (binary) | map yes→1, no→0 |
| `perineural_invasion_Pn` | categorical (binary) | map yes→1, no→0 |
| `resection_status` | categorical | frequency encode; R0=685, R1=51, RX=25 |
| `resection_status_carcinoma_in_situ` | categorical | frequency encode |
| `carcinoma_in_situ` | categorical (binary) | map yes→1, no→0 |
| `closest_resection_margin_in_cm` | mixed string/numeric | **special**: `"<0.1"` → `0.05`; `"0.5"` → `0.5`; else coerce |
| `histologic_type` | categorical | frequency encode; top: SCC_Conventional-Keratinizing |
| `infiltration_depth_in_mm` | numeric | `pd.to_numeric(errors="coerce")` |

**Step-by-step detailed logic:**

1. Load JSON → DataFrame. Assert `patient_id` exists.
2. Normalize `patient_id`: `str(int(x)).zfill(3)`.
3. Clean `closest_resection_margin_in_cm`:
   ```python
   def clean_margin(x):
       if pd.isna(x): return np.nan
       s = str(x).strip()
       if s.startswith("<"): return float(s[1:]) / 2.0
       try: return float(s)
       except: return np.nan
   df["closest_resection_margin_in_cm"] = df["closest_resection_margin_in_cm"].apply(clean_margin)
   ```
4. Force numeric on `infiltration_depth_in_mm`, `number_of_positive_lymph_nodes`, `number_of_resected_lymph_nodes`.
5. Map binary categoricals (yes/no) → 1/0. NaN stays NaN.
6. Frequency-encode remaining categoricals. Store encoders.
7. Create `miss__colname` indicators for every column.
8. Stack into `X_raw` (763 × ~36 features after missingness indicators).
9. Same Graph-VAE imputation pipeline as STEP 1 (share the patient graph).
   - Adjust k-NN graph: build on UNION of clinical + pathological numeric columns.
   - Use same graph for both modalities (they are the same 763 patients).
10. KAN encoder: same architecture as clinical.
11. Save outputs:
    - `pathological_preprocessed.h5`
    - `pathological_embedding_512.h5`
    - `pathological_preproc_objects.joblib`
    - `pathological_kan_encoder.pt`

**Validation:** After running, print:
```
pathological patients: 763
n_features total: ~36
Overlap with clinical: 763 (should be perfect)
embedding shape: (763, 512)
```

---

### STEP 3 — Blood Temporal Preprocessing with RWKV (`preprocess/temporal.py`)

**Input:**  
`PATH_TO_STRUCTURED_DATA/StructuredData/blood_data.json` — 23,234 records  
`PATH_TO_STRUCTURED_DATA/StructuredData/blood_data_reference_ranges.json` — 38 analytes

**Data fields:** `patient_id`, `value`, `unit`, `analyte_name`, `LOINC_code`, `LOINC_name`, `group`, `days_before_first_treatment`

**Step-by-step detailed logic:**

1. Load blood JSON → DataFrame. Normalize:
   - `patient_id` → zero-pad 3-digit string
   - `analyte_name` → strip whitespace
   - `value` → `pd.to_numeric(errors="coerce")`
   - `days_before_first_treatment` → int (range: 0–14 days from EDA)

2. Load reference ranges JSON → dict: `analyte_name → {male_min, male_max, female_min, female_max}`.

3. **Analyte selection:** Keep analytes present in ≥ 3 patients. From EDA: 38 analytes with coverage ranging from 38% (Urea) to 90% (Leukocytes, Hemoglobin etc.).
   ```python
   # Top analytes by patient coverage (from EDA):
   # Creatinine: 684 pts, Lymphocytes: 684, Granulocytes: 684, Hematocrit: 684
   # Hemoglobin: 684, Platelets: 684, Leukocytes: 684, etc.
   # All 38 analytes are valid (all present in >50% of patients)
   ```

4. **Time binning (16 bins over 0–14 days):**
   ```python
   SEQ_LEN = 16
   TIME_WINDOW = 14  # days before treatment
   edges = np.linspace(0, TIME_WINDOW, SEQ_LEN + 1)  # 17 edges → 16 bins
   # Bin each measurement: bin_idx = np.digitize(day, edges) - 1, clipped to [0, 15]
   # If multiple measurements in same bin → take mean
   ```
   Result per patient: matrix of shape `(38, 16)` — analytes × time bins — NaN where no measurement.

5. **Physiology-aware imputation (two-stage):**
   - **Stage A:** For analytes with ALL 16 bins missing → fill entire row with `(male_midpoint + female_midpoint) / 2` from reference ranges.
   - **Stage B:** For analytes with partial missingness → 1D linear interpolation along the time axis (fill endpoints with nearest observed value).

6. **Cohort-level KNN imputation:** Flatten each patient matrix to `(38×16,) = 608d`. Apply `KNNImputer(n_neighbors=8)` across all 693 patients. Save `knn_imputer` in joblib.

7. **StandardScaler:** Fit on flattened KNN-imputed data. Save `scaler` in joblib.

8. Reshape each patient to `(16, 38)` — time-first, analytes as features. This is the sequence input to RWKV.

9. **RWKV Temporal Encoder (replaces BiLSTM):**
   ```python
   class RWKVBlock(nn.Module):
       """
       Simplified RWKV-style block:
       - Time-mixing: linear recurrence with learned decay w, receptance r, key k, value v
       - Channel-mixing: gated FFN
       Both computed in parallel during training (like Transformer), sequential at inference.
       """
       def __init__(self, d_model, d_ffn=None):
           # d_model = n_analytes = 38 (padded to 64 for cleaner dims)
           # d_ffn = 4 * d_model
           # Time-mixing parameters: w (decay), u (bonus), r, k, v projections
           # Channel-mixing parameters: key, value, receptance projections
   
   class RWKVTemporalEncoder(nn.Module):
       """
       Stack of 3 RWKV blocks on sequence (16, 38).
       Input:  (B, seq_len=16, n_analytes=38)
       Output: mean over time → Linear(38→512) → 512d embedding
       """
       def __init__(self, input_dim=38, d_model=64, n_layers=3, out_dim=512):
           self.input_proj = nn.Linear(input_dim, d_model)
           self.layers = nn.ModuleList([RWKVBlock(d_model) for _ in range(n_layers)])
           self.out_proj = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, out_dim))
       def forward(self, x):
           # x: (B, 16, 38)
           x = self.input_proj(x)         # (B, 16, 64)
           for layer in self.layers:
               x = layer(x)              # (B, 16, 64)
           z = x.mean(dim=1)            # (B, 64)
           return self.out_proj(z)       # (B, 512)
   ```
   - Train as denoising: randomly zero 10% of time bins during training, reconstruct original.
   - Loss: MSE reconstruction on all 38 analytes across 16 time steps.
   - Epochs: 50, batch=64, Adam lr=1e-3.
   - **Fallback** if RWKV too complex to implement: 2-layer BiLSTM(hidden=256, bidirectional=True) → project to 512d. Same training loop.

10. Compute quality score per patient: `q = (observed_cells / total_cells)` where `total_cells = 38 * 16 = 608`.

11. Save outputs:
    - `temporal_embedding_512.h5`: `patient_id`, `embedding_512`, `quality_score`
    - `temporal_preproc_objects.joblib`: `analytes`, `edges`, `knn_imputer`, `scaler`
    - `temporal_rwkv_encoder.pt` (or `temporal_lstm_encoder.pt` if fallback)

**Validation:** After running, print:
```
temporal patients: 693
n_analytes: 38
seq_len: 16
time_window_days: 14
mean quality score: ~0.35 (expected, most measurements at day 0)
embedding shape: (693, 512)
```

---

### STEP 4 — Text Semantic Preprocessing (`preprocess/semantic.py`)

**⚠️ IMPORTANT: Use ONLY English folders. Do NOT use German folders.**

**Inputs (English only):**  
`PATH_TO_TEXT_DATA/histories_english/` — 529 files  
`PATH_TO_TEXT_DATA/reports_english/` — 742 files  
`PATH_TO_TEXT_DATA/surgery_descriptions_english/` — 763 files  

**Do NOT use:**  
- `histories/` (German)  
- `reports/` (German)  
- `surgery_descriptions/` (German)  

**What to do:**
1. Load all `.txt` files. Extract patient ID from filename using regex `(\d{3})` → zero-pad to 3 digits.
2. Clean text: remove HTML tags, collapse whitespace, strip.
3. For each patient, concatenate `[history + "\n" + report + "\n" + surgery_description]` into one document.
4. **Embedding:**
   - If `transformers` + `torch` available: use `Bio_ClinicalBERT` (emilyalsentzer/Bio_ClinicalBERT), mean-pool token embeddings with attention mask.
   - Fallback: TF-IDF (max_features=20000, ngram_range=(1,2)) → TruncatedSVD to 512d.
5. If BERT used (768d output): fit `TruncatedSVD(n_components=512)` on all patient embeddings, save as `projector_768_to_512` in joblib.
6. Save `text_semantic_embedding_512.h5`, `text_semantic_preproc_objects.joblib`.

**For patients missing a modality (e.g., no history file):** use zero vector for that sub-embedding, reduce weight in cross-modality average.

---

### STEP 5 — ICD and OPS Code Embedding with Hierarchical Code2Vec (`preprocess/codes.py`)

**Inputs:**  
`PATH_TO_TEXT_DATA/icd_codes/` — 712 files (`SurgeryReport_ICD_Codes_NNN.txt`)  
`PATH_TO_TEXT_DATA/ops_codes/` — 734 files (`SurgeryReports_OPS_Codes_NNN.txt`)

**ICD file format example:**
```
Glottiskarzinom[C32.0 R]
Bösartige Neubildung: Oropharynx überlappend[C10.8 B]
```

**OPS file format example:**
```
Frontolaterale Kehlkopfteilresektion[5-302.7 ]
Radikale zervikale Lymphadenektomie [Neck dissection][5-403.5 R]
```

**Step-by-step detailed logic:**

1. Load all ICD files. Extract patient ID from filename: `re.search(r'(\d{3})', fname).group(1)`.
2. Parse each file to extract codes using: `re.findall(r'\[([A-Z]\d{2}[\.\d]*)', line)` for ICD, `re.findall(r'\[(\d-\d{3}[\.\d]*)', line)` for OPS.
3. **Hierarchical decomposition of ICD codes:**
   ```python
   # Example: "C32.0" →
   # Level 1: "C"    (chapter: malignant neoplasm)
   # Level 2: "C32"  (category: larynx)
   # Level 3: "C32.0"(subcategory: glottis)
   # Create one token per level
   def decompose_icd(code):
       parts = []
       if len(code) >= 1: parts.append(code[0])       # letter chapter
       if len(code) >= 3: parts.append(code[:3])       # 3-char category
       parts.append(code)                              # full code
       return parts
   ```
4. Similarly for OPS codes: `"5-302.7"` → `["5", "5-302", "5-302.7"]`.
5. Build vocabulary: union of all code parts from all patients. Assign integer IDs.
   - Expected: ~150 ICD base codes × 3 levels + ~400 OPS base codes × 3 levels ≈ 1,650 vocab tokens.
6. **Per-patient representation:**
   - For each patient: collect all code parts (ICD + OPS combined).
   - Build multi-hot vector of shape `(vocab_size,)`.
   - Apply TF-IDF weighting: fit `TfidfTransformer` on the multi-hot matrix. Transform.
7. **Code2Vec-style hierarchical pooling:**
   ```python
   class HierarchicalCodeEmbedder(nn.Module):
       def __init__(self, vocab_size, embed_dim=128, out_dim=512):
           self.embed = nn.EmbeddingBag(vocab_size, embed_dim, mode="mean")
           # Three separate embeddings for three hierarchy levels
           self.level_embeds = nn.ModuleList([
               nn.EmbeddingBag(vocab_size, embed_dim, mode="mean")
               for _ in range(3)  # chapter, category, subcategory
           ])
           self.attention = nn.Sequential(
               nn.Linear(embed_dim, 1), nn.Softmax(dim=0)
           )
           self.out_proj = nn.Linear(embed_dim * 3, out_dim)
       def forward(self, code_ids_per_level):
           # code_ids_per_level: list of 3 tensors (per hierarchy level)
           level_vecs = [emb(ids) for emb, ids in zip(self.level_embeds, code_ids_per_level)]
           concat = torch.cat(level_vecs, dim=-1)   # (B, embed_dim*3)
           return self.out_proj(concat)              # (B, 512)
   ```
   - Train with reconstruction loss (masked autoencoding on code presence).
   - Alternatively: use TruncatedSVD(n_components=512) on TF-IDF matrix as a simpler fallback.
8. For patients with no ICD or OPS files (51 and 29 patients respectively): use zero vector + `present_mask = 0` for this modality.
9. Save:
   - `codes_embedding_512.h5`: `patient_id`, `embedding_512`
   - `codes_preproc_objects.joblib`: `vocab`, `tfidf_transformer`, `code_embedder_weights`
   - `codes_hierarchical_embedder.pt`

---

### STEP 6 — WSI Spatial Preprocessing with H-ABMIL + Nyströmformer (`preprocess/spatial.py`)

**Inputs (scan ALL 7 directories):**  
`PATH_TO_WSI/WSI_LymphNode/h5_files/`  
`PATH_TO_WSI/WSI_PrimaryTumor/WSI_PrimaryTumor_CUP/h5_files/`  
`PATH_TO_WSI/WSI_PrimaryTumor/WSI_PrimaryTumor_Hypopharynx/h5_files/`  
`PATH_TO_WSI/WSI_PrimaryTumor/WSI_PrimaryTumor_Larynx/h5_files/`  
`PATH_TO_WSI/WSI_PrimaryTumor/WSI_PrimaryTumor_OralCavity/h5_files/`  
`PATH_TO_WSI/WSI_PrimaryTumor/WSI_PrimaryTumor_Oropharynx_Part1/h5_files/`  
`PATH_TO_WSI/WSI_PrimaryTumor/WSI_PrimaryTumor_Oropharynx_Part2/h5_files/`

**⚠️ CRITICAL: Multi-h5 Patient Fix**
Some patients have multiple h5 files: e.g., `PrimaryTumor_HE_036.h5` and `PrimaryTumor_HE_036_a.h5`.  
Extract patient ID with: `re.search(r'HE_(\d+)', fname).group(1)` (ignore the `_a`, `_b` suffix).  
This correctly groups `_036` and `_036_a` under patient `036`.

**H5 file internal structure (verified from EDA):**
- `features`: shape `(n_patches, 1024)` — UNI patch embeddings, float32
- `coords`: shape `(n_patches, 2)` — pixel coordinates (x, y), int32

**Step-by-step detailed logic:**

1. Scan all 7 directories. Build dict: `patient_id → {"primary": [path1, path2, ...], "lymph": [path3, ...]}`.
2. For each patient, load and concatenate all primary tumor patch files:
   ```python
   all_feats, all_coords = [], []
   for path in patient_primary_files:
       with h5py.File(path) as f:
           all_feats.append(f["features"][:])
           all_coords.append(f["coords"][:])
   feats = np.concatenate(all_feats)   # (total_patches, 1024)
   coords = np.concatenate(all_coords) # (total_patches, 2)
   ```
3. L2-normalize each patch feature: `feats /= np.linalg.norm(feats, axis=1, keepdims=True)`.
4. Normalize coordinates to [0,1]: `coords_norm = (coords - coords.min(0)) / (coords.max(0) - coords.min(0) + 1e-6)`.
5. **Patch reduction if > MAX_PATCHES=2048:**
   Use MiniBatchKMeans on `[feats | coords_norm]` concatenated. Cluster center feature parts = representative patches.
6. **Positional encoding:** For each patch, compute position embedding:
   ```python
   class PositionalMLP(nn.Module):
       # Input: coords_norm (N, 2)
       # Output: positional bias (N, 512)
       def __init__(self): nn.Sequential(Linear(2,256), GELU, Linear(256,512))
   ```
7. **PatchProjector:** Linear(1024 → 512) + GELU + LayerNorm. Projects UNI 1024d to model 512d. Add positional bias.
8. **Hierarchical ABMIL (H-ABMIL):**
   ```python
   class HABMIL(nn.Module):
       """
       Two-level attention MIL:
       Level 1 (patch→cluster):
         - Cluster patches into K=32 groups via k-means on projected features
         - Within each cluster: ABMIL attention pooling → cluster embedding (512d)
       Level 2 (cluster→slide):
         - Apply Nyströmformer on 32 cluster embeddings
         - Nyströmformer uses n_landmarks=8 (so full attention approx with 8 landmarks)
         - Output: mean-pool the 32 attended cluster embeddings → slide embedding (512d)
       """
       def __init__(self, d_model=512, n_clusters=32, n_landmarks=8):
           # Level 1 attention: V=Linear(512,512), A=tanh(Linear(512,128)), gating=sigmoid
           # Level 2: NystromAttention(dim=512, heads=8, landmarks=n_landmarks)
   ```
   - **Nyströmformer** approximates `softmax(QK^T/√d)V` using n_landmarks=8 anchor points. Install: `pip install nystrom-attention`.
   - If unavailable: standard MHSA on 32 cluster embeddings is fine (only 32 tokens, no efficiency issue).

9. **LymphNode and PrimaryTumor separate embeddings:**
   - Compute `pt_emb (512d)` from primary tumor patches using H-ABMIL.
   - Compute `ln_emb (512d)` from lymph node patches using H-ABMIL (share weights or separate).
   - Final: `spatial_emb = Linear(1024→512)(concat([pt_emb, ln_emb]))` — learned fusion of both sites.
   - If patient has no lymph node WSI: `ln_emb = zeros(512)`, reduced weight in fusion.
   - If patient has no primary tumor WSI: `pt_emb = zeros(512)`, reduced weight.

10. **Monte-Carlo uncertainty:** T=8 forward passes with dropout enabled → mean ± variance. Store both.

11. Save:
    - `spatial_embedding_512.h5`: `patient_id`, `embedding_512`, `embedding_var_512`, `quality_score`, `n_patches`, `tumor_site`
    - `spatial_preproc_objects.joblib`: args, clustering params
    - `positional_mlp.pt`, `patch_projector.pt`, `habmil_model.pt`

**Validation:** After running, print:
```
Total h5 files: 1078
Unique patients: 747
Patients with >1 primary tumor file: ~328
Patients missing WSI entirely: 16
Patients missing lymph node WSI: ~394 (only 369 LN files exist)
embedding shape: (747, 512)
```

---

## 4. Training Plan (`train.py`)

### 4.1 Data Loading and Alignment

```python
def load_and_align_all_modalities(h5_paths: dict) -> dict:
    """
    h5_paths = {
        "clinical": "path/clinical_embedding_512.h5",
        "pathological": "path/pathological_embedding_512.h5",
        "temporal": "path/temporal_embedding_512.h5",
        "semantic": "path/text_semantic_embedding_512.h5",
        "codes": "path/codes_embedding_512.h5",
        "spatial": "path/spatial_embedding_512.h5",
    }
    Alignment rule: clinical patient_ids are the master list (763 patients).
    All other modalities align to clinical by patient_id.
    Missing = zero vector + present_mask = 0.
    """
    # Step 1: Load clinical as master
    # Step 2: For each other modality, build dict {pid: embedding}
    # Step 3: For each of 763 clinical patients:
    #   for each modality: lookup by pid → embedding or zeros(512)
    #   set present_mask[m] = 1 if found, 0 if not
    # Step 4: Stack: emb_stack shape (763, 6, 512)
    #         quality shape (763, 6)
    #         present_mask shape (763, 6) — dtype int8
    # Step 5: Load labels from clinical H5: surv_5yr_label, rec_2yr_label
```

**Modality order (must be consistent between train and inference):**
```
Index 0: clinical tabular
Index 1: pathological tabular
Index 2: blood temporal
Index 3: text semantic
Index 4: ICD+OPS codes
Index 5: WSI spatial
```

### 4.2 MIRAGE-Net Full Model Class

```python
class MIRAGENet(nn.Module):
    """
    Multimodal Integrative Research Architecture with Graph-Enhanced Networks.

    Key differences from GitHub HCAT:
    - 6 modalities (HCAT had 5, ignored ICD/OPS)
    - KAN encoders in prediction heads (HCAT used MLP)
    - Perceiver IO cross-modal fusion (HCAT used self-attention on 5 tokens)
    - Graph-VAE imputation (HCAT used standard VAE)
    - RWKV for temporal in preprocessing (HCAT used BiLSTM)
    - H-ABMIL + Nyströmformer for WSI (HCAT used Transformer CLS)
    - QualityGate hidden=64 EVERYWHERE (HCAT had inconsistency bug)
    - 4 modality groups for Perceiver cross-attention (HCAT had no grouping)
    """

    def __init__(
        self,
        d_model: int = 512,
        n_modalities: int = 6,
        n_latents: int = 64,       # Perceiver IO latent array size
        n_perceiver_layers: int = 4,  # self-attention layers on latents
        n_heads: int = 8,
        dropout: float = 0.2,
        use_graph_vae: bool = True,
        n_impute_iterations: int = 3,
        use_kan_heads: bool = True,   # KAN for prediction heads
    ):
        # --- Imputation module ---
        self.graph_vae_imputer = GraphVAEImputer(
            n_modalities=n_modalities,
            d_model=d_model,
            latent_dim=64,
            n_graph_layers=2,
        )
        # --- Quality gate (hidden=64, FIXED) ---
        self.quality_gate = QualityGate(n_mod=n_modalities, hidden=64)

        # --- Modality positional encoding ---
        self.mod_pos_enc = nn.Parameter(torch.randn(n_modalities, d_model) * 0.02)

        # --- Perceiver IO ---
        # Latent array (shared, learned)
        self.latents = nn.Parameter(torch.randn(n_latents, d_model) * 0.02)
        # 4 modality groups for cross-attention
        self.group_cross_attn = nn.ModuleList([
            nn.MultiheadAttention(d_model, n_heads, batch_first=True, dropout=0.1)
            for _ in range(4)  # tabular, temporal, semantic, spatial
        ])
        # Group assignments: 0=tabular (clinical,patho,codes), 1=temporal, 2=semantic, 3=spatial
        self.group_assignment = {
            "clinical": 0, "pathological": 0, "codes": 0,
            "temporal": 1,
            "semantic": 2,
            "spatial": 3,
        }
        # Self-attention on latents
        perceiver_layer = nn.TransformerEncoderLayer(
            d_model, n_heads, dim_feedforward=2048,
            dropout=dropout, activation="gelu",
            batch_first=True, norm_first=True
        )
        self.perceiver_self_attn = nn.TransformerEncoder(perceiver_layer, n_perceiver_layers)

        # --- Task routing ---
        self.route_surv = nn.Linear(d_model, 4)   # 4 group weights
        self.route_rec  = nn.Linear(d_model, 4)

        # --- Prediction heads (KAN or MLP) ---
        if use_kan_heads:
            # KAN: [512, 256, 128, 1] with spline activations
            self.head_surv = KAN([d_model, 256, 128, 1])
            self.head_rec  = KAN([d_model, 256, 128, 1])
        else:
            self.head_surv = nn.Sequential(
                nn.Linear(d_model, 256), nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 1)
            )
            self.head_rec = nn.Sequential(
                nn.Linear(d_model, 256), nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 1)
            )

        # --- Contrastive projection ---
        self.contrast_proj = nn.Sequential(nn.Linear(d_model, 128), nn.ReLU(), nn.Linear(128, 128))

        # --- Uncertainty tracking ---
        self.uncertainty_weight = nn.Parameter(torch.tensor(0.1))

    def forward(self, emb, quality, present_mask, patient_graph=None):
        B, M, D = emb.shape

        # Step 1: Graph-VAE imputation
        imputed_emb, imp_uncertainty, kl_loss, recon_loss = self.graph_vae_imputer(
            emb, present_mask, quality, patient_graph
        )
        missing = (present_mask == 0).unsqueeze(-1)
        emb = torch.where(missing, imputed_emb, emb)
        quality = torch.where(
            present_mask == 0,
            quality * (1.0 - imp_uncertainty * self.uncertainty_weight),
            quality
        )

        # Step 2: Quality gate + positional encoding
        gate = self.quality_gate(quality) * present_mask.float()  # (B, 6)
        x = emb + self.mod_pos_enc.unsqueeze(0)                   # (B, 6, 512)
        x = x * gate.unsqueeze(-1)                                # mask + gate

        # Step 3: Perceiver IO cross-attention
        # Expand latent array to batch
        latents = self.latents.unsqueeze(0).expand(B, -1, -1)     # (B, 64, 512)

        # Group modalities and cross-attend
        group_outputs = []
        for g_idx, mod_indices in enumerate([
            [0, 1, 4],  # tabular: clinical, pathological, codes
            [2],        # temporal
            [3],        # semantic
            [5],        # spatial
        ]):
            group_tokens = x[:, mod_indices, :]                   # (B, n_mods_in_group, 512)
            # Cross-attention: latents query group tokens
            attended, _ = self.group_cross_attn[g_idx](
                query=latents, key=group_tokens, value=group_tokens
            )
            latents = latents + attended                          # residual update
            group_outputs.append(latents.mean(dim=1))            # (B, 512) per group

        # Step 4: Self-attention on updated latents
        latents = self.perceiver_self_attn(latents)              # (B, 64, 512)
        pooled = latents.mean(dim=1)                             # (B, 512)

        # Step 5: Task-specific routing
        group_stack = torch.stack(group_outputs, dim=1)          # (B, 4, 512)
        w_surv = torch.softmax(self.route_surv(pooled), dim=-1).unsqueeze(-1)  # (B, 4, 1)
        w_rec  = torch.softmax(self.route_rec(pooled), dim=-1).unsqueeze(-1)
        rep_surv = (w_surv * group_stack).sum(dim=1)             # (B, 512)
        rep_rec  = (w_rec  * group_stack).sum(dim=1)

        # Step 6: Predictions
        logit_surv = self.head_surv(rep_surv).squeeze(-1)
        logit_rec  = self.head_rec(rep_rec).squeeze(-1)
        cproj = F.normalize(self.contrast_proj(pooled), dim=-1)

        return {
            "logit_surv": logit_surv,
            "logit_rec": logit_rec,
            "rep": pooled,
            "cproj": cproj,
            "kl_loss": kl_loss,
            "imputation_loss": recon_loss,
            "group_reps": group_stack,
        }
```

### 4.3 GraphVAEImputer — Detailed Specification

```python
class GraphVAEImputer(nn.Module):
    """
    Graph-conditioned VAE for cross-modal imputation.
    Patient similarity graph built from clinical+pathological features.
    GATv2Conv aggregates neighbor information before encoding.

    Forward pass:
      1. For each patient node: aggregate neighbor embeddings via GATv2 message passing.
      2. Concatenate own embedding + neighbor aggregate → richer representation.
      3. Encode to (mu, logvar) → sample z (64d latent).
      4. Decode z → reconstructed embeddings for ALL modalities.
      5. Use decoded embeddings to fill MISSING modalities only.

    Loss:
      recon_loss = MSE(decoded, original) masked to OBSERVED modalities only
      kl_loss = -0.5 * sum(1 + logvar - mu^2 - exp(logvar)) / batch_size * beta
      beta = 0.5 (beta-VAE)
    """
    def __init__(self, n_modalities=6, d_model=512, latent_dim=64, n_graph_layers=2):
        self.n_mod = n_modalities
        # GATv2 message passing layers (or simple k-NN averaging if torch_geometric unavailable)
        self.gat_layers = nn.ModuleList([
            GATv2Conv(d_model, d_model, heads=4, concat=False)
            for _ in range(n_graph_layers)
        ])
        # Encoder: (d_model * 2) → 128 → (mu, logvar) [×2]
        # Factor of 2 because we concatenate own + neighbor
        self.encoder = nn.Sequential(nn.Linear(d_model*2, 256), nn.GELU(), nn.Linear(256, 128))
        self.mu_head = nn.Linear(128, latent_dim)
        self.logvar_head = nn.Linear(128, latent_dim)
        # Per-modality decoders: latent_dim → d_model
        self.decoders = nn.ModuleList([
            nn.Sequential(nn.Linear(latent_dim, 256), nn.GELU(), nn.Linear(256, d_model))
            for _ in range(n_modalities)
        ])
        # Uncertainty heads: latent_dim → 1 scalar per modality
        self.uncertainty_heads = nn.ModuleList([
            nn.Sequential(nn.Linear(latent_dim, 32), nn.GELU(), nn.Linear(32, 1), nn.Sigmoid())
            for _ in range(n_modalities)
        ])
```

**Building the patient graph:**
```python
def build_patient_graph(clinical_features, pathological_features, k=10):
    """
    Build k-NN graph on clinical + pathological numeric features.
    Use sklearn NearestNeighbors. Return edge_index in torch_geometric format.
    Called ONCE before training. Stored as a fixed graph.
    """
    from sklearn.neighbors import NearestNeighbors
    X = np.concatenate([clinical_features, pathological_features], axis=1)
    X = StandardScaler().fit_transform(X)
    nn = NearestNeighbors(n_neighbors=k+1, metric="euclidean")
    nn.fit(X)
    distances, indices = nn.kneighbors(X)
    # Build edge_index: (2, n_nodes * k) tensor
    # edge weight = exp(-distance / sigma) for attention
```

### 4.4 Loss Function — Complete Specification

```python
def compute_total_loss(outputs, surv_targets, rec_targets, alpha):
    """
    All losses combined.
    alpha = {
        "surv": 1.0, "rec": 1.0,
        "kl": 0.05, "impute": 0.1, "contrastive": 0.1
    }
    """
    # --- Masks (only compute loss where label != -1) ---
    surv_mask = (surv_targets != -1)
    rec_mask  = (rec_targets  != -1)

    # --- Label smoothing ---
    eps = 0.05
    surv_smooth = surv_targets.float().clone()
    surv_smooth[surv_mask] = surv_smooth[surv_mask] * (1 - eps) + eps / 2
    rec_smooth  = rec_targets.float().clone()
    rec_smooth[rec_mask] = rec_smooth[rec_mask] * (1 - eps) + eps / 2

    # --- Focal loss with class weighting ---
    def focal_loss_masked(logits, targets, mask, pos_weight, gamma=2.0):
        if not mask.any(): return torch.tensor(0.0)
        l = logits[mask]; t = targets[mask]
        bce = F.binary_cross_entropy_with_logits(l, t, pos_weight=pos_weight, reduction="none")
        p_t = torch.exp(-bce)
        return (((1 - p_t) ** gamma) * bce).mean()

    pos_w_surv = torch.tensor([n_neg_surv / max(1, n_pos_surv)])  # computed from training split
    pos_w_rec  = torch.tensor([n_neg_rec  / max(1, n_pos_rec)])

    loss_surv = focal_loss_masked(outputs["logit_surv"], surv_smooth, surv_mask, pos_w_surv.to(device))
    loss_rec  = focal_loss_masked(outputs["logit_rec"],  rec_smooth,  rec_mask,  pos_w_rec.to(device))

    # --- InfoNCE contrastive loss ---
    # Positive pairs: augmented views of same patient (modality dropout)
    # Negative pairs: all other patients in batch
    def info_nce(z, temperature=0.07):
        z = F.normalize(z, dim=-1)
        sim = torch.matmul(z, z.T) / temperature
        labels = torch.arange(z.size(0), device=z.device)
        return F.cross_entropy(sim, labels)
    loss_con = info_nce(outputs["cproj"])

    # --- Total ---
    total = (
        alpha["surv"] * loss_surv
        + alpha["rec"] * loss_rec
        + alpha["kl"] * outputs["kl_loss"]
        + alpha["impute"] * outputs["imputation_loss"]
        + alpha["contrastive"] * loss_con
    )
    return total, {"surv": loss_surv.item(), "rec": loss_rec.item(),
                   "kl": outputs["kl_loss"].item(), "impute": outputs["imputation_loss"].item(),
                   "con": loss_con.item()}
```

### 4.5 Five-Fold Cross-Validation — Exact Implementation

```python
from sklearn.model_selection import StratifiedKFold
import numpy as np

def make_stratification_labels(surv_labels, rec_labels):
    """
    Create combined label for stratification.
    Only use patients where BOTH labels are known.
    For patients with one unknown: use the known label only.
    """
    combined = np.full(len(surv_labels), -1, dtype=np.int8)
    for i in range(len(surv_labels)):
        s, r = surv_labels[i], rec_labels[i]
        if s != -1 and r != -1:
            combined[i] = s * 2 + r    # 0,1,2,3
        elif s != -1:
            combined[i] = s * 2 + 4   # 4,5
        elif r != -1:
            combined[i] = r + 6       # 6,7
        # else: combined[i] = -1 (unknown both — assigned to any fold)
    return combined

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
stratify_labels = make_stratification_labels(surv_labels, rec_labels)
all_indices = np.arange(len(surv_labels))
# For -1 in stratify_labels: assign to folds proportionally (handle separately)

fold_results = []
for fold_idx, (train_idx, val_idx) in enumerate(skf.split(all_indices, stratify_labels)):
    print(f"\n{'='*60}")
    print(f"FOLD {fold_idx + 1}/5")
    print(f"Train: {len(train_idx)} | Val: {len(val_idx)}")
    
    # Build train/val datasets
    train_ds = MultiModalDataset(all_data, train_idx)
    val_ds   = MultiModalDataset(all_data, val_idx)
    
    # Fresh model for each fold
    model = MIRAGENet(...).to(device)
    
    # Train
    best_metrics = train_one_fold(model, train_ds, val_ds, args)
    fold_results.append({"fold": fold_idx, **best_metrics})

# Aggregate
for metric in ["surv_f1", "surv_auc", "rec_f1", "rec_auc", "avg_f1"]:
    vals = [r[metric] for r in fold_results]
    print(f"{metric}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")
```

### 4.6 Optimizer and Scheduler Configuration

```python
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=5e-5,
    weight_decay=1e-4,
    betas=(0.9, 0.999),
    eps=1e-8
)

# Warmup for first 5 epochs, then cosine anneal
def get_lr_lambda(epoch, warmup=5, total=60):
    if epoch < warmup:
        return float(epoch) / float(max(1, warmup))
    progress = float(epoch - warmup) / float(max(1, total - warmup))
    return max(0.01, 0.5 * (1.0 + math.cos(math.pi * progress)))

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=get_lr_lambda)

# Gradient clipping
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

### 4.7 Modality Dropout Augmentation

```python
def apply_modality_dropout(emb, present_mask, quality, p=0.1):
    """
    During training: randomly zero out non-clinical modalities with probability p.
    Clinical (index 0) is NEVER dropped — it is always 100% present.
    This forces the model to learn robust representations
    even when some modalities are missing.
    """
    B, M, D = emb.shape
    drop_mask = torch.rand(B, M, device=emb.device) < p
    drop_mask[:, 0] = False  # protect clinical
    drop_mask = drop_mask & present_mask.bool()  # only drop actually present ones
    
    emb_aug = emb.clone()
    emb_aug[drop_mask] = 0.0
    present_aug = present_mask.clone()
    present_aug[drop_mask] = 0
    quality_aug = quality.clone()
    quality_aug[drop_mask] = 0.0
    return emb_aug, present_aug, quality_aug
```

### 4.8 Metrics to Save for Journal Paper

After all 5 folds complete, compute and save to `training_summary.json`:

```json
{
  "model_name": "MIRAGE-Net",
  "dataset": "HANCOCK",
  "n_patients": 763,
  "n_modalities": 6,
  "per_fold_results": [
    {
      "fold": 0,
      "best_epoch": 23,
      "surv_f1": 0.0, "surv_auc": 0.0, "surv_acc": 0.0,
      "surv_precision": 0.0, "surv_recall": 0.0, "surv_brier": 0.0,
      "rec_f1": 0.0, "rec_auc": 0.0, "rec_acc": 0.0,
      "rec_precision": 0.0, "rec_recall": 0.0, "rec_brier": 0.0,
      "avg_f1": 0.0,
      "train_loss_curve": [],
      "val_avg_f1_curve": []
    }
  ],
  "aggregate": {
    "surv_f1_mean": 0.0, "surv_f1_std": 0.0,
    "surv_auc_mean": 0.0, "surv_auc_std": 0.0,
    "surv_precision_mean": 0.0, "surv_precision_std": 0.0,
    "surv_recall_mean": 0.0, "surv_recall_std": 0.0,
    "surv_brier_mean": 0.0, "surv_brier_std": 0.0,
    "rec_f1_mean": 0.0, "rec_f1_std": 0.0,
    "rec_auc_mean": 0.0, "rec_auc_std": 0.0,
    "avg_f1_mean": 0.0, "avg_f1_std": 0.0
  },
  "baselines": {
    "unimodal_clinical_lr":      {"surv_f1": 0.0, "rec_f1": 0.0},
    "unimodal_pathological_lr":  {"surv_f1": 0.0, "rec_f1": 0.0},
    "unimodal_temporal_lr":      {"surv_f1": 0.0, "rec_f1": 0.0},
    "unimodal_semantic_lr":      {"surv_f1": 0.0, "rec_f1": 0.0},
    "unimodal_codes_lr":         {"surv_f1": 0.0, "rec_f1": 0.0},
    "unimodal_spatial_lr":       {"surv_f1": 0.0, "rec_f1": 0.0},
    "concat_mlp":                {"surv_f1": 0.0, "rec_f1": 0.0},
    "abmil_6mod":                {"surv_f1": 0.0, "rec_f1": 0.0},
    "random_forest_tabular":     {"surv_f1": 0.0, "rec_f1": 0.0},
    "github_hcat_original":      {"surv_f1": 0.0, "rec_f1": 0.0}
  },
  "ablation": {
    "without_codes_modality":    {"avg_f1_mean": 0.0},
    "without_spatial_modality":  {"avg_f1_mean": 0.0},
    "without_temporal_modality": {"avg_f1_mean": 0.0},
    "without_perceiver_io":      {"avg_f1_mean": 0.0},
    "without_graph_vae":         {"avg_f1_mean": 0.0},
    "mlp_heads_instead_of_kan":  {"avg_f1_mean": 0.0}
  },
  "calibration": {
    "surv_ece_mean": 0.0, "rec_ece_mean": 0.0,
    "surv_brier_mean": 0.0, "rec_brier_mean": 0.0
  },
  "training_config": {
    "d_model": 512, "n_latents": 64, "n_perceiver_layers": 4,
    "n_heads": 8, "dropout": 0.2, "epochs": 60, "batch_size": 32,
    "lr": 5e-5, "weight_decay": 1e-4, "focal_gamma": 2.0,
    "label_smoothing": 0.05, "modality_dropout_p": 0.1,
    "graph_k": 10, "vae_latent_dim": 64, "vae_beta": 0.5,
    "alpha_surv": 1.0, "alpha_rec": 1.0, "alpha_kl": 0.05,
    "alpha_impute": 0.1, "alpha_contrastive": 0.1
  }
}
```

**Additional files to save per fold:**
- `fold_{i}_confusion_matrix_surv.npy` — (2×2) confusion matrix for survival
- `fold_{i}_confusion_matrix_rec.npy` — (2×2) for recurrence
- `fold_{i}_calibration_curve_surv.npy` — probability bins vs actual frequencies
- `fold_{i}_roc_curve_surv.npy` — (fpr, tpr, thresholds)
- `fold_{i}_roc_curve_rec.npy`
- `fold_{i}_best_model.pt`

### 4.9 Baselines Implementation (Under `--run_baselines` Flag)

```python
def run_all_baselines(data, surv_labels, rec_labels, cv_splits):
    """Run after main model training. Results saved to training_summary.json."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.pipeline import Pipeline
    
    results = {}
    
    # Unimodal LR: fit on single modality embedding (512d) per fold
    for mod_name in ["clinical", "pathological", "temporal", "semantic", "codes", "spatial"]:
        fold_f1s = []
        for train_idx, val_idx in cv_splits:
            X_train = data[mod_name][train_idx]
            X_val   = data[mod_name][val_idx]
            for task, labels in [("surv", surv_labels), ("rec", rec_labels)]:
                mask_train = (labels[train_idx] != -1)
                mask_val   = (labels[val_idx]   != -1)
                clf = LogisticRegression(max_iter=1000, class_weight="balanced", C=0.1)
                clf.fit(X_train[mask_train], labels[train_idx][mask_train])
                pred = clf.predict(X_val[mask_val])
                f1 = f1_score(labels[val_idx][mask_val], pred, zero_division=0)
                fold_f1s.append(f1)
        results[f"unimodal_{mod_name}_lr"] = np.mean(fold_f1s)
    
    # Concat-MLP: concatenate all 6 embeddings → 3072d → MLP → predict
    # ABMIL-6mod: treat 6 embeddings as a bag, attention pool, predict
    # Random Forest: only on tabular (clinical + pathological numeric features)
    # GitHub HCAT: run original HCAT code on same folds for fair comparison
    
    return results
```

---

## 5. Inference Plan (`inference/inference.py`)

### 5.1 Input Format (Challenge API)
```
/input/hancock-clinical-data.json
/input/hancock-pathological-data.json
/input/hancock-blood-data.json
/input/hancock-surgery-text-data.json
/input/hancock-primary-tumor-wsi-embeddings.json
/input/hancock-lymph-node-wsi-embeddings.json
```

All files are per-patient (single patient). Inference must handle any subset being missing (empty JSON or missing file).

### 5.2 Per-Modality Inference Functions

Each function lives in `inference/preprocess/`:

**`clinical_inference.py`:**
```python
def get_clinical_embedding(clinical_data: dict, resources_path: Path) -> torch.Tensor:
    # 1. Load clinical_preproc_objects.joblib → get feature_names, median_imputer, scaler, cat_encoders
    # 2. Build feature vector matching training feature_names exactly
    #    - For each feature: lookup from clinical_data dict
    #    - For frequency-encoded features: apply stored cat_encoders[col].get(value, 0.0)
    #    - For missing features: NaN → will be imputed
    #    - For miss__ indicators: 1 if NaN, 0 otherwise
    # 3. Impute: median_imputer.transform(X_raw) → NEVER fit_transform at inference
    # 4. Scale: scaler.transform(X_imputed) → NEVER fit at inference
    # 5. Build AE input: concat([X_scaled, zeros(X_scaled.shape), mask_float], axis=1)
    # 6. Load clinical_kan_encoder.pt → forward pass → 512d embedding
    # Returns: torch.Tensor shape (1, 512)
```

**`codes_inference.py` (NEW — must be added):**
```python
def get_codes_embedding(surgery_text_data: dict, resources_path: Path) -> torch.Tensor:
    # 1. Load codes_preproc_objects.joblib → vocab, tfidf_transformer
    # 2. Parse surgery_text_data for ICD/OPS code patterns using same regex as training
    #    - Look in: surgery_text_data.get("icd_codes", "")
    #               surgery_text_data.get("ops_codes", "")
    #               surgery_text_data.get("description", "")  # parse from text if needed
    # 3. Build hierarchical code representation (same as training)
    # 4. Apply tfidf_transformer.transform (NOT fit)
    # 5. Load codes_hierarchical_embedder.pt → forward → 512d
    # 6. If no codes found: return zeros(1, 512) + present_mask = 0
    # Returns: torch.Tensor shape (1, 512)
```

**`spatial_inference.py`:**
```python
def get_spatial_embedding(primary_wsi: dict, lymph_wsi: dict, resources_path: Path, device) -> torch.Tensor:
    # 1. Extract features and coords from JSON dicts
    # 2. Concatenate PT + LN patches
    # 3. L2-normalize, coord-normalize
    # 4. MiniBatchKMeans if > 2048 patches
    # 5. Load positional_mlp.pt, patch_projector.pt, habmil_model.pt
    # 6. PatchProjector: 1024→512
    # 7. PositionalMLP: coords→512 bias, add to projected features
    # 8. H-ABMIL: cluster→intra-ABMIL→Nyströmformer→pool → 512d
    # 9. If both PT and LN available: learned fusion
    #    If only one: use available embedding, zero for missing
    # Returns: torch.Tensor shape (1, 512)
```

### 5.3 Model Loading — Safe Pattern

```python
# resources/mirage_net_best_avg.pt  ← trained model weights (best fold or ensemble)
model = MIRAGENet(
    d_model=512, n_modalities=6, n_latents=64,
    n_perceiver_layers=4, n_heads=8, dropout=0.0,  # dropout=0 at inference
    use_graph_vae=True, use_kan_heads=True
).to(DEVICE)

state_dict = torch.load(RESOURCE_PATH / "mirage_net_best_avg.pt", map_location=DEVICE)
model.load_state_dict(state_dict, strict=False)  # strict=False always
model.eval()

# Patient graph is NOT available at inference (single patient).
# Pass patient_graph=None → GraphVAEImputer will use mean-field approximation
# (encode without neighbor aggregation, just own embedding).
```

### 5.4 Ensemble Strategy (Optional, for Higher Robustness)

Train produces 5 model checkpoints (one per fold). At inference:
```python
# Load all 5 fold models
fold_models = [load_model(f"fold_{i}_best_model.pt") for i in range(5)]

# Average logits across folds
all_logit_surv, all_logit_rec = [], []
for m in fold_models:
    m.eval()
    with torch.no_grad():
        out = m(emb_stack, quality, present_mask, patient_graph=None)
    all_logit_surv.append(out["logit_surv"])
    all_logit_rec.append(out["logit_rec"])

final_prob_surv = torch.sigmoid(torch.stack(all_logit_surv).mean(0)).item()
final_prob_rec  = torch.sigmoid(torch.stack(all_logit_rec).mean(0)).item()
```

### 5.5 Output Format

```python
PREDICTION_TARGET_SLUG = "5-year-survival"
# OR
PREDICTION_TARGET_SLUG = "2-year-recurrence-after-diagnosis"

if task == "5-year-survival":
    prediction_str = "living" if prob < 0.5 else "deceased"
    output_file = "5-year-survival.json"
else:
    prediction_str = "no recurrence" if prob < 0.5 else "recurrence"
    output_file = "2-year-recurrence.json"

write_json_file(OUTPUT_PATH / output_file, prediction_str)
```

### 5.6 Resources Directory Contents

```
inference/resources/
├── clinical_preproc_objects.joblib
├── clinical_kan_encoder.pt
├── pathological_preproc_objects.joblib
├── pathological_kan_encoder.pt
├── temporal_preproc_objects.joblib
├── temporal_rwkv_encoder.pt
├── text_semantic_preproc_objects.joblib
├── codes_preproc_objects.joblib
├── codes_hierarchical_embedder.pt
├── positional_mlp.pt
├── patch_projector.pt
├── habmil_model.pt
├── mirage_net_fold0_best.pt    ← one per fold (5 total)
├── mirage_net_fold1_best.pt
├── mirage_net_fold2_best.pt
├── mirage_net_fold3_best.pt
├── mirage_net_fold4_best.pt
├── mirage_net_best_avg.pt      ← best single model across folds
└── Bio_ClinicalBERT/           ← offline copy of transformer model
    ├── config.json
    ├── tokenizer_config.json
    ├── vocab.txt
    └── pytorch_model.bin
```

---

## 6. File and Folder Structure After Upgrade

```
project/
├── preprocess/
│   ├── clinical.py          # STEP 1: KAN encoder, Graph-VAE imputer
│   ├── pathological.py      # STEP 2: KAN encoder, shared patient graph
│   ├── temporal.py          # STEP 3: RWKV encoder (BiLSTM fallback)
│   ├── semantic.py          # STEP 4: English only, Bio_ClinicalBERT
│   ├── codes.py             # STEP 5: Hierarchical Code2Vec ICD+OPS
│   └── spatial.py           # STEP 6: H-ABMIL + Nyströmformer
│
├── train.py                 # 5-fold CV, MIRAGENet, Perceiver IO fusion
│                            # Baselines under --run_baselines flag
│                            # Full journal metrics saved to JSON
│
├── inference/
│   ├── inference.py         # Main challenge submission script
│   └── preprocess/
│       ├── clinical_inference.py
│       ├── pathological_inference.py
│       ├── temporal_inference.py
│       ├── semantic_inference.py
│       ├── codes_inference.py       # NEW (was missing from GitHub)
│       ├── spatial_inference.py
│       └── hcat_model.py → mirage_model.py   # RENAME + rewrite
│                                              # QualityGate hidden=64 FIXED
│
├── outputs/
│   ├── clinical_preprocessed.h5
│   ├── clinical_embedding_512.h5
│   ├── pathological_preprocessed.h5
│   ├── pathological_embedding_512.h5
│   ├── temporal_embedding_512.h5
│   ├── text_semantic_embedding_512.h5
│   ├── codes_embedding_512.h5
│   ├── spatial_embedding_512.h5
│   ├── fold_0_best_model.pt → fold_4_best_model.pt
│   ├── mirage_net_best_avg.pt
│   ├── fold_0_roc_curve_surv.npy → fold_4_roc_curve_rec.npy
│   └── training_summary.json
│
└── README.md                # Update after all scripts are finished
```

---

## 7. PATH PLACEHOLDERS — Update These Before Running

Open each preprocessing script and replace:

```python
# ============================================================
# UPDATE THESE PATHS BEFORE RUNNING
# ============================================================

# Kaggle dataset root
DATASET_ROOT = "/kaggle/input/datasets/sudharsananh/hancothon-2025-complete"

# Structured data JSONs
CLINICAL_JSON    = f"{DATASET_ROOT}/StructuredData/StructuredData/clinical_data.json"
PATHOLOGICAL_JSON= f"{DATASET_ROOT}/StructuredData/StructuredData/pathological_data.json"
BLOOD_JSON       = f"{DATASET_ROOT}/StructuredData/StructuredData/blood_data.json"
BLOOD_REF_JSON   = f"{DATASET_ROOT}/StructuredData/StructuredData/blood_data_reference_ranges.json"

# Text data (ENGLISH ONLY)
HISTORIES_DIR    = f"{DATASET_ROOT}/TextData/TextData/histories_english"
REPORTS_DIR      = f"{DATASET_ROOT}/TextData/TextData/reports_english"
SURGERY_DIR      = f"{DATASET_ROOT}/TextData/TextData/surgery_descriptions_english"
ICD_CODES_DIR    = f"{DATASET_ROOT}/TextData/TextData/icd_codes"
OPS_CODES_DIR    = f"{DATASET_ROOT}/TextData/TextData/ops_codes"

# WSI h5 directories
WSI_LYMPH        = f"{DATASET_ROOT}/WSI_UNI_encodings/WSI_LymphNode/h5_files"
WSI_CUP          = f"{DATASET_ROOT}/WSI_UNI_encodings/WSI_PrimaryTumor/WSI_PrimaryTumor_CUP/h5_files"
WSI_HYPO         = f"{DATASET_ROOT}/WSI_UNI_encodings/WSI_PrimaryTumor/WSI_PrimaryTumor_Hypopharynx/h5_files"
WSI_LARYNX       = f"{DATASET_ROOT}/WSI_UNI_encodings/WSI_PrimaryTumor/WSI_PrimaryTumor_Larynx/h5_files"
WSI_ORAL         = f"{DATASET_ROOT}/WSI_UNI_encodings/WSI_PrimaryTumor/WSI_PrimaryTumor_OralCavity/h5_files"
WSI_OROPH1       = f"{DATASET_ROOT}/WSI_UNI_encodings/WSI_PrimaryTumor/WSI_PrimaryTumor_Oropharynx_Part1/h5_files"
WSI_OROPH2       = f"{DATASET_ROOT}/WSI_UNI_encodings/WSI_PrimaryTumor/WSI_PrimaryTumor_Oropharynx_Part2/h5_files"

# Output directory (writable)
OUT_DIR = "/kaggle/working/outputs"
```

---

## 8. README.md Template (Create This After All Scripts Are Finished)

After all scripts are updated and tested, **delete the old README** and create a new one:

---

# MIRAGE-Net: Multimodal Integrative Research Architecture with Graph-Enhanced Networks

**Task:** 5-year survival and 2-year recurrence prediction for head-and-neck cancer  
**Dataset:** HANCOCK (763 patients, 6 modalities)  
**Challenge:** Hancothon 2025 (MICCAI 2025)

## Architecture Summary

| Module | Method | Novel Contribution |
|---|---|---|
| Tabular encoder | KAN (Kolmogorov-Arnold Network) | Spline-based activations, interpretable |
| Temporal encoder | RWKV Linear Recurrence | O(L) complexity, no vanishing gradient |
| WSI aggregator | H-ABMIL + Nyströmformer | Two-level hierarchical attention |
| Cross-modal fusion | **Perceiver IO** | 64-latent cross-attention to 4 modality groups |
| Imputation | **Graph-VAE** | GATv2 message passing as imputation prior |
| Code embedding | Hierarchical Code2Vec | ICD/OPS hierarchy-aware embedding |
| Prediction heads | KAN | Interpretable spline functions |

## Results (5-fold CV)

| Metric | Mean | Std |
|---|---|---|
| 5yr Survival F1 | — | — |
| 5yr Survival AUC | — | — |
| 2yr Recurrence F1 | — | — |
| 2yr Recurrence AUC | — | — |
| Average F1 | — | — |

*(Fill in after training)*

## Requirements

```bash
# Core
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install h5py scikit-learn pandas numpy joblib transformers tqdm

# Architecture components
pip install pykan            # KAN — Kolmogorov-Arnold Networks
pip install torch-geometric  # GATv2 for Graph-VAE
pip install nystrom-attention # Nyströmformer for WSI

# Optional (for Bio_ClinicalBERT — download before running offline)
# Download to inference/resources/Bio_ClinicalBERT/
# huggingface-cli download emilyalsentzer/Bio_ClinicalBERT --local-dir inference/resources/Bio_ClinicalBERT
```

**⚠️ Update all PATH_ variables at the top of every preprocess script before running.**

## Step 1: Update Paths in All Scripts

Open each file below and update the `PATH_` variables at the top:
- `preprocess/clinical.py`
- `preprocess/pathological.py`
- `preprocess/temporal.py`
- `preprocess/semantic.py`
- `preprocess/codes.py`
- `preprocess/spatial.py`

See Section 7 of `PLAN_UPGRADE.md` for the exact path values.

## Step 2: Run Preprocessing (in order — each depends on the previous)

```bash
# 1. Clinical tabular (produces labels used by all other scripts)
python preprocess/clinical.py

# 2. Pathological tabular (shares patient graph with clinical)
python preprocess/pathological.py

# 3. Blood temporal
python preprocess/temporal.py --mode rwkv --epochs 50
# If RWKV implementation not ready: python preprocess/temporal.py --mode lstm --epochs 50

# 4. Text semantic (English only — DO NOT use German folders)
python preprocess/semantic.py \
  --histories /kaggle/input/datasets/sudharsananh/hancothon-2025-complete/TextData/TextData/histories_english \
  --reports /kaggle/input/datasets/sudharsananh/hancothon-2025-complete/TextData/TextData/reports_english \
  --surgery /kaggle/input/datasets/sudharsananh/hancothon-2025-complete/TextData/TextData/surgery_descriptions_english \
  --outdir /kaggle/working/outputs \
  --use_transformer  # remove this flag if no GPU or transformers not installed

# 5. ICD + OPS codes (new modality)
python preprocess/codes.py

# 6. WSI spatial (longest step — ~1-2 hours on GPU)
python preprocess/spatial.py \
  --wsi_lymph /kaggle/input/datasets/sudharsananh/hancothon-2025-complete/WSI_UNI_encodings/WSI_LymphNode/h5_files \
  --wsi_cup /kaggle/input/datasets/sudharsananh/hancothon-2025-complete/WSI_UNI_encodings/WSI_PrimaryTumor/WSI_PrimaryTumor_CUP/h5_files \
  --wsi_hypo /kaggle/input/datasets/sudharsananh/hancothon-2025-complete/WSI_UNI_encodings/WSI_PrimaryTumor/WSI_PrimaryTumor_Hypopharynx/h5_files \
  --wsi_larynx /kaggle/input/datasets/sudharsananh/hancothon-2025-complete/WSI_UNI_encodings/WSI_PrimaryTumor/WSI_PrimaryTumor_Larynx/h5_files \
  --wsi_oral /kaggle/input/datasets/sudharsananh/hancothon-2025-complete/WSI_UNI_encodings/WSI_PrimaryTumor/WSI_PrimaryTumor_OralCavity/h5_files \
  --wsi_oroph1 /kaggle/input/datasets/sudharsananh/hancothon-2025-complete/WSI_UNI_encodings/WSI_PrimaryTumor/WSI_PrimaryTumor_Oropharynx_Part1/h5_files \
  --wsi_oroph2 /kaggle/input/datasets/sudharsananh/hancothon-2025-complete/WSI_UNI_encodings/WSI_PrimaryTumor/WSI_PrimaryTumor_Oropharynx_Part2/h5_files \
  --outdir /kaggle/working/outputs
```

## Step 3: Train with 5-Fold Cross-Validation

```bash
python train.py \
  --clinical /kaggle/working/outputs/clinical_embedding_512.h5 \
  --pathological /kaggle/working/outputs/pathological_embedding_512.h5 \
  --temporal /kaggle/working/outputs/temporal_embedding_512.h5 \
  --semantic /kaggle/working/outputs/text_semantic_embedding_512.h5 \
  --codes /kaggle/working/outputs/codes_embedding_512.h5 \
  --spatial /kaggle/working/outputs/spatial_embedding_512.h5 \
  --outdir /kaggle/working/checkpoints \
  --epochs 60 \
  --batch 32 \
  --lr 5e-5 \
  --n_folds 5 \
  --n_latents 64 \
  --n_perceiver_layers 4 \
  --use_focal_loss \
  --use_contrastive \
  --use_graph_vae \
  --use_kan_heads \
  --run_baselines \
  --device cuda
```

## Step 4: Check Results

```bash
cat /kaggle/working/checkpoints/training_summary.json
# Look for: aggregate.avg_f1_mean ± avg_f1_std
# Compare against: baselines.github_hcat_original
```

## Step 5: Inference (Challenge Submission)

```bash
cd inference/
python inference.py
# Change PREDICTION_TARGET_SLUG at top of inference.py to switch tasks:
#   "5-year-survival" or "2-year-recurrence-after-diagnosis"
```

## Citation

*(Fill in once paper is submitted)*

---

## 9. Checklist Before Submission

**Preprocessing:**
- [ ] All PATH_ variables updated in every preprocess script
- [ ] Only `*_english` text folders used — NEVER `histories/` or `reports/` (German)
- [ ] Clinical label derivation uses correct day thresholds (1825d, 730d)
- [ ] Pathological `closest_resection_margin_in_cm` cleaned (`<0.1` → `0.05`)
- [ ] Blood temporal uses 16-bin grid over 0–14 days (NOT 0–30 days)
- [ ] Multi-h5 WSI patients: patches concatenated BEFORE encoding
- [ ] ICD/OPS regex extracts codes correctly: `\[([A-Z]\d{2}[\.\d]*)` for ICD
- [ ] All 7 WSI directories scanned (including Oropharynx_Part1 AND Part2)

**Model:**
- [ ] QualityGate hidden=64 in BOTH `train.py` AND `inference/preprocess/mirage_model.py`
- [ ] 6 modalities (not 5 — ICD+OPS codes is modality index 4)
- [ ] Modality order is identical in train and inference: `[clinical, patho, temporal, semantic, codes, spatial]`
- [ ] Perceiver IO latent array size matches between train and inference (n_latents=64)
- [ ] `strict=False` in ALL `load_state_dict` calls in inference scripts

**Training:**
- [ ] 5-fold CV implemented with StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
- [ ] Stratification accounts for censored (-1) labels
- [ ] Focal loss with gamma=2.0 used (not plain BCE)
- [ ] Label smoothing eps=0.05 applied
- [ ] `pos_weight` computed per fold from training split (not hardcoded)
- [ ] Modality dropout never drops index 0 (clinical)
- [ ] `training_summary.json` includes all fields from Section 4.8
- [ ] Baseline models run and recorded under `--run_baselines`
- [ ] All 5 fold model checkpoints saved + best_avg.pt

**Inference:**
- [ ] `codes_inference.py` written and integrated into `inference.py`
- [ ] Patient graph = None at inference (single patient, use mean-field VAE)
- [ ] Bio_ClinicalBERT downloaded offline to `resources/Bio_ClinicalBERT/`
- [ ] PREDICTION_TARGET_SLUG set correctly before building Docker container
- [ ] Output filenames match challenge spec: `"5-year-survival.json"` or `"2-year-recurrence.json"`

**Paper readiness:**
- [ ] `training_summary.json` has mean ± std for all metrics across 5 folds
- [ ] Ablation results saved (6 single-modality LR baselines + concat-MLP + ABMIL)
- [ ] Calibration curves saved per fold
- [ ] ROC curves saved per fold
- [ ] Confusion matrices saved per fold

---

## 10. Priority Order for Implementation

Work in this exact order. Do not skip ahead.

**Priority 1 — Fixes (30 minutes):**
1. Fix path bug everywhere: `StructuredData/StructuredData/` (double nesting)
2. Fix QualityGate hidden=64 in both train.py and hcat_model.py

**Priority 2 — New modality (2 hours, highest ROI):**
3. Write `preprocess/codes.py` — ICD+OPS hierarchical Code2Vec embedding
4. Write `inference/preprocess/codes_inference.py`
5. Add codes modality to `train.py` data loading (index 4)

**Priority 3 — Preprocessing upgrades (4 hours each):**
6. Upgrade `clinical.py`: Graph-VAE imputer + KAN encoder
7. Upgrade `pathological.py`: same pipeline as clinical
8. Fix `spatial.py`: multi-h5 patch concatenation + H-ABMIL + Nyströmformer

**Priority 4 — Core architecture (1 day):**
9. Write `MIRAGENet` class in `train.py` with Perceiver IO fusion
10. Write `GraphVAEImputer` with GATv2 (or k-NN fallback)

**Priority 5 — Training infrastructure (4 hours):**
11. Wrap training in 5-fold CV with StratifiedKFold
12. Add baselines under `--run_baselines`
13. Add full metrics to `training_summary.json`

**Priority 6 — Temporal encoder (2 hours):**
14. Write `RWKVBlock` + `RWKVTemporalEncoder` in `temporal.py`
15. Update `temporal_inference.py` to load RWKV weights

**Priority 7 — Polish (2 hours):**
16. Update README with correct run commands
17. Verify inference pipeline end-to-end with a test patient

---

*End of Plan — MIRAGE-Net for HANCOCK 2025*
