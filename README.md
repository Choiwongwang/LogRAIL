# LogRAIL: Retrieval-Augmented LLM Inference Layer for Log Anomaly Detection

This repository provides a two-stage log anomaly detection pipeline:
1) Stage-1 (LogFormer) scores all windows.
2) Stage-2 re-verifies near-threshold windows via VDB + LLM in two modes:
   - Precision-oriented (Normal VDB, bottom-k)
   - Recall-oriented (Anomaly VDB, top-k)

The steps below match the current codebase and defaults.

![Figure 2: LogRAIL two-stage pipeline](assets/figure2.png)

## Pipeline Overview (Box Diagram)
```text
[Raw Logs / Labeled CSV]
          |
          v
[Preprocess: windowing + embeddings]
          |
          v
[Stage-1 LogFormer]
  |                |
  | (all windows)  |  -> scores + pred/prob
  v                v
[Window templates] [Stage-1 preds]
          |                |
          +--------+-------+
                   v
        [Stage-2 Candidate Windows]
          |                   |
          | Precision path    | Recall path
          v                   v
[Normal VDB bottom-k]   [Anomaly VDB top-k]
          |                   |
          +--------+----------+
                   v
              [LLM decision]
                   |
                   v
             [Final anomaly]
```

## Project Structure (Core)
- `dataset/Android_new_full.csv` : Labeled log dataset (public)
- `LogFormer/preprocess/preprocess.py` : Window preprocessing -> NPZ
- `LogFormer/train_transformer.py` : Stage-1 training
- `LogFormer/sweep_threshold.py` : Threshold sweep (optional)
- `LogFormer/infer_logformer.py` : Stage-1 inference
- `rebuild_window_repr_from_split.py` : Window_id -> templates mapping
- `make_gt_csv.py` : GT CSV from NPZ (test/all)
- `build_train_vdb_templates.py` : Train-only template corpora (normal/anomaly)
- `clean_normal_templates.py` / `clean_anomaly_templates.py` : Simple corpus builders
- `postprocess/RAG_Normal.py` : Stage-2 precision mode (Normal VDB)
- `postprocess/RAG_Abnormal.py` : Stage-2 recall mode (Anomaly VDB)

## Environment
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## End-to-End Pipeline (Recommended)
### 1) Preprocess -> NPZ
```bash
python LogFormer/preprocess/preprocess.py ^
  --csv dataset/Android_new_full.csv ^
  --out LogFormer/npz
```

### 2) Train Stage-1 (LogFormer)
```bash
python LogFormer/train_transformer.py
```
Outputs: `LogFormer/checkpoints/best_*.pt`

### 3) (Optional) Sweep Threshold
```bash
python LogFormer/sweep_threshold.py --ckpt LogFormer/checkpoints/best_*.pt --split test
```

### 4) Stage-1 Inference
```bash
python LogFormer/infer_logformer.py ^
  --ckpt LogFormer/checkpoints/best_*.pt ^
  --out output/logformer_preds_test.csv
```

### 5) Rebuild Window Templates (align with window_id)
```bash
python rebuild_window_repr_from_split.py ^
  --src_csv dataset/Android_new_full.csv ^
  --window_size 20 --train_ratio 0.7 --val_ratio 0.15 --test_ratio 0.15 --seed 42 ^
  --format json --out output/window_repr_by_pred_info_norm.csv
```
Note: Use a separate file for anomaly mode if desired (e.g., `..._anom.csv`).

### 6) Build VDB Template Corpora (Train-only)
```bash
python build_train_vdb_templates.py ^
  --src_csv dataset/Android_new_full.csv ^
  --out_normal dataset/normal_templates_clean.csv ^
  --out_anomaly dataset/anomaly_templates_clean.csv
```

### 7) Build GT (test/all)
```bash
python make_gt_csv.py --npz_dir LogFormer/npz
```
Outputs: `output/gt_test.csv`, `output/gt_all.csv`

### 8) Stage-2 Re-verification
Precision-oriented (Normal VDB, bottom-k):
```bash
python postprocess/RAG_Normal.py
```
Recall-oriented (Anomaly VDB, top-k):
```bash
python postprocess/RAG_Abnormal.py
```
Both scripts perform evaluation by default using `output/gt_test.csv`.

## Notes
- Stage-2 scripts already print final metrics. A separate evaluator is optional.
- Ensure `window_repr_by_pred_info_*.csv` and `logformer_preds_*.csv` use the same window_id order.
- If you want validation-only ablations, see `run_stage2_ablation.py` and `summarize_stage2_ablation.py`.
