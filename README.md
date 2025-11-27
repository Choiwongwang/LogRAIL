LogRAIL: A Retrieval-Augmented LLM Inference Layer for Log Anomaly Detection

```
┌───────────────────────────────┐
│ [1] Preprocess (preprocess.py) │
└───────────────┬───────────────┘
                │ (create npz)
                v
┌───────────────────────────────┐
│ [2] LogFormer train/infer      │
│     (train*/infer_logformer)   │
└───────────────┬───────────────┘
                │
                v
┌───────────────────────────────┐
│ [3] Rebuild window reps        │
│     (rebuild_window_repr_*)    │
└───────────────┬───────────────┘
                │ (window_id ↔ EventTemplate)
      ┌─────────┴───────────┐
      v                     v
┌───────────────────┐  ┌───────────────────┐
│ [4] RAG_Normal    │  │ [4′] RAG_Abnormal │
│ (precision, normal│  │ (recall, anomaly  │
│  VDB)             │  │  VDB)             │
└─────────┬─────────┘  └─────────┬─────────┘
          │(parallel/optional)   │
          └─────────┬────────────┘
                    v
        ┌────────────────────────┐
        │ [5] Eval (eval_final)   │
        └────────────────────────┘
```

## Key files/paths
- `dataset/Android.csv`              : Labeled raw logs (public data)
- `dataset/normal_templates_clean_.csv` / `dataset/anomaly_templates_clean.csv` : Normal/anomaly template corpora
- `LogFormer/preprocess/preprocess.py` : Window preprocessing (creates npz)
- `LogFormer/infer_logformer.py`       : Stage‑1 (LogFormer) inference
- `rebuild_window_repr_from_split.py`  : Rebuild window_id ↔ template mapping
- `postprocess/RAG_Normal.py`          : RAG precision phase (normal VDB, default top‑k=3)
- `postprocess/RAG_Abnormal.py`        : RAG recall phase (anomaly VDB, optional)
- `eval_final.py`                      : Final metric evaluation

## Environment (core packages)
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` highlights:
- PyTorch 1.13.0 (`torch`, `torchvision`)
- Transformers 4.37.2, langchain 0.2.x, langchain_community/openai
- pandas 2.0.x, numpy 1.24.x, scikit-learn 1.3.x
- matplotlib 3.7.x, tqdm
- TensorFlow 2.17.0 is unused; you can remove it if desired.

## How to run (example in backup path)
```bash
# 1) Preprocess → NPZ
python LogFormer/preprocess/preprocess.py \
  --csv dataset/Android.csv \
  --out LogFormer/npz

# 2) LogFormer inference (ckpt defaults embedded)
python LogFormer/infer_logformer.py

# 3) Rebuild window → template mapping
python rebuild_window_repr_from_split.py

# 4) RAG precision (normal VDB, default top‑k=3)
python postprocess/RAG_Normal.py

# 5) RAG recall (optional, anomaly VDB)
python postprocess/RAG_Abnormal.py   # run only if needed

# 6) Evaluation (GT/pred window_id must match)
python eval_final.py \
  --pred_csv output/rag_result.csv \
  --gt_csv output/gt_all.csv \
  --export_report output/metrics_final_only.json
```

## Data / artifacts
- Only raw data under `dataset/` is kept in the repo.  
- Artifacts (`output/`, `rag_db/`, `.venv/`, npz/checkpoints) are ignored via `.gitignore`; generate them with the steps above.

## Notes
- When regenerating GT, ensure `gt_all.csv` matches the prediction window_id set (2,309 windows).
- If your data is not public, exclude `dataset/` from commits and share only the run instructions.
