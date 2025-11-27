import os, argparse, json, random
import pandas as pd, numpy as np
from tqdm import tqdm
import torch
from sentence_transformers import SentenceTransformer

# ------------------------- constants -------------------------
WINDOW_SIZE   = 20        # 20 log lines = 1 window
TRAIN_RATIO   = 0.7       # Train 70% / Test 30%
SEED          = 42
EMBED_MODEL   = "distilbert-base-nli-mean-tokens"   # 768‑D
# -----------------------------------------------------------

def normalize_tpl(s: str) -> str:
    t = str(s).strip()
    return " ".join(t.split())

def make_window_label(labels):
    return np.array([0,1], np.int64) if (labels == 1).any() else np.array([1,0], np.int64)

def encode_windows(df, tpl2vec):
    num_win = len(df) // WINDOW_SIZE
    df = df.iloc[: num_win * WINDOW_SIZE].reset_index(drop=True)

    X_list, Y_list = [], []
    for i in tqdm(range(num_win), desc="Building windows", ncols=80):
        block = df.iloc[i*WINDOW_SIZE:(i+1)*WINDOW_SIZE]

        vecs = np.stack([tpl2vec[t] for t in block["EventTemplate_norm"]], axis=0)  # (20,768)
        X_list.append(vecs)
        Y_list.append(make_window_label(block["label"].values))

    return np.array(X_list, object), np.array(Y_list, object)

def save_npz(path, x, y):
    np.savez(path, x=np.array(x, dtype=object), y=np.array(y, dtype=object))
    print(f"[✓] {os.path.basename(path):<25}  x:{len(x):>5}  y:{len(y):>5}")

def main(args):
    os.makedirs(args.out, exist_ok=True)

    # 1. Load CSV
    df = pd.read_csv(args.csv)
    df["EventTemplate_norm"] = df["EventTemplate"].apply(normalize_tpl)
    df["label"] = df["label"].astype(int)
    print(f"[INFO] Loaded {len(df):,} rows  (normal={sum(df.label==0):,}, anomaly={sum(df.label==1):,})")

    # 2. Sentence‑Transformer embeddings (unique templates)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model  = SentenceTransformer(EMBED_MODEL, device=device)
    uniq_tpls = df["EventTemplate_norm"].unique().tolist()
    print(f"[STEP] Encoding {len(uniq_tpls):,} unique templates → {EMBED_MODEL} ({device})")
    tpl_vecs  = model.encode(uniq_tpls, batch_size=256, show_progress_bar=True)
    tpl2vec   = {t:v for t,v in zip(uniq_tpls, tpl_vecs)}

    # 3. Encode per-window
    X, Y = encode_windows(df, tpl2vec)
    print(f"[INFO] Total windows built: {len(X):,}")

    # 4. Train/Test split (7:3)
    idx = list(range(len(X)))
    random.seed(SEED); random.shuffle(idx)
    n_train = int(len(idx) * TRAIN_RATIO)
    idx_train, idx_test = idx[:n_train], idx[n_train:]

    X_train, Y_train = X[idx_train], Y[idx_train]
    X_test,  Y_test  = X[idx_test],  Y[idx_test]

    # 5. Save
    save_npz(os.path.join(args.out, "train.npz"), X_train, Y_train)
    save_npz(os.path.join(args.out, "test.npz"),  X_test,  Y_test)

    # 6. Meta info
    meta = {
        "window_size": WINDOW_SIZE,
        "embedding_model": EMBED_MODEL,
        "train_windows": len(X_train),
        "test_windows":  len(X_test),
        "seed": SEED
    }
    with open(os.path.join(args.out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[✓] saved meta.json")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Path to Android.csv")
    ap.add_argument("--out", required=True, help="Directory to save npz files")
    args = ap.parse_args()
    main(args)
