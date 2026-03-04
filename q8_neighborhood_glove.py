import argparse
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


def l2_normalize(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.maximum(n, eps)


def apply_transform(X: np.ndarray, transform: str) -> np.ndarray:
    if transform == "raw":
        return X
    if transform == "mean_center_l2":
        Xc = X - X.mean(axis=0, keepdims=True)
        return l2_normalize(Xc)
    raise ValueError(f"Unknown transform: {transform}")


def build_rel_maps_from_pairs(pairs: List[Tuple[str, str]]) -> Dict[str, set]:
    m: Dict[str, set] = {}
    for a, b in pairs:
        a = a.lower()
        b = b.lower()
        m.setdefault(a, set()).add(b)
        m.setdefault(b, set()).add(a)
    return m


def load_glove_vectors_filtered(glove_path: Path, needed_words: set[str]) -> Dict[str, np.ndarray]:
    vecs: Dict[str, np.ndarray] = {}
    needed_words = {w.lower() for w in needed_words}

    with open(glove_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.rstrip().split(" ")
            if len(parts) < 10:
                continue
            w = parts[0].lower()
            if w not in needed_words:
                continue
            try:
                v = np.array([float(x) for x in parts[1:]], dtype=np.float32)
            except ValueError:
                continue
            vecs[w] = v
    return vecs


def neighborhood_eval_topk(
    word_vecs: Dict[str, np.ndarray],
    syn_map: Dict[str, set],
    ant_map: Dict[str, set],
    transform: str,
    k: int = 10,
    n_targets: int = 50,
    seed: int = 42,
) -> dict:
    rng = random.Random(seed)

    vocab = list(word_vecs.keys())
    candidates = [
        w for w in vocab
        if (w in syn_map and len(syn_map[w]) > 0) and (w in ant_map and len(ant_map[w]) > 0)
    ]
    rng.shuffle(candidates)
    candidates = candidates[: min(n_targets, len(candidates))]

    if not candidates:
        return {"transform": transform, "k": k, "n_vocab": len(vocab), "n_targets": 0, "n_eligible": 0, "n_hits": 0, "rate": float("nan")}

    keys = list(dict.fromkeys(candidates + vocab))
    X = np.stack([word_vecs[w] for w in keys], axis=0)
    X = apply_transform(X, transform)
    Xn = l2_normalize(X)

    idx = {w: i for i, w in enumerate(keys)}

    eligible = 0
    hits = 0

    for t in candidates:
        ti = idx[t]
        sims = Xn @ Xn[ti]
        sims[ti] = -1.0
        nn_idx = np.argpartition(-sims, k)[:k]
        nn_idx = nn_idx[np.argsort(-sims[nn_idx])]
        nn_words = [keys[i] for i in nn_idx]

        # condition: at least one synonym exists in vocab
        if not any((s in idx) for s in syn_map.get(t, set())):
            continue

        eligible += 1
        if any((a in nn_words) for a in ant_map.get(t, set())):
            hits += 1

    rate = hits / eligible if eligible else float("nan")
    return {"transform": transform, "k": k, "n_vocab": len(vocab), "n_targets": len(candidates), "n_eligible": eligible, "n_hits": hits, "rate": rate}


def main(glove: Path, syn_csv: Path, ant_csv: Path, k: int, n_targets: int, out_csv: Path):
    syn_df = pd.read_csv(syn_csv)
    ant_df = pd.read_csv(ant_csv)

    syn_pairs = [(a.lower(), b.lower()) for a, b in zip(syn_df["w1"], syn_df["w2"])]
    ant_pairs = [(a.lower(), b.lower()) for a, b in zip(ant_df["w1"], ant_df["w2"])]

    syn_map = build_rel_maps_from_pairs(syn_pairs)
    ant_map = build_rel_maps_from_pairs(ant_pairs)

    wn_vocab = sorted(
        set(syn_df["w1"].str.lower()) | set(syn_df["w2"].str.lower()) |
        set(ant_df["w1"].str.lower()) | set(ant_df["w2"].str.lower())
    )

    glove_vecs = load_glove_vectors_filtered(glove, set(wn_vocab))

    rows = []
    for tr in ["raw", "mean_center_l2"]:
        rows.append(neighborhood_eval_topk(glove_vecs, syn_map, ant_map, tr, k=k, n_targets=n_targets, seed=42))

    out = pd.DataFrame(rows)
    out.to_csv(out_csv, index=False)

    print("\nQ8 Neighbourhood evaluation (GloVe)")
    print(out)
    print(f"\nSaved: {out_csv}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--glove", required=True, help="Path to glove .txt (e.g. glove.6B.100d.txt)")
    ap.add_argument("--syn", default="outputs/tables/wordnet_syn_pairs.csv")
    ap.add_argument("--ant", default="outputs/tables/wordnet_ant_pairs.csv")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--n_targets", type=int, default=50)
    ap.add_argument("--out", default="outputs/tables/q8_neighborhood_glove.csv")
    args = ap.parse_args()

    main(Path(args.glove), Path(args.syn), Path(args.ant), args.k, args.n_targets, Path(args.out))