from __future__ import annotations

import re
import json
import math
import random
import pickle
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score


# -----------------------------
# CONFIG
# -----------------------------

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "outputs"
CACHE_DIR = DATA_DIR / "cache"
PLOTS_DIR = OUT_DIR / "plots"
TABLES_DIR = OUT_DIR / "tables"

for d in [DATA_DIR, OUT_DIR, CACHE_DIR, PLOTS_DIR, TABLES_DIR]:
    d.mkdir(parents=True, exist_ok=True)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# Lab constraints
MIN_LINE_TOKENS = 5
MAX_SENT_TOKENS = 40
MIN_CTX = 10
MAX_CTX = 50

# You choose this; justify in report
MIN_WORD_FREQ = 10

# Debug speed control (set None for full corpus)
DEFAULT_CORPUS_CAP = None  # e.g. 400_000 for debugging


# -----------------------------
# UTILS
# -----------------------------

def simple_tokenize(text: str) -> List[str]:
    return re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?|[0-9]+", text.lower())


def save_pickle(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)


def cosine(u: np.ndarray, v: np.ndarray, eps: float = 1e-12) -> float:
    nu = np.linalg.norm(u)
    nv = np.linalg.norm(v)
    if nu < eps or nv < eps:
        return float("nan")
    return float(np.dot(u, v) / (nu * nv))


# -----------------------------
# 1) CORPUS: WikiText-103 train
# -----------------------------

def load_wikitext103_train() -> List[str]:
    """
    Requires:
      pip install datasets
    """
    from datasets import load_dataset  # type: ignore
    ds = load_dataset("wikitext", "wikitext-103-v1", split="train")
    return [x for x in ds["text"]]


def clean_wikitext_lines(lines: List[str]) -> List[str]:
    cleaned = []
    for ln in lines:
        s = (ln or "").strip()
        if not s:
            continue
        if s.startswith("="):
            continue
        if len(simple_tokenize(s)) < MIN_LINE_TOKENS:
            continue
        cleaned.append(s)
    return cleaned


def sentencize_regex(lines: List[str]) -> List[str]:
    """
    spaCy-free sentence splitting:
    split on sentence end punctuation followed by whitespace.
    Enforce MAX_SENT_TOKENS and dedupe.
    """
    splitter = re.compile(r"(?<=[\.\!\?\;\:])\s+")
    sents: List[str] = []

    for text in lines:
        text = text.strip()
        if not text:
            continue
        text = re.sub(r"\s+", " ", text)
        parts = splitter.split(text)
        for p in parts:
            t = p.strip()
            if not t:
                continue
            toks = simple_tokenize(t)
            if len(toks) > MAX_SENT_TOKENS:
                continue
            sents.append(t)

    # dedupe preserving order
    sents = list(dict.fromkeys(sents))
    return sents


def build_corpus_cache(force: bool = False) -> List[str]:
    cache_path = CACHE_DIR / "wikitext103_train_sentences.pkl"
    if cache_path.exists() and not force:
        return load_pickle(cache_path)

    lines = load_wikitext103_train()
    lines = clean_wikitext_lines(lines)
    sents = sentencize_regex(lines)

    save_pickle(sents, cache_path)
    return sents


# -----------------------------
# 1B) Morphological families
# -----------------------------

def load_morph_families(tsv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(tsv_path, sep="\t", header=None)
    df.columns = ["lemma", "forms", "family_type", "transform_type"][: df.shape[1]]
    return df


def explode_family_forms(df: pd.DataFrame) -> pd.DataFrame:
    out_rows = []
    for _, row in df.iterrows():
        lemma = str(row["lemma"]).strip()
        forms = str(row["forms"]).strip()
        family_type = row.get("family_type", "")
        transform_type = row.get("transform_type", "")

        parts = re.split(r"[,\s]+", forms)
        parts = [p for p in parts if p]

        all_words = [lemma] + parts
        for w in dict.fromkeys([x.lower() for x in all_words]).keys():
            out_rows.append(
                {
                    "lemma": lemma.lower(),
                    "word": w,
                    "family_type": family_type,
                    "transform_type": transform_type,
                }
            )
    return pd.DataFrame(out_rows)


# -----------------------------
# 1C) Frequencies + contexts
# -----------------------------

def word_frequencies(sentences: List[str]) -> Dict[str, int]:
    freqs: Dict[str, int] = {}
    for s in sentences:
        for w in simple_tokenize(s):
            freqs[w] = freqs.get(w, 0) + 1
    return freqs


def _stable_seed(word: str, base_seed: int = SEED) -> int:
    h = hashlib.md5(word.encode("utf-8")).hexdigest()
    return base_seed + (int(h[:8], 16) % 1_000_000_000)


def collect_contexts_single_pass(
    sentences: List[str],
    target_words: List[str],
    min_ctx: int = MIN_CTX,
    max_ctx: int = MAX_CTX,
    base_seed: int = SEED,
    progress_every: int = 200_000,
) -> Tuple[Dict[str, List[str]], Dict[str, Dict[str, object]]]:
    """
    One-pass context collection with reservoir sampling up to max_ctx=50 per word.
    - counts total contexts per word
    - if total >= 50: keeps a uniform sample of 50
    - if 10..49: keeps all
    - if <10: excluded
    """
    target_set = set(w.lower() for w in target_words)

    counts = {w: 0 for w in target_set}
    samples: Dict[str, List[str]] = {w: [] for w in target_set}
    rngs = {w: random.Random(_stable_seed(w, base_seed)) for w in target_set}

    total = len(sentences)
    for i, s in enumerate(sentences, start=1):
        if progress_every and i % progress_every == 0:
            print(f"  processed {i:,}/{total:,} sentences...")

        toks = simple_tokenize(s)
        hits = target_set.intersection(toks)
        if not hits:
            continue

        for w in hits:
            counts[w] += 1
            c = counts[w]
            lst = samples[w]

            if len(lst) < max_ctx:
                lst.append(s)
            else:
                j = rngs[w].randint(0, c - 1)
                if j < max_ctx:
                    lst[j] = s

    word_to_contexts: Dict[str, List[str]] = {}
    word_status: Dict[str, Dict[str, object]] = {}

    for w in sorted(target_set):
        n = counts[w]
        if n >= max_ctx:
            word_to_contexts[w] = samples[w]
            status = "ok_50_sampled"
        elif min_ctx <= n < max_ctx:
            word_to_contexts[w] = samples[w]
            status = "ok_lowfreq"
        else:
            status = "excluded_lt10"

        word_status[w] = {"n_ctx": int(n), "status": status}

    return word_to_contexts, word_status


# -----------------------------
# 1D) GloVe loader (filtered)
# -----------------------------

def load_glove_vectors_filtered(glove_path: Path, needed_words: set[str]) -> Dict[str, np.ndarray]:
    """
    Load ONLY vectors whose word is in needed_words.
    Keeps RAM and time low.
    """
    vecs: Dict[str, np.ndarray] = {}
    needed_words = {w.lower() for w in needed_words}

    with open(glove_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.rstrip().split(" ")
            if len(parts) < 10:
                continue
            word = parts[0].lower()
            if word not in needed_words:
                continue
            try:
                v = np.array([float(x) for x in parts[1:]], dtype=np.float32)
            except ValueError:
                continue
            vecs[word] = v

    return vecs


# -----------------------------
# 1E) WordNet synonym/antonym pairs
# -----------------------------

def build_wordnet_pairs() -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
    """
    Requires:
      pip install nltk
      python -c "import nltk; nltk.download('wordnet'); nltk.download('omw-1.4')"
    """
    from nltk.corpus import wordnet as wn  # type: ignore

    syn_pairs = set()
    ant_pairs = set()

    for syn in wn.all_synsets():
        lemmas = [l.name().replace("_", " ").lower() for l in syn.lemmas()]
        lemmas = [x for x in lemmas if " " not in x]
        for i in range(len(lemmas)):
            for j in range(i + 1, len(lemmas)):
                a, b = lemmas[i], lemmas[j]
                if a != b:
                    syn_pairs.add(tuple(sorted((a, b))))

    for syn in wn.all_synsets():
        for l in syn.lemmas():
            a = l.name().replace("_", " ").lower()
            if " " in a:
                continue
            for ant in l.antonyms():
                b = ant.name().replace("_", " ").lower()
                if " " in b:
                    continue
                if a != b:
                    ant_pairs.add(tuple(sorted((a, b))))

    return sorted(syn_pairs), sorted(ant_pairs)


def filter_pairs(
    pairs: List[Tuple[str, str]],
    glove_vocab: set[str],
    freqs: Dict[str, int],
    min_freq: int,
    target_n: int = 200,
) -> List[Tuple[str, str]]:
    ok = [
        (a, b)
        for (a, b) in pairs
        if a in glove_vocab
        and b in glove_vocab
        and freqs.get(a, 0) >= min_freq
        and freqs.get(b, 0) >= min_freq
    ]
    rng = random.Random(SEED)
    rng.shuffle(ok)
    return ok[:target_n]


# -----------------------------
# 2) BERT embedding extraction
# -----------------------------

@dataclass(frozen=True)
class BertConfig:
    model_name: str = "bert-base-uncased"
    layers: Tuple[int, ...] = (1, 6, 12)
    device: str = "cpu"


def load_bert(cfg: BertConfig):
    """
    Requires:
      pip install torch transformers
    """
    from transformers import AutoTokenizer, AutoModel  # type: ignore

    tok = AutoTokenizer.from_pretrained(cfg.model_name, use_fast=True)
    model = AutoModel.from_pretrained(cfg.model_name, output_hidden_states=True)
    model.eval()
    model.to(cfg.device)
    return tok, model


def bert_word_embedding(sentence: str, target_word: str, layer: int, tok, model) -> Optional[np.ndarray]:
    """
    Mean-pool subword pieces for the target word.
    Picks first occurrence in the sentence.
    """
    import torch  # type: ignore

    word = target_word.lower()
    if word not in simple_tokenize(sentence):
        return None

    enc = tok(sentence, return_tensors="pt")
    enc = {k: v.to(model.device) for k, v in enc.items()}

    with torch.no_grad():
        out = model(**enc)
        hs = out.hidden_states  # (embeddings, layer1..layer12)
        h = hs[layer][0]        # (seq_len, hidden)

    tokens = tok.convert_ids_to_tokens(enc["input_ids"][0].tolist())
    target_pieces = tok.tokenize(word)
    if not target_pieces:
        return None

    start = None
    for i in range(len(tokens) - len(target_pieces) + 1):
        if tokens[i : i + len(target_pieces)] == target_pieces:
            start = i
            break

    if start is None:
        stripped = [t.replace("##", "") for t in tokens]
        tgt_stripped = [t.replace("##", "") for t in target_pieces]
        for i in range(len(stripped) - len(tgt_stripped) + 1):
            if stripped[i : i + len(tgt_stripped)] == tgt_stripped:
                start = i
                break

    if start is None:
        return None

    idxs = list(range(start, start + len(target_pieces)))
    vec = h[idxs].mean(dim=0).detach().cpu().numpy().astype(np.float32)
    return vec


def cached_bert_embedding(sentence: str, word: str, layer: int, tok, model, cache: Dict) -> Optional[np.ndarray]:
    key = (sentence, word.lower(), int(layer))
    if key in cache:
        return cache[key]
    v = bert_word_embedding(sentence, word, layer, tok, model)
    cache[key] = v
    return v


# -----------------------------
# 3) Anisotropy metrics
# -----------------------------

def l2_normalize(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.maximum(n, eps)


def cosine_matrix(X: np.ndarray) -> np.ndarray:
    Xn = l2_normalize(X)
    return Xn @ Xn.T


def mean_pairwise_cosine(X: np.ndarray) -> float:
    C = cosine_matrix(X)
    n = C.shape[0]
    return float((C.sum() - np.trace(C)) / (n * (n - 1)))


def cos_to_mean_stats(X: np.ndarray) -> Tuple[float, float]:
    Xn = l2_normalize(X)
    mu = l2_normalize(Xn.mean(axis=0, keepdims=True))
    sims = (Xn @ mu.T).reshape(-1)
    return float(sims.mean()), float(sims.std())


def pca_evr(X: np.ndarray, k: int = 10) -> np.ndarray:
    from sklearn.decomposition import PCA  # type: ignore
    pca = PCA(n_components=min(k, X.shape[1]))
    pca.fit(X)
    return pca.explained_variance_ratio_


def remove_top_pcs(X: np.ndarray, m: int) -> np.ndarray:
    from sklearn.decomposition import PCA  # type: ignore
    Xc = X - X.mean(axis=0, keepdims=True)
    pca = PCA(n_components=m)
    pca.fit(Xc)
    comps = pca.components_
    proj = Xc @ comps.T @ comps
    return Xc - proj


def apply_transform(X: np.ndarray, transform: str) -> np.ndarray:
    """
    transform in {"raw","mean_center","mean_center_l2","rm_top_pcs_1","rm_top_pcs_2","rm_top_pcs_5"}.
    """
    if transform == "raw":
        return X
    if transform == "mean_center":
        return X - X.mean(axis=0, keepdims=True)
    if transform == "mean_center_l2":
        Xc = X - X.mean(axis=0, keepdims=True)
        return l2_normalize(Xc)
    if transform.startswith("rm_top_pcs_"):
        m = int(transform.split("_")[-1])
        return remove_top_pcs(X, m=m)
    raise ValueError(f"Unknown transform: {transform}")


# -----------------------------
# Part 3/4 helpers
# -----------------------------

def build_word_vectors_glove(glove: Dict[str, np.ndarray], words: List[str]) -> Dict[str, np.ndarray]:
    out = {}
    for w in words:
        w2 = w.lower()
        if w2 in glove:
            out[w2] = glove[w2]
    return out


def build_word_vectors_bert_from_contexts(
    word_to_contexts: Dict[str, List[str]],
    layers: List[int],
    tok,
    model,
    bert_cache: Dict,
    max_ctx_per_word: int = 50,
) -> Dict[int, Dict[str, np.ndarray]]:
    """
    For each word, compute one vector per layer by averaging contextual embeddings across contexts.
    """
    out: Dict[int, Dict[str, np.ndarray]] = {L: {} for L in layers}
    for w, ctxs in word_to_contexts.items():
        ctxs = ctxs[:max_ctx_per_word]
        for L in layers:
            vecs = []
            for sent in ctxs:
                v = cached_bert_embedding(sent, w, L, tok, model, bert_cache)
                if v is not None:
                    vecs.append(v)
            if vecs:
                out[L][w.lower()] = np.mean(np.stack(vecs, axis=0), axis=0)
    return out


def sample_inter_family_pairs(mf_keep: pd.DataFrame, n_pairs: int, seed: int = 42) -> List[Tuple[str, str]]:
    """
    Sample random pairs of words from different lemmas (different families).
    """
    rng = random.Random(seed)
    lemma_to_words = mf_keep.groupby("lemma")["word"].apply(list).to_dict()
    lemmas = list(lemma_to_words.keys())

    pairs = set()
    while len(pairs) < n_pairs and len(lemmas) >= 2:
        a, b = rng.sample(lemmas, 2)
        wa = rng.choice(lemma_to_words[a]).lower()
        wb = rng.choice(lemma_to_words[b]).lower()
        if wa != wb:
            pairs.add(tuple(sorted((wa, wb))))
    return list(pairs)


def compute_pair_sims(word_vecs: Dict[str, np.ndarray], pairs: List[Tuple[str, str]]) -> np.ndarray:
    sims = []
    for a, b in pairs:
        if a in word_vecs and b in word_vecs:
            sims.append(cosine(word_vecs[a], word_vecs[b]))
    return np.array([x for x in sims if not np.isnan(x)], dtype=float)


# -----------------------------
# Part 3(i): Intra vs inter
# -----------------------------

def run_part3_i_intra_inter(
    mf_keep: pd.DataFrame,
    glove: Dict[str, np.ndarray],
    bert_word_vecs_by_layer: Dict[int, Dict[str, np.ndarray]],
    transforms: List[str],
):
    intra_pairs = []
    for _, grp in mf_keep.groupby("lemma"):
        words = [w.lower() for w in grp["word"].tolist()]
        words = list(dict.fromkeys(words))
        for i in range(len(words)):
            for j in range(i + 1, len(words)):
                intra_pairs.append(tuple(sorted((words[i], words[j]))))

    inter_pairs = sample_inter_family_pairs(mf_keep, n_pairs=len(intra_pairs), seed=SEED)

    rows = []

    # GloVe
    glove_words = sorted(set(mf_keep["word"].str.lower()))
    glove_vecs = build_word_vectors_glove(glove, glove_words)
    keys = list(glove_vecs.keys())
    X = np.stack([glove_vecs[w] for w in keys], axis=0)

    for tr in transforms:
        Xtr = apply_transform(X, tr)
        glove_vecs_tr = {w: Xtr[i] for i, w in enumerate(keys)}

        intra = compute_pair_sims(glove_vecs_tr, intra_pairs)
        inter = compute_pair_sims(glove_vecs_tr, inter_pairs)

        for s in intra:
            rows.append({"model": "glove", "layer": None, "transform": tr, "group": "intra", "cos": float(s)})
        for s in inter:
            rows.append({"model": "glove", "layer": None, "transform": tr, "group": "inter", "cos": float(s)})

    # BERT
    for L, wvecs in bert_word_vecs_by_layer.items():
        if not wvecs:
            continue
        keys = list(wvecs.keys())
        X = np.stack([wvecs[w] for w in keys], axis=0)

        for tr in transforms:
            Xtr = apply_transform(X, tr)
            wvecs_tr = {w: Xtr[i] for i, w in enumerate(keys)}

            intra = compute_pair_sims(wvecs_tr, intra_pairs)
            inter = compute_pair_sims(wvecs_tr, inter_pairs)

            for s in intra:
                rows.append({"model": "bert", "layer": L, "transform": tr, "group": "intra", "cos": float(s)})
            for s in inter:
                rows.append({"model": "bert", "layer": L, "transform": tr, "group": "inter", "cos": float(s)})

    df = pd.DataFrame(rows)
    out_csv = TABLES_DIR / "part3_i_intra_inter.csv"
    df.to_csv(out_csv, index=False)
    print(f"[Part 3(i)] wrote {out_csv}")

    # quick plots for mean_center_l2 (glove and bert)
    for model in ["glove", "bert"]:
        sub = df[(df["model"] == model) & (df["transform"] == "mean_center_l2")]
        if sub.empty:
            continue
        plt.figure()
        for grp in ["intra", "inter"]:
            vals = sub[sub["group"] == grp]["cos"].values
            plt.hist(vals, bins=40, alpha=0.5, label=grp)
        plt.legend()
        plt.title(f"Part 3(i) {model} (mean_center_l2)")
        plt.xlabel("cosine similarity")
        plt.ylabel("count")
        plt.tight_layout()
        out_png = PLOTS_DIR / f"part3_i_{model}_hist.png"
        plt.savefig(out_png, dpi=150)
        plt.close()
        print(f"[Part 3(i)] saved {out_png}")


# -----------------------------
# Part 3(ii): Offset consistency
# -----------------------------

def run_part3_ii_offsets(
    mf_keep: pd.DataFrame,
    glove: Dict[str, np.ndarray],
    bert_word_vecs_by_layer: Dict[int, Dict[str, np.ndarray]],
    transform_choice: Optional[str] = None,
):
    if "transform_type" not in mf_keep.columns:
        print("[Part 3(ii)] No transform_type column found; skipping.")
        return

    counts = mf_keep["transform_type"].value_counts()
    if counts.empty:
        print("[Part 3(ii)] Empty transform_type counts; skipping.")
        return

    if transform_choice is None:
        transform_choice = str(counts.index[0])

    sub = mf_keep[mf_keep["transform_type"] == transform_choice].copy()

    pairs = []
    for _, r in sub.iterrows():
        lemma = str(r["lemma"]).lower()
        w = str(r["word"]).lower()
        if lemma != w:
            pairs.append((lemma, w))

    if len(pairs) < 10:
        print(f"[Part 3(ii)] Not enough pairs for transform_type={transform_choice} (found {len(pairs)}).")
        return

    def offset_sims(word_vecs: Dict[str, np.ndarray]) -> np.ndarray:
        deltas = []
        for lemma, w in pairs:
            if lemma in word_vecs and w in word_vecs:
                deltas.append(word_vecs[w] - word_vecs[lemma])
        if len(deltas) < 5:
            return np.array([], dtype=float)
        deltas = np.stack(deltas, axis=0)
        deltas = l2_normalize(deltas)

        sims = []
        for i in range(len(deltas)):
            for j in range(i + 1, len(deltas)):
                sims.append(float(np.dot(deltas[i], deltas[j])))
        return np.array(sims, dtype=float)

    rows = []

    # GloVe
    glove_words = sorted(set(mf_keep["word"].str.lower()) | set(mf_keep["lemma"].str.lower()))
    glove_vecs = build_word_vectors_glove(glove, glove_words)
    sims = offset_sims(glove_vecs)
    for s in sims:
        rows.append({"model": "glove", "layer": None, "transform_type": transform_choice, "delta_cos": float(s)})

    # BERT
    for L, wvecs in bert_word_vecs_by_layer.items():
        sims = offset_sims(wvecs)
        for s in sims:
            rows.append({"model": "bert", "layer": L, "transform_type": transform_choice, "delta_cos": float(s)})

    df = pd.DataFrame(rows)
    out_csv = TABLES_DIR / "part3_ii_offset_consistency.csv"
    df.to_csv(out_csv, index=False)
    print(f"[Part 3(ii)] transform_type={transform_choice} wrote {out_csv}")


# -----------------------------
# Part 3(iii): Probing classifier
# -----------------------------

def run_part3_iii_probe(
    mf_keep: pd.DataFrame,
    glove: Dict[str, np.ndarray],
    bert_word_vecs_by_layer: Dict[int, Dict[str, np.ndarray]],
):
    data = []
    for _, r in mf_keep.iterrows():
        lemma = str(r["lemma"]).lower()
        w = str(r["word"]).lower()
        y = 0 if w == lemma else 1
        data.append((w, y))

    tmp: Dict[str, List[int]] = {}
    for w, y in data:
        tmp.setdefault(w, []).append(y)

    items = [(w, int(round(np.mean(ys)))) for w, ys in tmp.items()]
    words = [w for w, _ in items]
    y = np.array([lab for _, lab in items], dtype=int)

    def eval_probe(word_vecs: Dict[str, np.ndarray], model_name: str, layer: Optional[int]):
        X_list, y_list = [], []
        for w, lab in zip(words, y):
            if w in word_vecs:
                X_list.append(word_vecs[w])
                y_list.append(int(lab))

        if len(set(y_list)) < 2 or len(y_list) < 20:
            return None

        X = np.stack(X_list, axis=0)
        y2 = np.array(y_list, dtype=int)

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
        accs, f1s = [], []
        for train, test in skf.split(X, y2):
            clf = LogisticRegression(max_iter=2000)
            clf.fit(X[train], y2[train])
            pred = clf.predict(X[test])
            accs.append(accuracy_score(y2[test], pred))
            f1s.append(f1_score(y2[test], pred, average="macro"))

        return {
            "model": model_name,
            "layer": layer,
            "n": int(len(y2)),
            "acc_mean": float(np.mean(accs)),
            "acc_std": float(np.std(accs)),
            "macro_f1_mean": float(np.mean(f1s)),
            "macro_f1_std": float(np.std(f1s)),
        }

    rows = []

    glove_vecs = build_word_vectors_glove(glove, words)
    res = eval_probe(glove_vecs, "glove", None)
    if res:
        rows.append(res)

    for L, wvecs in bert_word_vecs_by_layer.items():
        res = eval_probe(wvecs, "bert", L)
        if res:
            rows.append(res)

    df = pd.DataFrame(rows)
    out_csv = TABLES_DIR / "part3_iii_probe.csv"
    df.to_csv(out_csv, index=False)
    print(f"[Part 3(iii)] wrote {out_csv}")


# -----------------------------
# Part 4: Synonyms vs antonyms
# -----------------------------

def run_part4_syn_ant(glove: Dict[str, np.ndarray], bert_word_vecs_by_layer: Dict[int, Dict[str, np.ndarray]]):
    syn_csv = TABLES_DIR / "wordnet_syn_pairs.csv"
    ant_csv = TABLES_DIR / "wordnet_ant_pairs.csv"
    if not syn_csv.exists() or not ant_csv.exists():
        print("[Part 4] Missing WordNet pair CSVs; skipping.")
        return

    syn = pd.read_csv(syn_csv)
    ant = pd.read_csv(ant_csv)
    syn_pairs = [(a.lower(), b.lower()) for a, b in zip(syn["w1"], syn["w2"])]
    ant_pairs = [(a.lower(), b.lower()) for a, b in zip(ant["w1"], ant["w2"])]

    rng = random.Random(SEED)
    vocab = sorted(
        set(
            list(syn["w1"].str.lower())
            + list(syn["w2"].str.lower())
            + list(ant["w1"].str.lower())
            + list(ant["w2"].str.lower())
        )
    )

    rand_pairs = []
    while len(rand_pairs) < 200 and len(vocab) >= 2:
        a, b = rng.sample(vocab, 2)
        rand_pairs.append((a, b))

    rows = []

    # glove
    glove_vecs = build_word_vectors_glove(glove, vocab)
    for label, pairs in [("syn", syn_pairs), ("ant", ant_pairs), ("rand", rand_pairs)]:
        sims = compute_pair_sims(glove_vecs, pairs)
        for s in sims:
            rows.append({"model": "glove", "layer": None, "pair_type": label, "cos": float(s)})

    # bert
    for L, wvecs in bert_word_vecs_by_layer.items():
        for label, pairs in [("syn", syn_pairs), ("ant", ant_pairs), ("rand", rand_pairs)]:
            sims = compute_pair_sims(wvecs, pairs)
            for s in sims:
                rows.append({"model": "bert", "layer": L, "pair_type": label, "cos": float(s)})

    df = pd.DataFrame(rows)
    out_csv = TABLES_DIR / "part4_syn_ant_sims.csv"
    df.to_csv(out_csv, index=False)
    print(f"[Part 4] wrote {out_csv}")

    # quick plot for glove syn vs ant
    plt.figure()
    sub = df[(df["model"] == "glove") & (df["pair_type"].isin(["syn", "ant"]))]
    for label in ["syn", "ant"]:
        vals = sub[sub["pair_type"] == label]["cos"].values
        plt.hist(vals, bins=40, alpha=0.5, label=label)
    plt.legend()
    plt.title("Part 4 GloVe: synonyms vs antonyms")
    plt.xlabel("cosine similarity")
    plt.ylabel("count")
    plt.tight_layout()
    out_png = PLOTS_DIR / "part4_glove_syn_ant_hist.png"
    plt.savefig(out_png, dpi=150)
    plt.close()
    print(f"[Part 4] saved {out_png}")


# -----------------------------
# MAIN
# -----------------------------

def main(
    morph_path: Path,
    glove_path: Optional[Path],
    force_rebuild_corpus: bool = False,
    corpus_cap: Optional[int] = DEFAULT_CORPUS_CAP,
):
    print("Loading/Building corpus cache...")
    sents = build_corpus_cache(force=force_rebuild_corpus)
    if corpus_cap is not None:
        sents = sents[:corpus_cap]
        print(f"Sentences: {len(sents):,} (capped)")
    else:
        print(f"Sentences: {len(sents):,}")

    freqs = word_frequencies(sents)

    print("Loading morph families...")
    mf = load_morph_families(morph_path)
    mf_long = explode_family_forms(mf)
    print(f"Families rows (long): {len(mf_long):,}")

    # load GloVe filtered by morph words (first pass)
    glove = None
    glove_vocab: set[str] = set()
    if glove_path is not None:
        print("Loading GloVe vectors (filtered)...")
        needed = set(mf_long["word"].str.lower().tolist())
        glove = load_glove_vectors_filtered(glove_path, needed_words=needed)
        glove_vocab = set(glove.keys())
        print(f"GloVe loaded: {len(glove_vocab):,} dim={len(next(iter(glove.values())))}")

    print("Filtering morph words by frequency and GloVe vocab (if available)...")
    mf_long["freq"] = mf_long["word"].map(lambda w: freqs.get(w, 0))

    if glove is not None:
        mf_keep = mf_long[(mf_long["freq"] >= MIN_WORD_FREQ) & (mf_long["word"].isin(glove_vocab))].copy()
    else:
        mf_keep = mf_long[(mf_long["freq"] >= MIN_WORD_FREQ)].copy()

    mf_keep.to_csv(TABLES_DIR / "morph_words_filtered.csv", index=False)
    print(f"Kept morph words: {len(mf_keep):,} (saved to outputs/tables/morph_words_filtered.csv)")

    print("Collecting contexts per word (single pass, 10..50)...")
    target_words = sorted(mf_keep["word"].unique())

    word_to_contexts, word_status = collect_contexts_single_pass(
        sentences=sents,
        target_words=target_words,
        min_ctx=MIN_CTX,
        max_ctx=MAX_CTX,
        base_seed=SEED,
        progress_every=200_000,
    )

    save_pickle(word_to_contexts, CACHE_DIR / "word_contexts.pkl")
    pd.DataFrame.from_dict(word_status, orient="index").reset_index().rename(columns={"index": "word"}).to_csv(
        TABLES_DIR / "word_context_status.csv", index=False
    )
    print("Saved contexts cache + status table.")

    if glove is None:
        print("\n[STOP] GloVe not provided. Provide --glove path/to/glove.txt to run full pipeline.")
        return

    print("Building WordNet pairs...")
    syn_raw, ant_raw = build_wordnet_pairs()

    # Reload GloVe filtered on (morph words ∪ subset of WordNet words) for pair filtering
    wn_needed = set()
    for a, b in syn_raw[:5000]:
        wn_needed.add(a)
        wn_needed.add(b)
    for a, b in ant_raw[:5000]:
        wn_needed.add(a)
        wn_needed.add(b)

    needed2 = set(mf_long["word"].str.lower().tolist()) | wn_needed
    glove = load_glove_vectors_filtered(glove_path, needed_words=needed2)
    glove_vocab = set(glove.keys())
    print(f"GloVe reloaded for WordNet filtering: {len(glove_vocab):,} vectors")

    syn = filter_pairs(syn_raw, glove_vocab, freqs, min_freq=MIN_WORD_FREQ, target_n=200)
    ant = filter_pairs(ant_raw, glove_vocab, freqs, min_freq=MIN_WORD_FREQ, target_n=200)

    pd.DataFrame(syn, columns=["w1", "w2"]).to_csv(TABLES_DIR / "wordnet_syn_pairs.csv", index=False)
    pd.DataFrame(ant, columns=["w1", "w2"]).to_csv(TABLES_DIR / "wordnet_ant_pairs.csv", index=False)
    print(f"Saved syn pairs: {len(syn)} | ant pairs: {len(ant)}")

    print("Loading BERT...")
    cfg = BertConfig()
    tok, model = load_bert(cfg)

    bert_cache_path = CACHE_DIR / "bert_cache.pkl"
    bert_cache = load_pickle(bert_cache_path) if bert_cache_path.exists() else {}

    # Part 2: anisotropy metrics using sampled contextual embeddings
    print("Sampling contextual embeddings for anisotropy metrics...")
    candidates = [(w, sent) for w, ctxs in word_to_contexts.items() for sent in ctxs]
    random.Random(SEED).shuffle(candidates)
    M = min(2000, len(candidates))
    candidates = candidates[:M]

    static_words = sorted({w for w, _ in candidates if w in glove})
    X_glove = np.stack([glove[w] for w in static_words], axis=0)

    results = []

    for layer in cfg.layers:
        X_bert = []
        for w, sent in candidates:
            v = cached_bert_embedding(sent, w, layer, tok, model, bert_cache)
            if v is not None:
                X_bert.append(v)
        X_bert = np.stack(X_bert, axis=0)
        print(f"Layer {layer}: collected {len(X_bert)} contextual vectors")

        res = {
            "model": "bert",
            "layer": layer,
            "transform": "raw",
            "M": int(X_bert.shape[0]),
            "A1_mean_pairwise_cos": mean_pairwise_cosine(X_bert),
            "cos_to_mean_mu": cos_to_mean_stats(X_bert)[0],
            "cos_to_mean_sigma": cos_to_mean_stats(X_bert)[1],
            "pca_evr_k10": json.dumps(pca_evr(X_bert, k=10).tolist()),
        }
        results.append(res)

        Xc = X_bert - X_bert.mean(axis=0, keepdims=True)
        res2 = res.copy()
        res2["transform"] = "mean_center"
        res2["A1_mean_pairwise_cos"] = mean_pairwise_cosine(Xc)
        res2["cos_to_mean_mu"], res2["cos_to_mean_sigma"] = cos_to_mean_stats(Xc)
        res2["pca_evr_k10"] = json.dumps(pca_evr(Xc, k=10).tolist())
        results.append(res2)

        Xcn = l2_normalize(Xc)
        res3 = res.copy()
        res3["transform"] = "mean_center_l2"
        res3["A1_mean_pairwise_cos"] = mean_pairwise_cosine(Xcn)
        res3["cos_to_mean_mu"], res3["cos_to_mean_sigma"] = cos_to_mean_stats(Xcn)
        res3["pca_evr_k10"] = json.dumps(pca_evr(Xcn, k=10).tolist())
        results.append(res3)

        for m in [1, 2, 5]:
            Xr = remove_top_pcs(X_bert, m=m)
            results.append(
                {
                    "model": "bert",
                    "layer": layer,
                    "transform": f"rm_top_pcs_{m}",
                    "M": int(Xr.shape[0]),
                    "A1_mean_pairwise_cos": mean_pairwise_cosine(Xr),
                    "cos_to_mean_mu": cos_to_mean_stats(Xr)[0],
                    "cos_to_mean_sigma": cos_to_mean_stats(Xr)[1],
                    "pca_evr_k10": json.dumps(pca_evr(Xr, k=10).tolist()),
                }
            )

    print("GloVe anisotropy metrics...")
    X = X_glove
    results.append(
        {
            "model": "glove",
            "layer": None,
            "transform": "raw",
            "M": int(X.shape[0]),
            "A1_mean_pairwise_cos": mean_pairwise_cosine(X),
            "cos_to_mean_mu": cos_to_mean_stats(X)[0],
            "cos_to_mean_sigma": cos_to_mean_stats(X)[1],
            "pca_evr_k10": json.dumps(pca_evr(X, k=10).tolist()),
        }
    )
    Xc = X - X.mean(axis=0, keepdims=True)
    results.append(
        {
            "model": "glove",
            "layer": None,
            "transform": "mean_center",
            "M": int(Xc.shape[0]),
            "A1_mean_pairwise_cos": mean_pairwise_cosine(Xc),
            "cos_to_mean_mu": cos_to_mean_stats(Xc)[0],
            "cos_to_mean_sigma": cos_to_mean_stats(Xc)[1],
            "pca_evr_k10": json.dumps(pca_evr(Xc, k=10).tolist()),
        }
    )
    Xcn = l2_normalize(Xc)
    results.append(
        {
            "model": "glove",
            "layer": None,
            "transform": "mean_center_l2",
            "M": int(Xcn.shape[0]),
            "A1_mean_pairwise_cos": mean_pairwise_cosine(Xcn),
            "cos_to_mean_mu": cos_to_mean_stats(Xcn)[0],
            "cos_to_mean_sigma": cos_to_mean_stats(Xcn)[1],
            "pca_evr_k10": json.dumps(pca_evr(Xcn, k=10).tolist()),
        }
    )
    for m in [1, 2, 5]:
        Xr = remove_top_pcs(X, m=m)
        results.append(
            {
                "model": "glove",
                "layer": None,
                "transform": f"rm_top_pcs_{m}",
                "M": int(Xr.shape[0]),
                "A1_mean_pairwise_cos": mean_pairwise_cosine(Xr),
                "cos_to_mean_mu": cos_to_mean_stats(Xr)[0],
                "cos_to_mean_sigma": cos_to_mean_stats(Xr)[1],
                "pca_evr_k10": json.dumps(pca_evr(Xr, k=10).tolist()),
            }
        )

    metrics_df = pd.DataFrame(results)
    metrics_df.to_csv(TABLES_DIR / "anisotropy_metrics.csv", index=False)
    print("Saved anisotropy metrics table: outputs/tables/anisotropy_metrics.csv")

    save_pickle(bert_cache, bert_cache_path)
    print("Saved BERT cache.")

    # -----------------------------
    # PART 3 + PART 4 runs
    # -----------------------------
    print("Building per-word BERT vectors (mean over contexts) for Part 3/4...")
    layers = [1, 6, 12]
    bert_word_vecs_by_layer = build_word_vectors_bert_from_contexts(
        word_to_contexts=word_to_contexts,
        layers=layers,
        tok=tok,
        model=model,
        bert_cache=bert_cache,
        max_ctx_per_word=50,
    )

    run_part3_i_intra_inter(
        mf_keep=mf_keep,
        glove=glove,
        bert_word_vecs_by_layer=bert_word_vecs_by_layer,
        transforms=["raw", "mean_center_l2", "rm_top_pcs_1", "rm_top_pcs_2", "rm_top_pcs_5"],
    )

    run_part3_ii_offsets(
        mf_keep=mf_keep,
        glove=glove,
        bert_word_vecs_by_layer=bert_word_vecs_by_layer,
        transform_choice=None,
    )

    run_part3_iii_probe(
        mf_keep=mf_keep,
        glove=glove,
        bert_word_vecs_by_layer=bert_word_vecs_by_layer,
    )

    run_part4_syn_ant(
        glove=glove,
        bert_word_vecs_by_layer=bert_word_vecs_by_layer,
    )

    print("Part 3/4 finished. Check outputs/tables and outputs/plots.")


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--morph", type=str, default="morph_families.tsv", help="Path to morph_families.tsv")
    p.add_argument("--glove", type=str, default="", help="Path to GloVe .txt vectors")
    p.add_argument("--force_corpus", action="store_true", help="Force rebuild corpus cache")
    p.add_argument("--cap", type=int, default=0, help="Optional cap on number of sentences (0 = no cap)")
    args = p.parse_args()

    morph_path = Path(args.morph)
    glove_path = Path(args.glove) if args.glove.strip() else None
    cap = args.cap if args.cap > 0 else None

    main(morph_path=morph_path, glove_path=glove_path, force_rebuild_corpus=args.force_corpus, corpus_cap=cap)