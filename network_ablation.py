#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import math
import os
import random
import re
import sys
import time
import unicodedata
import warnings
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Dict, Iterable, List, Sequence, Set, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.optimize import minimize
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

warnings.filterwarnings("ignore", message="The least populated class")


# =============================================================================
# COPIED UNCHANGED from Experiments_P1.py
# =============================================================================
ENCODERS = {
    "mbert": "bert-base-multilingual-cased",
    "xlmr": "FacebookAI/xlm-roberta-base",
    "parsbert": "HooshvareLab/bert-base-parsbert-uncased",
}


COARSE_CLASSES = ["normal", "offensive", "hateful"]


@dataclass(frozen=True)
class Config:
    split_seed: int = 42
    n_folds: int = 5
    val_user_frac: float = 0.15
    model_seed: int = 47
    max_posts_per_train_user: int = 20
    hidden_dim: int = 128
    dropout: float = 0.30
    lr: float = 5e-4
    weight_decay: float = 1e-4
    batch_stage1: int = 512
    batch_stage2: int = 256
    max_epochs_stage1: int = 70
    max_epochs_stage2: int = 120
    patience_stage1: int = 10
    patience_stage2: int = 15
    hateful_class_weight_boost: float = 2.5
    grad_clip: float = 1.0
    stage1_prior_strength: float = 2.0
    stage2_user_profile_strength: float = 2.0
    stage2_social_strength: float = 5.0
    inner_alpha_folds: int = 3
    inner_stage2_max_epochs: int = 80
    inner_stage2_patience: int = 10
    stage2_alpha_max: float = 2.5
    stage2_alpha_reg: float = 0.02
    max_length: int = 128
    encoder_batch: int = 16


@dataclass
class Fold:
    fold: int
    model_users: Set
    val_users: Set
    test_users: Set
    model_rows: np.ndarray
    val_rows: np.ndarray
    test_rows: np.ndarray


@dataclass
class DataState:
    df: pd.DataFrame
    users: List
    user_index: Dict
    user_posts: Dict
    post_user_index: np.ndarray
    adjacency: np.ndarray
    adjacency_exact2: np.ndarray
    y_stage1: np.ndarray
    y_stage2: np.ndarray
    y_final9: np.ndarray
    stage2_classes: List[str]
    final_classes: List[str]
    folds: List[Fold]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalize_probs(x: np.ndarray, fallback: np.ndarray | None = None) -> np.ndarray:
    x = np.maximum(np.asarray(x, dtype=np.float32), 0.0)
    sums = x.sum(axis=1, keepdims=True)
    out = x / np.maximum(sums, 1e-12)
    if fallback is not None:
        bad = sums[:, 0] <= 1e-12
        if bad.any():
            out[bad] = np.asarray(fallback, dtype=np.float32)
    return out.astype(np.float32)


def softmax_np(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=np.float64)
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return (e / np.maximum(e.sum(axis=1, keepdims=True), 1e-12)).astype(np.float32)


def safe_log(p: np.ndarray) -> np.ndarray:
    return np.log(np.clip(np.asarray(p, dtype=np.float64), 1e-8, 1.0))


def exact_two_hop(a: np.ndarray) -> np.ndarray:
    b = (a > 0).astype(np.float32)
    x = (b @ b) > 0
    x[b > 0] = False
    np.fill_diagonal(x, False)
    return x.astype(np.float32)


def encode_texts(state: DataState, model_id: str, cfg: Config, device: torch.device) -> np.ndarray:
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id).to(device).eval()
    for p in model.parameters():
        p.requires_grad = False

    texts = state.df["text"].astype(str).tolist()
    chunks = []
    for start in tqdm(range(0, len(texts), cfg.encoder_batch), desc=model_id):
        batch = tokenizer(
            texts[start:start + cfg.encoder_batch],
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=cfg.max_length,
        ).to(device)
        with torch.no_grad():
            hidden = model(**batch).last_hidden_state[:, 0, :]
        chunks.append(hidden.cpu().numpy())

    embeddings = np.concatenate(chunks, axis=0).astype(np.float32)
    del model, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return embeddings


class TextClassifier(nn.Module):
    def __init__(self, input_dim: int, n_classes: int, cfg: Config):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, cfg.hidden_dim),
            nn.LayerNorm(cfg.hidden_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim // 2),
            nn.LayerNorm(cfg.hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_dim // 2, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def class_weights(y: np.ndarray, n_classes: int, device: torch.device, hateful_boost: float | None = None) -> torch.Tensor:
    counts = np.bincount(np.asarray(y, dtype=int), minlength=n_classes).astype(np.float32)
    counts = np.maximum(counts, 1.0)
    weights = len(y) / (n_classes * counts)
    if hateful_boost is not None and n_classes == 3:
        weights[2] *= hateful_boost
    return torch.tensor(weights, dtype=torch.float32, device=device)


def predict_logits(model: nn.Module, x: np.ndarray, device: torch.device, batch_size: int = 4096) -> np.ndarray:
    model.eval()
    out = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            xb = torch.tensor(x[start:start + batch_size], dtype=torch.float32, device=device)
            out.append(model(xb).cpu().numpy())
    return np.concatenate(out, axis=0).astype(np.float32)


def train_classifier(
    x: np.ndarray,
    y: np.ndarray,
    train_rows: np.ndarray,
    val_rows: np.ndarray,
    n_classes: int,
    max_epochs: int,
    patience: int,
    batch_size: int,
    seed: int,
    cfg: Config,
    device: torch.device,
) -> nn.Module:
    seed_everything(seed)
    train_rows = np.asarray(train_rows, dtype=np.int64)
    val_rows = np.asarray(val_rows, dtype=np.int64)
    model = TextClassifier(x.shape[1], n_classes, cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    criterion = nn.CrossEntropyLoss(
        weight=class_weights(
            y[train_rows],
            n_classes,
            device,
            cfg.hateful_class_weight_boost if n_classes == 3 else None,
        )
    )
    rng = np.random.RandomState(seed + 77)
    best_score = -math.inf
    best_state = copy.deepcopy(model.state_dict())
    bad_epochs = 0

    for _ in range(max_epochs):
        model.train()
        epoch_rows = rng.permutation(train_rows)
        for start in range(0, len(epoch_rows), batch_size):
            idx = epoch_rows[start:start + batch_size]
            if len(idx) < 2:
                continue
            xb = torch.tensor(x[idx], dtype=torch.float32, device=device)
            yb = torch.tensor(y[idx], dtype=torch.long, device=device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()

        if len(val_rows):
            pred = predict_logits(model, x[val_rows], device).argmax(axis=1)
            score = f1_score(
                y[val_rows], pred, average="macro", labels=list(range(n_classes)), zero_division=0
            )
        else:
            score = 0.0

        if score > best_score + 1e-5:
            best_score = score
            best_state = copy.deepcopy(model.state_dict())
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break

    model.load_state_dict(best_state)
    return model


def fold_text_logits(
    state: DataState,
    embeddings: np.ndarray,
    fold: Fold,
    cfg: Config,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    stage1_model = train_classifier(
        embeddings,
        state.y_stage1,
        fold.model_rows,
        fold.val_rows,
        3,
        cfg.max_epochs_stage1,
        cfg.patience_stage1,
        cfg.batch_stage1,
        cfg.model_seed * 100 + fold.fold,
        cfg,
        device,
    )
    logits1 = predict_logits(stage1_model, embeddings, device)
    del stage1_model

    tr2 = fold.model_rows[(state.y_stage1[fold.model_rows] == 2) & pd.notna(state.y_stage2[fold.model_rows])]
    va2 = fold.val_rows[(state.y_stage1[fold.val_rows] == 2) & pd.notna(state.y_stage2[fold.val_rows])]
    y2 = np.nan_to_num(state.y_stage2, nan=0).astype(int)
    if len(tr2) < 10:
        raise RuntimeError(f"Fold {fold.fold}: insufficient Stage-2 training rows.")

    stage2_model = train_classifier(
        embeddings,
        y2,
        tr2,
        va2,
        len(state.stage2_classes),
        cfg.max_epochs_stage2,
        cfg.patience_stage2,
        cfg.batch_stage2,
        cfg.model_seed * 100 + fold.fold + 50000,
        cfg,
        device,
    )
    logits2 = predict_logits(stage2_model, embeddings, device)
    del stage2_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return logits1, logits2


def stage1_user_profiles(state: DataState, fold: Fold) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    profiles = np.zeros((len(state.users), 3), dtype=np.float32)
    source_mask = np.zeros(len(state.users), dtype=bool)
    global_counts = np.zeros(3, dtype=np.float64)
    model_df = state.df.iloc[fold.model_rows]

    for user, g in model_df.groupby("user"):
        counts = np.bincount(g["stage1_id"].values.astype(int), minlength=3).astype(np.float32)
        if counts.sum() > 0:
            profiles[state.user_index[user]] = counts / counts.sum()
            source_mask[state.user_index[user]] = True
            global_counts += counts

    global_prior = (global_counts / global_counts.sum()).astype(np.float32)
    profiles[~source_mask] = global_prior
    return normalize_probs(profiles, global_prior), source_mask, global_prior


def graph_prior(
    adjacency: np.ndarray,
    profiles: np.ndarray,
    source_mask: np.ndarray,
    global_prior: np.ndarray,
    strength: float,
) -> np.ndarray:
    weights = adjacency * source_mask[None, :].astype(np.float32)
    counts = weights.sum(axis=1, keepdims=True)
    sums = weights @ profiles
    p = (sums + strength * global_prior[None, :]) / np.maximum(counts + strength, 1e-8)
    return normalize_probs(p, global_prior)


def stage2_user_profiles(
    state: DataState,
    source_rows: np.ndarray,
    source_users: Set,
    cfg: Config,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_users = len(state.users)
    n_classes = len(state.stage2_classes)
    counts = np.zeros((n_users, n_classes), dtype=np.float64)
    support = np.zeros(n_users, dtype=np.int32)
    source_df = state.df.iloc[np.asarray(source_rows, dtype=np.int64)]
    source_df = source_df[(source_df["stage1_id"] == 2) & source_df["stage2_id"].notna()]
    global_counts = np.zeros(n_classes, dtype=np.float64)

    for user, g in source_df.groupby("user"):
        if user not in source_users:
            continue
        c = np.bincount(g["stage2_id"].values.astype(int), minlength=n_classes).astype(np.float64)
        counts[state.user_index[user]] = c
        support[state.user_index[user]] = int(c.sum())
        global_counts += c

    if global_counts.sum() <= 0:
        raise RuntimeError("No Stage-2 labels in source rows.")

    global_prior = (global_counts / global_counts.sum()).astype(np.float32)
    source_mask = support > 0
    profiles = np.tile(global_prior[None, :], (n_users, 1)).astype(np.float32)

    for i in np.where(source_mask)[0]:
        profiles[i] = (
            counts[i] + cfg.stage2_user_profile_strength * global_prior
        ) / (support[i] + cfg.stage2_user_profile_strength)

    return normalize_probs(profiles, global_prior), source_mask, global_prior


def balanced_nll(logits: np.ndarray, y: np.ndarray, n_classes: int) -> float:
    probs = softmax_np(logits)
    y = np.asarray(y, dtype=int)
    counts = np.bincount(y, minlength=n_classes).astype(np.float64)
    class_w = np.zeros(n_classes, dtype=np.float64)
    nz = counts > 0
    class_w[nz] = len(y) / (max(int(nz.sum()), 1) * counts[nz])
    sample_w = class_w[y]
    sample_w = sample_w / max(sample_w.mean(), 1e-12)
    loss = -np.log(np.clip(probs[np.arange(len(y)), y], 1e-9, 1.0))
    return float(np.sum(sample_w * loss) / np.maximum(sample_w.sum(), 1e-12))


def fit_alpha(text_logits: np.ndarray, residual: np.ndarray, y: np.ndarray, cfg: Config, n_classes: int) -> float:
    if len(y) < 20:
        return 1.0

    def objective(x):
        alpha = float(x[0])
        return balanced_nll(text_logits + alpha * residual, y, n_classes) + cfg.stage2_alpha_reg * (alpha - 1.0) ** 2

    result = minimize(
        objective,
        np.array([1.0]),
        method="L-BFGS-B",
        bounds=[(0.0, cfg.stage2_alpha_max)],
        options={"maxiter": 150},
    )
    return float(result.x[0] if result.success else 1.0)


def sampled_rows_for_users(state: DataState, users: Iterable) -> np.ndarray:
    chunks = [state.sample_user_posts(u) for u in sorted(set(users), key=str)]
    chunks = [x for x in chunks if len(x)]
    return np.concatenate(chunks).astype(np.int64) if chunks else np.empty(0, dtype=np.int64)


def user_has_hateful(state: DataState, user) -> int:
    rows = state.user_posts[user]
    return int(np.any((state.y_stage1[rows] == 2) & pd.notna(state.y_stage2[rows])))


def final9(stage1_pred: np.ndarray, stage2_pred: np.ndarray) -> np.ndarray:
    out = np.asarray(stage1_pred, dtype=int).copy()
    hateful = stage1_pred == 2
    out[hateful] = np.asarray(stage2_pred, dtype=int)[hateful] + 2
    return out


def metric_values(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> dict:
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(n_classes)), zero_division=0
    )
    return {
        "macro_precision": float(precision.mean()),
        "macro_recall": float(recall.mean()),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1.mean()),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "support": support,
    }

# =============================================================================
# end of COPIED UNCHANGED block


_FA_FIX = str.maketrans({"ي": "ی", "ى": "ی", "ك": "ک", "‌": "", "‍": "",
                         "‏": "", "‎": "", "ـ": "", " ": ""})


def _norm_text(v) -> str:
    return unicodedata.normalize("NFKC", str(v)).translate(_FA_FIX).strip().lower()


def _coarse_name(v):
    s = _norm_text(v)
    if "تنفر" in s or s.startswith("hate"):
        return "hateful"
    if "توهین" in s or s.startswith("offens"):
        return "offensive"
    if "عادی" in s or s == "normal":
        return "normal"
    return None


def norm_id(x) -> str:
    s = str(x).strip()
    if re.fullmatch(r"-?\d+\.0+", s):
        s = s.split(".")[0]
    return s


def _read_csv(path):
    try:
        return pd.read_csv(path, encoding="utf-8-sig")
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="utf-8-sig", encoding_errors="replace")


def _edge_columns(edges: pd.DataFrame, users: List, forced: str) -> Tuple[str, str]:
    if forced:
        cols = [c.strip() for c in forced.split(",")]
        if len(cols) != 2 or any(c not in edges.columns for c in cols):
            raise ValueError(f"--edge_cols {forced} not found. Columns: {list(edges.columns)}")
        return cols[0], cols[1]
    if "source" in edges.columns and "target" in edges.columns:
        return "source", "target"
    ids = {norm_id(u) for u in users}
    scored = sorted(((edges[c].map(norm_id).isin(ids).mean(), c) for c in edges.columns), key=lambda t: -t[0])
    good = [c for frac, c in scored if frac >= 0.5]
    if len(good) < 2:
        raise ValueError(f"Could not find two user-id columns in the edge file: {list(edges.columns)}")
    return good[0], good[1]


def load_data(args: SimpleNamespace, cfg: Config) -> DataState:
    df = _read_csv(args.data)
    required = [args.user_col, args.text_col, args.stage1_col, args.stage2_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing data columns: {missing}. Columns: {list(df.columns)}")

    rename = {
        args.user_col: "user",
        args.text_col: "text",
        args.stage1_col: "stage1",
        args.stage2_col: "stage2",
    }
    df = df.rename(columns=rename).copy()

    raw_stage1 = df["stage1"].astype(str).str.strip()
    mapped = raw_stage1.map(_coarse_name)
    if mapped.isna().any():
        raise ValueError("Unexpected Stage-1 labels:\n" + raw_stage1[mapped.isna()].value_counts().to_string())
    df["stage1"] = mapped
    df["stage2"] = df["stage2"].apply(
        lambda x: str(x).strip() if pd.notna(x) and str(x).strip() else None
    )

    hateful_rows = df["stage1"] == "hateful"
    if df.loc[hateful_rows, "stage2"].isna().any():
        raise ValueError("Every hateful row must have a Stage-2 subtype label.")

    stage2_classes = sorted(df.loc[hateful_rows, "stage2"].dropna().unique().tolist(), key=str)
    if len(stage2_classes) < 2:
        raise ValueError("Stage-2 must contain at least two subtype classes.")
    final_classes = ["normal", "offensive"] + stage2_classes

    coarse_to_id = {"normal": 0, "offensive": 1, "hateful": 2}
    subtype_to_id = {c: i for i, c in enumerate(stage2_classes)}
    final_to_id = {c: i for i, c in enumerate(final_classes)}

    df["stage1_id"] = df["stage1"].map(coarse_to_id).astype(int)
    df["stage2_id"] = df["stage2"].map(subtype_to_id)
    df["final_label"] = np.where(df["stage1"] == "hateful", df["stage2"], df["stage1"])
    df["final_id"] = df["final_label"].map(final_to_id)
    if df["final_id"].isna().any():
        raise ValueError("Could not construct final 9-class labels.")

    df = df.reset_index(drop=True)
    df["row_id"] = np.arange(len(df), dtype=np.int64)

    users = sorted(df["user"].unique(), key=str)
    user_index = {u: i for i, u in enumerate(users)}
    user_posts = {u: g.index.values.astype(np.int64) for u, g in df.groupby("user")}

    edges = _read_csv(args.edges)
    src_col, dst_col = _edge_columns(edges, users, args.edge_cols)
    norm_index = {norm_id(u): i for i, u in enumerate(users)}

    def _lookup(v):
        if v in user_index:
            return user_index[v]
        return norm_index.get(norm_id(v))

    edge_set = set()
    for s, t in zip(edges[src_col], edges[dst_col]):
        a, b = _lookup(s), _lookup(t)
        if a is None or b is None:
            continue
        if a == b:
            continue
        edge_set.add((a, b) if a < b else (b, a))

    adjacency = np.zeros((len(users), len(users)), dtype=np.float32)
    for a, b in edge_set:
        adjacency[a, b] = 1.0
        adjacency[b, a] = 1.0

    y_stage1 = df["stage1_id"].to_numpy(dtype=int)
    y_stage2 = df["stage2_id"].to_numpy()
    y_final9 = df["final_id"].to_numpy(dtype=int)
    post_user_index = np.array([user_index[u] for u in df["user"].values], dtype=np.int64)

    _dominant_cache = {}

    def dominant_user_class(user) -> int:
        if user not in _dominant_cache:
            values = df.loc[df["user"] == user, "stage1_id"].values.astype(int)
            _dominant_cache[user] = int(np.bincount(values, minlength=3).argmax())
        return _dominant_cache[user]

    def _sample_user_posts(user) -> np.ndarray:
        g = df[df["user"] == user]
        cap = cfg.max_posts_per_train_user
        if cap is None or len(g) <= cap:
            return g.index.values.astype(np.int64)
        hateful = g[g.stage1_id == 2]
        offensive = g[g.stage1_id == 1]
        normal = g[g.stage1_id == 0]
        selected = []
        remaining = int(cap)
        if len(hateful):
            selected.append(hateful)
            remaining -= len(hateful)
        if remaining > 0 and len(offensive):
            n = min(remaining, len(offensive))
            selected.append(offensive.head(n))
            remaining -= n
        if remaining > 0 and len(normal):
            selected.append(normal.head(min(remaining, len(normal))))
        out = pd.concat(selected) if selected else g.head(0)
        return out.index.values.astype(np.int64)

    _sample_cache = {}

    def sample_user_posts(user) -> np.ndarray:
        if user not in _sample_cache:
            _sample_cache[user] = _sample_user_posts(user)
        return _sample_cache[user]

    def inner_user_split(outer_train_users: Sequence, split_seed: int) -> Tuple[Set, Set]:
        arr = np.array(list(outer_train_users), dtype=object)
        strata = np.array([dominant_user_class(u) for u in arr], dtype=int)
        try:
            splitter = StratifiedShuffleSplit(
                n_splits=1, test_size=cfg.val_user_frac, random_state=split_seed
            )
            train_i, val_i = next(splitter.split(arr, strata))
        except Exception:
            rng = np.random.RandomState(split_seed)
            perm = rng.permutation(len(arr))
            n_val = max(1, int(round(cfg.val_user_frac * len(arr))))
            val_i, train_i = perm[:n_val], perm[n_val:]
        return set(arr[train_i].tolist()), set(arr[val_i].tolist())

    user_strata = np.array([dominant_user_class(u) for u in users], dtype=int)
    outer = StratifiedKFold(cfg.n_folds, shuffle=True, random_state=cfg.split_seed)
    folds = []

    for fold_id, (train_i, test_i) in enumerate(outer.split(users, user_strata), 1):
        outer_train_users = [users[i] for i in train_i]
        test_users = {users[i] for i in test_i}
        model_users, val_users = inner_user_split(outer_train_users, cfg.split_seed + 100 * fold_id)
        sampled = {u: sample_user_posts(u) for u in model_users | val_users}
        model_rows = np.concatenate([sampled[u] for u in sorted(model_users, key=str)]).astype(np.int64)
        val_rows = np.concatenate([sampled[u] for u in sorted(val_users, key=str)]).astype(np.int64)
        test_rows = df.index[df["user"].isin(test_users)].values.astype(np.int64)
        folds.append(Fold(fold_id, model_users, val_users, test_users, model_rows, val_rows, test_rows))

    state = DataState(
        df=df,
        users=users,
        user_index=user_index,
        user_posts=user_posts,
        post_user_index=post_user_index,
        adjacency=adjacency,
        adjacency_exact2=exact_two_hop(adjacency),
        y_stage1=y_stage1,
        y_stage2=y_stage2,
        y_final9=y_final9,
        stage2_classes=stage2_classes,
        final_classes=final_classes,
        folds=folds,
    )
    state.sample_user_posts = sample_user_posts
    state.inner_user_split = inner_user_split
    return state


def stage2_social_prior(
    state: DataState,
    profiles: np.ndarray,
    source_mask: np.ndarray,
    global_prior: np.ndarray,
    cfg: Config,
    adjacency: np.ndarray | None = None,
) -> np.ndarray:
    if adjacency is None:
        adjacency = state.adjacency
    weights = adjacency * source_mask[None, :].astype(np.float32)
    counts = weights.sum(axis=1, keepdims=True)
    sums = weights @ profiles
    p = (
        sums + cfg.stage2_social_strength * global_prior[None, :]
    ) / np.maximum(counts + cfg.stage2_social_strength, 1e-8)
    return normalize_probs(p, global_prior)


def inner_alpha_pieces(
    state: DataState,
    embeddings: np.ndarray,
    fold: Fold,
    cfg: Config,
    device: torch.device,
) -> List[dict]:
    users = np.array(sorted(fold.model_users, key=str), dtype=object)
    strata = np.array([user_has_hateful(state, u) for u in users], dtype=int)
    n_splits = cfg.inner_alpha_folds
    if len(np.unique(strata)) > 1:
        n_splits = min(n_splits, max(2, int(np.bincount(strata).min())))
    n_splits = min(n_splits, len(users))
    if n_splits < 2:
        return []

    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=cfg.model_seed * 1000 + fold.fold)
    try:
        splits = list(splitter.split(users, strata))
    except Exception:
        splits = list(splitter.split(users, np.zeros(len(users), dtype=int)))

    pieces = []
    y2 = np.nan_to_num(state.y_stage2, nan=0).astype(int)

    for inner_fold, (source_i, held_i) in enumerate(splits, 1):
        source_users = set(users[source_i].tolist())
        held_users = set(users[held_i].tolist())
        inner_train_users, inner_val_users = state.inner_user_split(
            list(source_users), cfg.model_seed * 100000 + fold.fold * 100 + inner_fold
        )
        train_rows = sampled_rows_for_users(state, inner_train_users)
        val_rows = sampled_rows_for_users(state, inner_val_users)
        held_rows = sampled_rows_for_users(state, held_users)
        train_h = train_rows[(state.y_stage1[train_rows] == 2) & pd.notna(state.y_stage2[train_rows])]
        val_h = val_rows[(state.y_stage1[val_rows] == 2) & pd.notna(state.y_stage2[val_rows])]
        held_h = held_rows[(state.y_stage1[held_rows] == 2) & pd.notna(state.y_stage2[held_rows])]
        if len(train_h) < 10 or len(held_h) == 0:
            continue

        model = train_classifier(
            embeddings,
            y2,
            train_h,
            val_h,
            len(state.stage2_classes),
            cfg.inner_stage2_max_epochs,
            cfg.inner_stage2_patience,
            cfg.batch_stage2,
            cfg.model_seed * 1000000 + fold.fold * 10000 + inner_fold * 100,
            cfg,
            device,
        )
        text_logits = predict_logits(model, embeddings[held_h], device)
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        source_rows = sampled_rows_for_users(state, source_users)
        profiles, source_mask, pi7 = stage2_user_profiles(state, source_rows, source_users, cfg)
        pieces.append({"text_logits": text_logits, "held_h": held_h, "profiles": profiles,
                       "source_mask": source_mask, "pi7": pi7, "y": state.y_stage2[held_h].astype(int)})
    return pieces


def alpha_from_pieces(state: DataState, pieces: List[dict], adjacency: np.ndarray, cfg: Config) -> float:
    if not pieces:
        return 1.0
    all_text, all_residual, all_y = [], [], []
    for pc in pieces:
        p_social = stage2_social_prior(state, pc["profiles"], pc["source_mask"], pc["pi7"], cfg, adjacency)
        residual = safe_log(p_social[state.post_user_index[pc["held_h"]]]) - safe_log(pc["pi7"])[None, :]
        all_text.append(pc["text_logits"])
        all_residual.append(residual.astype(np.float32))
        all_y.append(pc["y"])
    return fit_alpha(
        np.concatenate(all_text),
        np.concatenate(all_residual),
        np.concatenate(all_y),
        cfg,
        len(state.stage2_classes),
    )


CONFIG = {
    "csv": "phond.csv",
    "edges": "edges.csv",
    "out": "network_ablation_out",
    "cache_dir": "",
    "user_col": "User_SId",
    "text_col": "Text",
    "stage1_col": "classification",
    "stage2_col": "hate_class",
    "edge_cols": "",
    "encoders": "parsbert,xlmr,mbert",
    "enc_parsbert": ENCODERS["parsbert"],
    "enc_xlmr": ENCODERS["xlmr"],
    "enc_mbert": ENCODERS["mbert"],
    "enc_bertweetfa": "arm-on/BERTweet-FA",
    "seeds": "11,29,47,71,101",
    "n_rewire": 200,
    "swaps_per_edge": 10,
    "n_boot": 1000,
    "bias_range": 3.0,
    "bias_steps": 25,
    "seed": 2026,
    "quick": 0,
}

PAPER = {
    "mbert": {"final9_text": 0.2912, "final9_final": 0.3528, "stage1_text": 0.5682, "stage1_direct12": 0.6364,
              "stage2_text": 0.4726, "stage2_social": 0.4853},
    "xlmr": {"final9_text": 0.3114, "final9_final": 0.3683, "stage1_text": 0.5801, "stage1_direct12": 0.6534,
             "stage2_text": 0.5100, "stage2_social": 0.5344},
    "parsbert": {"final9_text": 0.3649, "final9_final": 0.4192, "stage1_text": 0.6306, "stage1_direct12": 0.6775,
                 "stage2_text": 0.5861, "stage2_social": 0.6000},
}

SYSTEMS = ["text", "text_cal", "prior", "hop1", "hop2", "direct12", "residual", "final", "final_cal", "net_only"]
S1_RULE = {"text": "text", "text_cal": "text", "prior": "prior", "hop1": "hop1", "hop2": "hop2",
           "direct12": "direct12", "residual": "residual", "final": "direct12", "final_cal": "direct12",
           "net_only": "net"}
S2_RULE = {"text": "text", "text_cal": "text", "prior": "text", "hop1": "text", "hop2": "text",
           "direct12": "text", "residual": "social", "final": "social", "final_cal": "social", "net_only": "net"}
CALIBRATED = {"text_cal", "final_cal"}
REPORTED = SYSTEMS + ["final_inductive"]

COMPARISONS = [
    ("final", "text", "network gain (text+network vs text)"),
    ("prior", "text", "prior shift alone (text + 2 log pi vs text)"),
    ("direct12", "prior", "Stage 1 neighbourhood beyond the prior shift"),
    ("final", "prior", "full system beyond the prior shift"),
    ("residual", "text", "neighbourhood with no prior shift at all"),
    ("final_cal", "text_cal", "both arms calibrated on validation"),
    ("hop1", "prior", "one hop beyond the prior shift"),
    ("hop2", "prior", "two hops beyond the prior shift"),
    ("direct12", "hop1", "two hops added to one hop"),
    ("final", "direct12", "Stage 2 neighbourhood (Eq. 4)"),
    ("net_only", "text", "network only (no text) vs text"),
    ("final_inductive", "text", "inductive graph (no test-test edges) vs text"),
    ("final_inductive", "final", "inductive vs transductive graph"),
]

_TYPE_NAMES = [("ethnic", ("قوم", "ethnic")), ("national", ("ملیت", "ملی", "nation")),
               ("religious", ("مذهب", "دینی", "relig")), ("gender", ("جنسیت", "جنس", "gender", "sex")),
               ("political", ("سیاس", "politic")), ("age", ("سنی", "age")), ("other", ("سایر", "دیگر", "other"))]


def type_display(raw) -> str:
    s = _norm_text(raw)
    for name, keys in _TYPE_NAMES:
        if any(k in s for k in keys):
            return name
    return str(raw)


class Tee:
    def __init__(self, path):
        self.f = open(path, "w", encoding="utf-8")
        self.out = sys.stdout

    def write(self, s):
        self.out.write(s)
        self.f.write(s)

    def flush(self):
        self.out.flush()
        self.f.flush()

    def isatty(self):
        return False

    def close(self):
        self.f.close()

    def __getattr__(self, name):
        return getattr(self.out, name)


def log(msg=""):
    print(msg, flush=True)


def parse_run_config() -> dict:
    ap = argparse.ArgumentParser(description="PHOND: ablation of the network terms")
    for k, v in CONFIG.items():
        ap.add_argument(f"--{k}", type=type(v), default=v)
    args, _unknown = ap.parse_known_args()
    run = vars(args)
    run["seeds"] = [int(s) for s in str(run["seeds"]).split(",") if s.strip()]
    run["encoders"] = [s.strip().lower() for s in str(run["encoders"]).split(",") if s.strip()]
    for enc in run["encoders"]:
        if f"enc_{enc}" not in run:
            sys.exit(f"[error] unknown encoder '{enc}'. Use parsbert, xlmr, mbert.")
    if run["quick"]:
        run["seeds"] = run["seeds"][:1]
        run["n_rewire"] = min(run["n_rewire"], 10)
        run["n_boot"] = min(run["n_boot"], 200)
        run["bias_steps"] = min(run["bias_steps"], 13)
    if not run["cache_dir"]:
        run["cache_dir"] = run["out"]
    for key in ("csv", "edges"):
        if not os.path.exists(run[key]):
            sys.exit(f"[error] file not found: {run[key]}  (pass it with --{key} \"...\")")
    return run


def encode_cached(state: DataState, model_id: str, cfg: Config, device: torch.device, cache_dir: str) -> np.ndarray:
    texts = state.df["text"].astype(str).tolist()
    tag = f"|{model_id}|{cfg.max_length}|fp32|b{cfg.encoder_batch}"
    h = hashlib.md5(("\n".join(texts) + tag).encode("utf-8", "replace")).hexdigest()[:10]
    slug = re.sub(r"[^A-Za-z0-9]+", "_", model_id).strip("_")
    path = os.path.join(cache_dir, f"emb_table3_{slug}_{h}.npy")
    if os.path.exists(path):
        arr = np.load(path)
        if arr.shape[0] == len(texts):
            log(f"  loaded cached embeddings: {path}")
            return arr
    log(f"  encoding {len(texts):,} texts with {model_id} (as in Experiments_P1: fp32, batch {cfg.encoder_batch})")
    emb = encode_texts(state, model_id, cfg, device)
    np.save(path, emb)
    log(f"  saved embeddings: {path}")
    return emb


def _swap_kernel(edges, adj, ia, ib, flip):
    n_ok = 0
    for t in range(ia.shape[0]):
        i = ia[t]
        j = ib[t]
        if i == j:
            continue
        a = edges[i, 0]
        b = edges[i, 1]
        if flip[t] == 1:
            c = edges[j, 1]
            d = edges[j, 0]
        else:
            c = edges[j, 0]
            d = edges[j, 1]
        if a == d or c == b:
            continue
        if adj[a, d] == 1 or adj[c, b] == 1:
            continue
        adj[a, b] = 0
        adj[b, a] = 0
        adj[c, d] = 0
        adj[d, c] = 0
        adj[a, d] = 1
        adj[d, a] = 1
        adj[c, b] = 1
        adj[b, c] = 1
        edges[i, 0] = a
        edges[i, 1] = d
        edges[j, 0] = c
        edges[j, 1] = b
        n_ok += 1
    return n_ok


try:
    import numba
    _SWAP = numba.njit(cache=False)(_swap_kernel)
    HAVE_NUMBA = True
except Exception:
    _SWAP = _swap_kernel
    HAVE_NUMBA = False


def rewire_many(e0: np.ndarray, n: int, n_rep: int, swaps_per_edge: int, rng) -> Tuple[np.ndarray, float]:
    E = len(e0)
    out = np.empty((n_rep, E, 2), dtype=np.int32)
    n_try = int(swaps_per_edge * E)
    acc = 0.0
    for r in tqdm(range(n_rep), desc="rewiring", mininterval=2):
        e = np.ascontiguousarray(e0.copy())
        adj = np.zeros((n, n), dtype=np.uint8)
        adj[e[:, 0], e[:, 1]] = 1
        adj[e[:, 1], e[:, 0]] = 1
        acc += _SWAP(e, adj, rng.integers(0, E, n_try), rng.integers(0, E, n_try),
                     rng.integers(0, 2, n_try).astype(np.uint8)) / n_try
        out[r] = e
    d0 = np.bincount(e0.ravel(), minlength=n)
    assert all(np.array_equal(d0, np.bincount(out[r].ravel(), minlength=n)) for r in range(min(n_rep, 3)))
    return out, acc / max(n_rep, 1)


def adjacency_from_edges(e: np.ndarray, n: int) -> np.ndarray:
    a = np.zeros((n, n), dtype=np.float32)
    a[e[:, 0], e[:, 1]] = 1.0
    a[e[:, 1], e[:, 0]] = 1.0
    return a


def train_seed_store(state: DataState, embeddings: np.ndarray, cfg: Config, device: torch.device) -> List[dict]:
    store = []
    for fold in state.folds:
        logits1, logits2 = fold_text_logits(state, embeddings, fold, cfg, device)
        prof3, src3, pi3 = stage1_user_profiles(state, fold)
        prof7, src7, pi7 = stage2_user_profiles(state, fold.model_rows, fold.model_users, cfg)
        pieces = inner_alpha_pieces(state, embeddings, fold, cfg, device)
        val_all = np.where(state.df["user"].isin(fold.val_users).values)[0].astype(np.int64)
        store.append({"fold": fold, "logits1": logits1, "logits2": logits2, "prof3": prof3, "src3": src3,
                      "pi3": pi3, "prof7": prof7, "src7": src7, "pi7": pi7, "pieces": pieces, "val_all": val_all})
    return store


def stage_scores(state: DataState, fs: dict, rows: np.ndarray, P1, P2, P7, alpha: float):
    u = state.post_user_index[rows]
    L1 = fs["logits1"][rows]
    L2 = fs["logits2"][rows]
    lp1 = safe_log(P1[u])
    lp2 = safe_log(P2[u])
    lpi = safe_log(fs["pi3"])[None, :]
    z1 = {
        "text": L1,
        "prior": L1 + lpi + lpi,
        "hop1": L1 + lp1 + lpi,
        "hop2": L1 + lpi + lp2,
        "direct12": L1 + safe_log(P1[u]) + safe_log(P2[u]),
        "residual": L1 + (lp1 - lpi) + (lp2 - lpi),
        "net": lp1 + lp2,
    }
    s2 = {
        "text": L2.argmax(axis=1),
        "social": (L2 + alpha * (safe_log(P7[u]) - safe_log(fs["pi7"])[None, :])).argmax(axis=1),
        "net": safe_log(P7[u]).argmax(axis=1),
    }
    return z1, s2


def macro_f1_fast(y: np.ndarray, p: np.ndarray, n_cls: int) -> float:
    cm = np.bincount(y * n_cls + p, minlength=n_cls * n_cls).reshape(n_cls, n_cls).astype(float)
    tp = np.diag(cm)
    den = cm.sum(1) + cm.sum(0)
    return float(np.where(den > 0, 2 * tp / np.maximum(den, 1e-12), 0.0).mean())


def tune_bias(z1: np.ndarray, s2: np.ndarray, y9: np.ndarray, run: dict) -> np.ndarray:
    grid = np.linspace(-run["bias_range"], run["bias_range"], run["bias_steps"])
    best = (-1.0, 0.0, 0.0)
    for b1 in grid:
        for b2 in grid:
            y1 = (z1 + np.array([0.0, b1, b2])).argmax(axis=1)
            f = macro_f1_fast(y9, final9(y1, s2), 9)
            if f > best[0] + 1e-12 or (abs(f - best[0]) <= 1e-12 and b1 * b1 + b2 * b2 < best[1] ** 2 + best[2] ** 2):
                best = (f, b1, b2)
    return np.array([0.0, best[1], best[2]])


def inductive_adjacency(state, adjacency, fold):
    tu = np.array(sorted(state.user_index[u] for u in fold.test_users), dtype=np.int64)
    adj = adjacency.copy()
    adj[np.ix_(tu, tu)] = 0.0
    return adj, exact_two_hop(adj)


def evaluate_graph(state, stores, seeds, adjacency, adjacency2, systems, cfgs, run, calibrate, inductive=False):
    n = len(state.df)
    res = {}
    for s in seeds:
        cfg = cfgs[s]
        p9 = {k: np.full(n, -1, dtype=int) for k in systems}
        p1 = {k: np.full(n, -1, dtype=int) for k in systems}
        p2 = {"text": np.full(n, -1, dtype=int), "social": np.full(n, -1, dtype=int)}
        alphas, bias_log = [], []
        for fs in stores[s]:
            fold = fs["fold"]
            adj_f, adj2_f = inductive_adjacency(state, adjacency, fold) if inductive else (adjacency, adjacency2)
            P1 = graph_prior(adj_f, fs["prof3"], fs["src3"], fs["pi3"], cfg.stage1_prior_strength)
            P2 = graph_prior(adj2_f, fs["prof3"], fs["src3"], fs["pi3"], cfg.stage1_prior_strength)
            P7 = stage2_social_prior(state, fs["prof7"], fs["src7"], fs["pi7"], cfg, adj_f)
            alpha = alpha_from_pieces(state, fs["pieces"], adj_f, cfg)
            alphas.append(alpha)
            rows = fold.test_rows
            z1, s2 = stage_scores(state, fs, rows, P1, P2, P7, alpha)
            biases = {}
            if calibrate:
                vrows = fs["val_all"]
                z1v, s2v = stage_scores(state, fs, vrows, P1, P2, P7, alpha)
                for name in CALIBRATED & set(systems):
                    biases[name] = tune_bias(z1v[S1_RULE[name]], s2v[S2_RULE[name]], state.y_final9[vrows], run)
            bias_log.append(biases)
            for name in systems:
                z = z1[S1_RULE[name]]
                if name in biases:
                    z = z + biases[name]
                y1 = z.argmax(axis=1)
                p1[name][rows] = y1
                p9[name][rows] = final9(y1, s2[S2_RULE[name]])
            p2["text"][rows] = s2["text"]
            p2["social"][rows] = s2["social"]
        res[s] = {"p9": p9, "p1": p1, "p2": p2, "alphas": alphas, "biases": bias_log}
    return res


def user_confusion(uidx: np.ndarray, y9: np.ndarray, p: np.ndarray, n_users: int) -> np.ndarray:
    code = uidx * 81 + y9 * 9 + p
    return np.bincount(code, minlength=n_users * 81).reshape(n_users, 81).astype(np.float64)


def macro_f1_from_conf(C: np.ndarray) -> np.ndarray:
    M = C.reshape(-1, 9, 9)
    tp = np.einsum("bii->bi", M)
    den = M.sum(2) + M.sum(1)
    present = den > 0
    f1 = np.where(present, 2 * tp / np.maximum(den, 1e-12), 0.0)
    return (f1 * present).sum(1) / np.maximum(present.sum(1), 1)


def compare(res, seeds, a, b, state, Cboot) -> dict:
    y9 = state.y_final9
    uidx = state.post_user_index
    n_users = len(state.users)
    da = np.array([metric_values(y9, res[s]["p9"][a], 9)["macro_f1"] for s in seeds])
    db = np.array([metric_values(y9, res[s]["p9"][b], 9)["macro_f1"] for s in seeds])
    d = da - db
    fa = np.mean([macro_f1_from_conf(Cboot @ user_confusion(uidx, y9, res[s]["p9"][a], n_users)) for s in seeds], 0)
    fb = np.mean([macro_f1_from_conf(Cboot @ user_confusion(uidx, y9, res[s]["p9"][b], n_users)) for s in seeds], 0)
    bd = fa - fb
    lo, hi = np.percentile(bd, [2.5, 97.5])
    p = 2 * min((bd <= 0).mean(), (bd >= 0).mean())
    return {"diff_mean": float(d.mean()), "diff_sd": float(d.std(ddof=1)) if len(d) > 1 else 0.0,
            "seeds_positive": int((d > 0).sum()), "n_seeds": len(d), "ci_low": float(lo), "ci_high": float(hi),
            "p_boot": float(max(p, 1.0 / len(bd)))}


def fmt_p(p: float, B: int) -> str:
    return f"<{1.0 / B:.3f}" if p <= 1.0 / B + 1e-12 else f"{p:.3f}"


def main() -> None:
    run = parse_run_config()
    os.makedirs(run["out"], exist_ok=True)
    os.makedirs(run["cache_dir"], exist_ok=True)
    tee = Tee(os.path.join(run["out"], "report.txt"))
    sys.stdout = tee
    t0 = time.time()
    try:
        _main(run)
    finally:
        log(f"\nTotal time: {(time.time() - t0) / 60:.1f} min. Outputs in {run['out']}")
        sys.stdout = tee.out
        tee.close()


def _main(run: dict) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_cfg = Config()
    rng = np.random.default_rng(run["seed"])
    log("PHOND: ablation of the network terms on the pipeline of Experiments_P1.py")
    log(f"csv   : {run['csv']}\nedges : {run['edges']}\nout   : {run['out']}")
    log(f"device {device.type}, numba {'yes' if HAVE_NUMBA else 'no'}, encoders {run['encoders']}, "
        f"seeds {run['seeds']}, rewired graphs {run['n_rewire']}, bootstrap {run['n_boot']}")

    args = SimpleNamespace(data=run["csv"], edges=run["edges"], user_col=run["user_col"], text_col=run["text_col"],
                           stage1_col=run["stage1_col"], stage2_col=run["stage2_col"], edge_cols=run["edge_cols"])
    state = load_data(args, base_cfg)
    n_users = len(state.users)
    e_real = np.argwhere(np.triu(state.adjacency) > 0).astype(np.int32)
    class_names = ["normal", "offensive"] + [f"hate:{type_display(c)}" for c in state.stage2_classes]

    log("\n=== Data check ===")
    y1 = state.y_stage1
    log(f"  texts {len(state.df)}, users {n_users}, edges {len(e_real)}; normal {int((y1 == 0).sum())}, "
        f"offensive {int((y1 == 1).sum())}, hateful {int((y1 == 2).sum())}   (paper 35723, 599, 7626; 25273 / 9026 / 1424)")
    tc = pd.Series(state.df.loc[y1 == 2, "stage2"]).value_counts()
    log("  hate types: " + ", ".join(f"{type_display(t)} {c}" for t, c in tc.items()))
    for fold in state.folds:
        mr = fold.model_rows
        log(f"  fold {fold.fold}: {len(fold.model_users)} model users ({len(mr)} sampled texts, "
            f"{int((state.y_stage1[mr] == 2).sum())} hateful), {len(fold.val_users)} validation users, "
            f"{len(fold.test_users)} test users ({len(fold.test_rows)} texts)")

    rewired, acc = rewire_many(e_real, n_users, run["n_rewire"], run["swaps_per_edge"], rng)
    log(f"  {run['n_rewire']} degree preserving rewired graphs, swap acceptance {acc:.2f}")
    Cboot = rng.multinomial(n_users, np.full(n_users, 1.0 / n_users), size=run["n_boot"]).astype(np.float64)

    rows_seed, rows_sum, rows_cmp, rows_rew, rows_cls, rows_alpha = [], [], [], [], [], []
    preds_to_save = {"y9": state.y_final9, "user": np.asarray([norm_id(u) for u in state.df["user"]]).astype(str)}
    summary_lines = []
    hate_rows = np.where((state.y_stage1 == 2) & pd.notna(state.y_stage2))[0]
    y2_true = state.y_stage2[hate_rows].astype(int)

    for enc in run["encoders"]:
        model_id = run[f"enc_{enc}"]
        log(f"\n=== Encoder {enc} ({model_id}) ===")
        embeddings = encode_cached(state, model_id, base_cfg, device, run["cache_dir"])
        stores, cfgs = {}, {}
        for seed in run["seeds"]:
            t1 = time.time()
            cfg = Config(model_seed=int(seed))
            seed_everything(seed)
            cfgs[seed] = cfg
            stores[seed] = train_seed_store(state, embeddings, cfg, device)
            log(f"  seed {seed}: trained in {time.time() - t1:.0f}s")
        del embeddings
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        res = evaluate_graph(state, stores, run["seeds"], state.adjacency, state.adjacency_exact2, SYSTEMS, cfgs,
                             run, calibrate=True)
        res_ind = evaluate_graph(state, stores, run["seeds"], state.adjacency, state.adjacency_exact2, ["final"], cfgs,
                                 run, calibrate=False, inductive=True)
        for s in run["seeds"]:
            res[s]["p9"]["final_inductive"] = res_ind[s]["p9"]["final"]
            res[s]["p1"]["final_inductive"] = res_ind[s]["p1"]["final"]
            for fold_store, a in zip(stores[s], res[s]["alphas"]):
                rows_alpha.append({"encoder": enc, "seed": s, "fold": fold_store["fold"].fold, "alpha_social1": a})

        table = {}
        for name in REPORTED:
            vals = []
            for s in run["seeds"]:
                m9 = metric_values(state.y_final9, res[s]["p9"][name], 9)
                m1 = metric_values(state.y_stage1, res[s]["p1"][name], 3)
                vals.append((m9["macro_precision"], m9["macro_recall"], m9["accuracy"], m9["macro_f1"], m1["macro_f1"]))
                rows_seed.append({"encoder": enc, "seed": s, "system": name, "precision": vals[-1][0],
                                  "recall": vals[-1][1], "accuracy": vals[-1][2], "macro_f1": vals[-1][3],
                                  "stage1_macro_f1": vals[-1][4]})
            v = np.array(vals)
            mu = v.mean(0)
            sd = v.std(0, ddof=1) if len(vals) > 1 else np.zeros(5)
            table[name] = (mu, sd)
            rows_sum.append({"encoder": enc, "system": name, "precision": mu[0], "precision_sd": sd[0],
                             "recall": mu[1], "recall_sd": sd[1], "accuracy": mu[2], "accuracy_sd": sd[2],
                             "macro_f1": mu[3], "macro_f1_sd": sd[3], "stage1_macro_f1": mu[4],
                             "stage1_macro_f1_sd": sd[4]})
        s2t = np.mean([metric_values(y2_true, res[s]["p2"]["text"][hate_rows], len(state.stage2_classes))["macro_f1"]
                       for s in run["seeds"]])
        s2s = np.mean([metric_values(y2_true, res[s]["p2"]["social"][hate_rows], len(state.stage2_classes))["macro_f1"]
                       for s in run["seeds"]])
        alphas = np.array([res[s]["alphas"] for s in run["seeds"]])

        ref = PAPER.get(enc)
        got = {"final9_text": table["text"][0][3], "final9_final": table["final"][0][3],
               "stage1_text": table["text"][0][4], "stage1_direct12": table["direct12"][0][4],
               "stage2_text": s2t, "stage2_social": s2s}
        log("  reproduction check against Table 5 (five seed means):")
        for key, label in [("final9_text", "Final9 text"), ("final9_final", "Final9 text+network"),
                           ("stage1_text", "Stage 1 text"), ("stage1_direct12", "Stage 1 Eq. 2"),
                           ("stage2_text", "Stage 2 text"), ("stage2_social", "Stage 2 Eq. 4")]:
            if ref and model_id == ENCODERS.get(enc):
                log(f"    {label:<22}{got[key]:.4f}   Table 5 {ref[key]:.4f}   diff {got[key] - ref[key]:+.4f}")
            else:
                log(f"    {label:<22}{got[key]:.4f}")
        log(f"  alpha mean {alphas.mean():.2f} (range {alphas.min():.2f} to {alphas.max():.2f})")
        log(f"\n  {'system':<16}{'precision':>17}{'recall':>17}{'accuracy':>17}{'macro F1':>17}{'Stage 1 F1':>13}")
        for name in REPORTED:
            mu, sd = table[name]
            log(f"  {name:<16}" + "".join(f"{mu[i]:>10.3f} ± {sd[i]:.3f}" for i in range(4)) + f"{mu[4]:>13.3f}")

        log(f"\n  {'comparison':<28}{'diff':>8}{'sd':>7}{'seeds>0':>9}{'95% CI (users)':>20}{'p':>8}   meaning")
        for a, b, meaning in COMPARISONS:
            c = compare(res, run["seeds"], a, b, state, Cboot)
            rows_cmp.append({"encoder": enc, "comparison": f"{a} vs {b}", "meaning": meaning, **c})
            log(f"  {a + ' vs ' + b:<28}{c['diff_mean']:>+8.3f}{c['diff_sd']:>7.3f}{c['seeds_positive']:>5}/{c['n_seeds']:<3}"
                f"   [{c['ci_low']:+.3f}, {c['ci_high']:+.3f}]{fmt_p(c['p_boot'], run['n_boot']):>8}   {meaning}")

        rew_d12, rew_fin, rew_s1 = [], [], []
        for g in tqdm(range(run["n_rewire"]), desc=f"{enc} rewired graphs", mininterval=5):
            Ag = adjacency_from_edges(rewired[g], n_users)
            rg = evaluate_graph(state, stores, run["seeds"], Ag, exact_two_hop(Ag), ["direct12", "final"], cfgs,
                                run, calibrate=False)
            rew_d12.append(np.mean([metric_values(state.y_final9, rg[s]["p9"]["direct12"], 9)["macro_f1"]
                                    for s in run["seeds"]]))
            rew_fin.append(np.mean([metric_values(state.y_final9, rg[s]["p9"]["final"], 9)["macro_f1"]
                                    for s in run["seeds"]]))
            rew_s1.append(np.mean([metric_values(state.y_stage1, rg[s]["p1"]["direct12"], 3)["macro_f1"]
                                   for s in run["seeds"]]))
            rows_rew.append({"encoder": enc, "graph": g, "direct12_macro_f1": rew_d12[-1],
                             "final_macro_f1": rew_fin[-1], "direct12_stage1_f1": rew_s1[-1]})
        rew_d12, rew_fin, rew_s1 = map(np.array, (rew_d12, rew_fin, rew_s1))
        real_fin, real_d12 = table["final"][0][3], table["direct12"][0][3]
        p_fin = (1 + np.sum(rew_fin >= real_fin - 1e-12)) / (len(rew_fin) + 1)
        p_d12 = (1 + np.sum(rew_d12 >= real_d12 - 1e-12)) / (len(rew_d12) + 1)
        log(f"\n  rewired graphs ({len(rew_fin)}): final {rew_fin.mean():.3f} ± {rew_fin.std():.3f} vs real {real_fin:.3f} "
            f"(p {p_fin:.3f}); direct12 {rew_d12.mean():.3f} ± {rew_d12.std():.3f} vs real {real_d12:.3f} "
            f"(p {p_d12:.3f}); Stage 1 Eq. 2 {rew_s1.mean():.3f} vs real {table['direct12'][0][4]:.3f}")
        log(f"  for reference: text {table['text'][0][3]:.3f}, prior {table['prior'][0][3]:.3f}")

        log(f"\n  per class F1, mean over seeds   {'text':>8}{'final':>8}{'support':>9}")
        per = {}
        for name in ("text", "final"):
            P, R, F = [], [], []
            cm = np.zeros((9, 9))
            for s in run["seeds"]:
                m = metric_values(state.y_final9, res[s]["p9"][name], 9)
                P.append(m["precision"])
                R.append(m["recall"])
                F.append(m["f1"])
                sup = m["support"]
                cm += np.bincount(state.y_final9 * 9 + res[s]["p9"][name], minlength=81).reshape(9, 9)
            cm /= len(run["seeds"])
            per[name] = np.mean(F, 0)
            for i, cname in enumerate(class_names):
                rows_cls.append({"encoder": enc, "system": name, "class": cname, "precision": np.mean(P, 0)[i],
                                 "recall": np.mean(R, 0)[i], "f1": np.mean(F, 0)[i], "support": int(sup[i])})
            pd.DataFrame(cm, index=class_names, columns=class_names).round(1).to_csv(
                os.path.join(run["out"], f"confusion_{enc}_{name}.csv"))
        for i, cname in enumerate(class_names):
            log(f"  {cname:<30}{per['text'][i]:>8.3f}{per['final'][i]:>8.3f}{int(sup[i]):>9}")

        for name in REPORTED:
            for s in run["seeds"]:
                preds_to_save[f"{enc}__{name}__seed{s}"] = res[s]["p9"][name].astype(np.int8)

        cmp = {r["comparison"]: r for r in rows_cmp if r["encoder"] == enc}

        def line(a, b):
            c = cmp[f"{a} vs {b}"]
            return (f"{c['diff_mean']:+.3f} (CI {c['ci_low']:+.3f} to {c['ci_high']:+.3f}, "
                    f"{c['seeds_positive']}/{c['n_seeds']} seeds)")
        summary_lines += [
            f"  [{enc}] " + ", ".join(f"{k} {table[k][0][3]:.3f}" for k in REPORTED),
            f"  [{enc}] final vs text {line('final', 'text')}; prior vs text {line('prior', 'text')}; "
            f"direct12 vs prior {line('direct12', 'prior')}; final vs prior {line('final', 'prior')}",
            f"  [{enc}] calibrated: final_cal vs text_cal {line('final_cal', 'text_cal')}; "
            f"hops: hop1 vs prior {line('hop1', 'prior')}, hop2 vs prior {line('hop2', 'prior')}",
            f"  [{enc}] rewired: final {rew_fin.mean():.3f} ± {rew_fin.std():.3f} vs real {real_fin:.3f} "
            f"(p {p_fin:.3f} over {len(rew_fin)} graphs)",
            f"  [{enc}] inductive (no test-test edges): final_inductive vs text {line('final_inductive', 'text')}; "
            f"vs transductive final {line('final_inductive', 'final')}",
        ]
        out = run["out"]
        pd.DataFrame(rows_seed).to_csv(os.path.join(out, "results_by_seed.csv"), index=False)
        pd.DataFrame(rows_sum).to_csv(os.path.join(out, "results_summary.csv"), index=False)
        pd.DataFrame(rows_cmp).to_csv(os.path.join(out, "comparisons.csv"), index=False)
        pd.DataFrame(rows_rew).to_csv(os.path.join(out, "rewired.csv"), index=False)
        pd.DataFrame(rows_cls).to_csv(os.path.join(out, "per_class.csv"), index=False)
        pd.DataFrame(rows_alpha).to_csv(os.path.join(out, "alpha_by_fold.csv"), index=False)
        np.savez_compressed(os.path.join(out, "oof_predictions.npz"), **preds_to_save)

    log("\n=== Summary ===")
    log("  Metrics pool the out of fold predictions of all texts per seed; mean over seeds. Differences are paired")
    log("  per seed; the CI resamples users with replacement (user level bootstrap) and averages over seeds.")
    for s_line in summary_lines:
        log(s_line)


if __name__ == "__main__":
    main()
