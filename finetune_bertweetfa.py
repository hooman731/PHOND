#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import contextlib
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
try:
    from transformers.utils import logging as _hf_logging
    _hf_logging.set_verbosity_error()
except Exception:
    pass

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
            alpha = fs["alpha"]
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


CONFIG = {
    "csv": "phond.csv",
    "edges": "edges.csv",
    "out": "finetune_bertweetfa_out",
    "cache_dir": "",
    "user_col": "User_SId",
    "text_col": "Text",
    "stage1_col": "classification",
    "stage2_col": "hate_class",
    "edge_cols": "",
    "encoders": "bertweetfa",
    "enc_bertweetfa": "arm-on/BERTweet-FA",
    "enc_parsbert": ENCODERS["parsbert"],
    "enc_xlmr": ENCODERS["xlmr"],
    "enc_mbert": ENCODERS["mbert"],
    "seeds": "47",
    "epochs": 2,
    "train_layers": 0,
    "batch": 32,
    "lr_encoder": 2e-5,
    "lr_head": 5e-4,
    "warmup": 0.1,
    "eval_batch": 128,
    "alpha": 1.0,
    "n_rewire": 200,
    "swaps_per_edge": 10,
    "n_boot": 1000,
    "bias_range": 3.0,
    "bias_steps": 25,
    "seed": 2026,
    "quick": 0,
    "quick_rows": 1500,
}

FROZEN = {
    "mbert": {"text": 0.292, "text_cal": 0.326, "prior": 0.320, "hop1": 0.334, "hop2": 0.333, "direct12": 0.348,
              "final": 0.353, "residual": 0.324, "final_cal": 0.345, "net_only": 0.135, "rewired": 0.328},
    "xlmr": {"text": 0.306, "text_cal": 0.341, "prior": 0.338, "hop1": 0.354, "hop2": 0.346, "direct12": 0.360,
             "final": 0.369, "residual": 0.344, "final_cal": 0.368, "net_only": 0.135, "rewired": 0.324},
    "parsbert": {"text": 0.364, "text_cal": 0.393, "prior": 0.385, "hop1": 0.402, "hop2": 0.399, "direct12": 0.413,
                 "final": 0.419, "residual": 0.397, "final_cal": 0.420, "net_only": 0.135, "rewired": 0.400},
}


def parse_run_config() -> dict:
    ap = argparse.ArgumentParser(description="PHOND: fine-tuned BERTweet-FA with the network features")
    for k, v in CONFIG.items():
        ap.add_argument(f"--{k}", type=type(v), default=v)
    args, _unknown = ap.parse_known_args()
    run = vars(args)
    run["seeds"] = [int(s) for s in str(run["seeds"]).split(",") if s.strip()]
    run["encoders"] = [s.strip().lower() for s in str(run["encoders"]).split(",") if s.strip()]
    for enc in run["encoders"]:
        if f"enc_{enc}" not in run:
            sys.exit(f"[error] unknown encoder '{enc}'. Use bertweetfa, parsbert, xlmr, mbert.")
    if run["quick"]:
        run["seeds"] = run["seeds"][:1]
        run["epochs"] = 1
        run["n_rewire"] = min(run["n_rewire"], 10)
        run["n_boot"] = min(run["n_boot"], 200)
        run["bias_steps"] = min(run["bias_steps"], 13)
    if not run["cache_dir"]:
        run["cache_dir"] = os.path.join(run["out"], "ft_cache")
    for key in ("csv", "edges"):
        if not os.path.exists(run[key]):
            sys.exit(f"[error] file not found: {run[key]}  (pass it with --{key} \"...\")")
    return run


class FineTuneNet(nn.Module):
    def __init__(self, model_id: str, n_stage2: int, cfg: Config, train_layers: int = 0):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_id)
        if train_layers > 0:
            layers = self.encoder.encoder.layer
            for prm in self.encoder.parameters():
                prm.requires_grad = False
            for layer in layers[len(layers) - min(train_layers, len(layers)):]:
                for prm in layer.parameters():
                    prm.requires_grad = True
        d = int(self.encoder.config.hidden_size)
        self.head1 = TextClassifier(d, 3, cfg)
        self.head2 = TextClassifier(d, n_stage2, cfg)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        h = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0, :]
        return self.head1(h), self.head2(h)


def amp_ctx(device: torch.device, use_amp: bool):
    if use_amp:
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return contextlib.nullcontext()


def make_batch(ids_list: List[List[int]], rows: np.ndarray, pad_id: int, device: torch.device):
    seqs = [ids_list[int(r)] for r in rows]
    width = max(len(s) for s in seqs)
    x = np.full((len(seqs), width), pad_id, dtype=np.int64)
    m = np.zeros((len(seqs), width), dtype=np.int64)
    for i, s in enumerate(seqs):
        x[i, :len(s)] = s
        m[i, :len(s)] = 1
    return torch.from_numpy(x).to(device), torch.from_numpy(m).to(device)


@torch.no_grad()
def predict_ft(net, ids_list, rows, pad_id, device, batch_size, use_amp, n2) -> Tuple[np.ndarray, np.ndarray]:
    net.eval()
    rows = np.asarray(rows, dtype=np.int64)
    order = np.argsort(np.array([len(ids_list[int(r)]) for r in rows]), kind="stable")
    out1 = np.zeros((len(rows), 3), dtype=np.float32)
    out2 = np.zeros((len(rows), n2), dtype=np.float32)
    for start in range(0, len(order), batch_size):
        sel = order[start:start + batch_size]
        x, m = make_batch(ids_list, rows[sel], pad_id, device)
        with amp_ctx(device, use_amp):
            l1, l2 = net(x, m)
        out1[sel] = l1.float().cpu().numpy()
        out2[sel] = l2.float().cpu().numpy()
    return out1, out2


def fine_tune_fold(state: DataState, fold: Fold, ids_list, pad_id: int, model_id: str, cfg: Config, run: dict,
                   device: torch.device, seed: int, log_rows: list, tag: str) -> dict:
    n2 = len(state.stage2_classes)
    y1 = state.y_stage1.astype(np.int64)
    hate = (state.y_stage1 == 2) & pd.notna(state.y_stage2)
    y2 = np.where(hate, np.nan_to_num(state.y_stage2.astype(float), nan=0.0), -100).astype(np.int64)

    train = np.asarray(fold.model_rows, dtype=np.int64)
    if run["quick"] and len(train) > run["quick_rows"]:
        train = np.sort(np.random.RandomState(seed).choice(train, run["quick_rows"], replace=False))
    tr2 = train[hate[train]]
    va = np.asarray(fold.val_rows, dtype=np.int64)
    va2 = va[hate[va]]
    val_all = np.where(state.df["user"].isin(fold.val_users).values)[0].astype(np.int64)
    eval_rows = np.unique(np.concatenate([fold.test_rows, val_all])).astype(np.int64)
    va_pos = np.searchsorted(eval_rows, va)
    va2_pos = np.searchsorted(eval_rows, va2)
    if len(tr2) < 10:
        raise RuntimeError(f"Fold {fold.fold}: insufficient Stage-2 training rows.")

    seed_everything(seed)
    net = FineTuneNet(model_id, n2, cfg, int(run["train_layers"])).to(device)
    crit1 = nn.CrossEntropyLoss(weight=class_weights(y1[train], 3, device, cfg.hateful_class_weight_boost))
    crit2 = nn.CrossEntropyLoss(weight=class_weights(y2[tr2], n2, device))
    enc_params = [p for n, p in net.named_parameters() if n.startswith("encoder.") and p.requires_grad]
    head_params = [p for n, p in net.named_parameters() if not n.startswith("encoder.")]
    opt = torch.optim.AdamW([{"params": enc_params, "lr": run["lr_encoder"], "weight_decay": 0.01},
                             {"params": head_params, "lr": run["lr_head"], "weight_decay": cfg.weight_decay}])
    if fold.fold == 1:
        n_enc = sum(p.numel() for p in net.encoder.parameters())
        log(f"    trainable encoder parameters: {sum(p.numel() for p in enc_params):,} of {n_enc:,}")
    bs = int(run["batch"])
    total = max(1, int(run["epochs"]) * math.ceil(len(train) / bs))
    warm = max(1, int(run["warmup"] * total))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else max(0.0, (total - s) / max(total - warm, 1)))
    use_amp = device.type == "cuda"
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    except (AttributeError, TypeError):
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    rng = np.random.RandomState(seed + 77)

    best = {1: (-math.inf, None, 0), 2: (-math.inf, None, 0)}
    for epoch in range(1, int(run["epochs"]) + 1):
        t0 = time.time()
        net.train()
        perm = rng.permutation(train)
        loss_sum, n_batches = 0.0, 0
        for start in tqdm(range(0, len(perm), bs), desc=f"  fold {fold.fold} epoch {epoch}", mininterval=10,
                          leave=False):
            idx = perm[start:start + bs]
            if len(idx) < 2:
                continue
            x, m = make_batch(ids_list, idx, pad_id, device)
            t1 = torch.from_numpy(y1[idx]).to(device)
            t2 = torch.from_numpy(y2[idx]).to(device)
            with amp_ctx(device, use_amp):
                l1, l2 = net(x, m)
            loss = crit1(l1.float(), t1)
            mask = t2 >= 0
            if bool(mask.any()):
                loss = loss + crit2(l2.float()[mask], t2[mask])
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(net.parameters(), cfg.grad_clip)
            scale_before = scaler.get_scale() if use_amp else 1.0
            scaler.step(opt)
            scaler.update()
            if not use_amp or scaler.get_scale() >= scale_before:
                sched.step()
            loss_sum += float(loss.item())
            n_batches += 1

        L1, L2 = predict_ft(net, ids_list, eval_rows, pad_id, device, int(run["eval_batch"]), use_amp, n2)
        f1_val = f1_score(y1[va], L1[va_pos].argmax(axis=1), average="macro", labels=[0, 1, 2], zero_division=0)
        f2_val = (f1_score(y2[va2], L2[va2_pos].argmax(axis=1), average="macro", labels=list(range(n2)),
                           zero_division=0) if len(va2) else 0.0)
        if f1_val > best[1][0] + 1e-5:
            best[1] = (f1_val, L1, epoch)
        if f2_val > best[2][0] + 1e-5:
            best[2] = (f2_val, L2, epoch)
        log_rows.append({"run": tag, "fold": fold.fold, "epoch": epoch, "train_texts": len(train),
                         "train_loss": loss_sum / max(n_batches, 1), "val_stage1_macro_f1": f1_val,
                         "val_stage2_macro_f1": f2_val, "seconds": round(time.time() - t0, 1)})
        log(f"    fold {fold.fold} epoch {epoch}: loss {loss_sum / max(n_batches, 1):.3f}, validation macro F1 "
            f"Stage 1 {f1_val:.3f}, Stage 2 {f2_val:.3f} ({time.time() - t0:.0f}s)")

    del net, opt, scaler
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {"rows": eval_rows, "l1": best[1][1], "l2": best[2][1], "e1": best[1][2], "e2": best[2][2],
            "v1": best[1][0], "v2": best[2][0]}


def removed_text(real_fin: float, rew_mean: float, real_prior: float) -> str:
    gain = real_fin - real_prior
    if gain <= 1e-9:
        return "no gain over the prior for rewiring to remove"
    share = (real_fin - rew_mean) / gain
    if share >= 1:
        return "rewiring removes all of the gain over the prior"
    return f"rewiring removes {share:.0%} of the gain over the prior"


def layers_text(run: dict) -> str:
    k = int(run["train_layers"])
    return "every encoder layer trained" if k <= 0 else f"only the top {k} encoder layer(s) trained"


def data_hash(state: DataState) -> str:
    h = hashlib.md5()
    for col in ("user", "text", "stage1", "stage2"):
        h.update(("\n".join(str(v) for v in state.df[col].tolist()) + "|").encode("utf-8", "replace"))
    return h.hexdigest()[:10]


def train_seed_store_ft(state: DataState, ids_list, pad_id: int, model_id: str, enc: str, cfg: Config, run: dict,
                        device: torch.device, dhash: str, log_rows: list) -> List[dict]:
    n = len(state.df)
    n2 = len(state.stage2_classes)
    settings = (f"{model_id}|{cfg.max_length}|{run['epochs']}|{run['batch']}|{run['lr_encoder']}|{run['lr_head']}|"
                f"{run['warmup']}|{run['quick']}|{run['quick_rows']}|{cfg.split_seed}|{dhash}|L{run['train_layers']}")
    tag = hashlib.md5(settings.encode()).hexdigest()[:8]
    store = []
    for fold in state.folds:
        seed = cfg.model_seed * 100 + fold.fold
        path = os.path.join(run["cache_dir"], f"ft_{enc}_seed{cfg.model_seed}_fold{fold.fold}_{tag}.npz")
        if os.path.exists(path):
            z = np.load(path)
            out = {k: z[k] for k in z.files}
            out["e1"], out["e2"] = int(out["e1"]), int(out["e2"])
            log(f"    fold {fold.fold}: loaded the fine-tuned logits from {path}")
        else:
            out = fine_tune_fold(state, fold, ids_list, pad_id, model_id, cfg, run, device, seed, log_rows,
                                 f"{enc}_seed{cfg.model_seed}")
            np.savez_compressed(path, **out)
        L1 = np.zeros((n, 3), dtype=np.float32)
        L2 = np.zeros((n, n2), dtype=np.float32)
        L1[out["rows"]] = out["l1"]
        L2[out["rows"]] = out["l2"]
        prof3, src3, pi3 = stage1_user_profiles(state, fold)
        prof7, src7, pi7 = stage2_user_profiles(state, fold.model_rows, fold.model_users, cfg)
        val_all = np.where(state.df["user"].isin(fold.val_users).values)[0].astype(np.int64)
        store.append({"fold": fold, "logits1": L1, "logits2": L2, "prof3": prof3, "src3": src3, "pi3": pi3,
                      "prof7": prof7, "src7": src7, "pi7": pi7, "alpha": float(run["alpha"]), "val_all": val_all,
                      "epochs": (out["e1"], out["e2"])})
    return store


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
        print(f"Results: {bundle_results(run['out'])}", flush=True)


def bundle_results(out: str) -> str:
    import zipfile
    path = os.path.join(out, "phond_bertweetfa_results.zip")
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        for name in sorted(os.listdir(out)):
            if name.endswith((".txt", ".csv")):
                z.write(os.path.join(out, name), name)
    return path


def _main(run: dict) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_cfg = Config()
    rng = np.random.default_rng(run["seed"])
    log("PHOND: fine-tuned BERTweet-FA with the network features")
    log(f"csv   : {run['csv']}\nedges : {run['edges']}\nout   : {run['out']}")
    log(f"device {device.type}, numba {'yes' if HAVE_NUMBA else 'no'}, encoders {run['encoders']}, "
        f"seeds {run['seeds']}, rewired graphs {run['n_rewire']}, bootstrap {run['n_boot']}")
    log(f"fine-tuning: {run['epochs']} epoch(s), batch {run['batch']}, lr encoder {run['lr_encoder']:g}, "
        f"lr heads {run['lr_head']:g}, warm-up {run['warmup']:g}, max length {base_cfg.max_length}, "
        f"{layers_text(run)}, "
        f"alpha fixed at {run['alpha']:g}" + (f", QUICK CHECK ({run['quick_rows']} texts per fold)" if run["quick"] else ""))
    if device.type != "cuda":
        log("[warning] no GPU found. Fine-tuning on a CPU takes many hours. In Colab: Runtime > Change runtime "
            "type > T4 GPU, then run again.")

    args = SimpleNamespace(data=run["csv"], edges=run["edges"], user_col=run["user_col"], text_col=run["text_col"],
                           stage1_col=run["stage1_col"], stage2_col=run["stage2_col"], edge_cols=run["edge_cols"])
    state = load_data(args, base_cfg)
    n_users = len(state.users)
    e_real = np.argwhere(np.triu(state.adjacency) > 0).astype(np.int32)
    class_names = ["normal", "offensive"] + [f"hate:{type_display(c)}" for c in state.stage2_classes]
    dhash = data_hash(state)

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

    rows_seed, rows_sum, rows_cmp, rows_rew, rows_cls, log_rows = [], [], [], [], [], []
    preds_to_save = {"y9": state.y_final9, "user": np.asarray([norm_id(u) for u in state.df["user"]]).astype(str)}
    summary_lines = []
    hate_rows = np.where((state.y_stage1 == 2) & pd.notna(state.y_stage2))[0]
    y2_true = state.y_stage2[hate_rows].astype(int)
    texts = [str(v) for v in state.df["text"].tolist()]

    for enc in run["encoders"]:
        model_id = run[f"enc_{enc}"]
        frozen = FROZEN.get(enc, {}) if model_id == ENCODERS.get(enc) else {}
        log(f"\n=== Encoder {enc} ({model_id}), fine-tuning ===")
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        from transformers import AutoConfig
        ecfg = AutoConfig.from_pretrained(model_id)
        log(f"  encoder config: {getattr(ecfg, 'model_type', '?')}, {getattr(ecfg, 'num_hidden_layers', '?')} layers, "
            f"hidden {getattr(ecfg, 'hidden_size', '?')}, vocabulary {getattr(ecfg, 'vocab_size', '?')}, "
            f"max positions {getattr(ecfg, 'max_position_embeddings', '?')}; tokenizer {type(tokenizer).__name__}")
        max_pos = int(getattr(ecfg, "max_position_embeddings", 512) or 512)
        if max_pos < base_cfg.max_length + 2:
            sys.exit(f"[error] {model_id} accepts {max_pos} positions, fewer than the max length of {base_cfg.max_length}.")
        if tokenizer.cls_token_id is None:
            sys.exit(f"[error] the tokenizer of {model_id} has no first (CLS) token, which the heads read.")
        ids_list = tokenizer(texts, truncation=True, max_length=base_cfg.max_length)["input_ids"]
        n_tok = np.asarray([len(x) for x in ids_list])
        unk = tokenizer.unk_token_id
        unk_share = float(np.mean([t == unk for x in ids_list for t in x])) if unk is not None else 0.0
        log(f"  tokens per text: median {int(np.median(n_tok))}, 95th percentile {int(np.percentile(n_tok, 95))}, "
            f"cut at {base_cfg.max_length}: {int((n_tok >= base_cfg.max_length).sum())} texts; unknown tokens {unk_share:.2%}")
        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        stores, cfgs, kept = {}, {}, {}
        for seed in run["seeds"]:
            t1 = time.time()
            cfg = Config(model_seed=int(seed))
            cfgs[seed] = cfg
            stores[seed] = train_seed_store_ft(state, ids_list, pad_id, model_id, enc, cfg, run, device, dhash, log_rows)
            kept[seed] = [fs["epochs"] for fs in stores[seed]]
            log(f"  seed {seed}: done in {(time.time() - t1) / 60:.1f} min; epoch kept per fold "
                f"(Stage 1, Stage 2): {kept[seed]}")
            pd.DataFrame(log_rows).to_csv(os.path.join(run["out"], "finetune_log.csv"), index=False)

        res = evaluate_graph(state, stores, run["seeds"], state.adjacency, state.adjacency_exact2, SYSTEMS, cfgs,
                             run, calibrate=True)
        res_ind = evaluate_graph(state, stores, run["seeds"], state.adjacency, state.adjacency_exact2, ["final"], cfgs,
                                 run, calibrate=False, inductive=True)
        for s in run["seeds"]:
            res[s]["p9"]["final_inductive"] = res_ind[s]["p9"]["final"]
            res[s]["p1"]["final_inductive"] = res_ind[s]["p1"]["final"]

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
                             "stage1_macro_f1_sd": sd[4], "frozen_macro_f1_table6": frozen.get(name, np.nan)})
        s2t = np.mean([metric_values(y2_true, res[s]["p2"]["text"][hate_rows], len(state.stage2_classes))["macro_f1"]
                       for s in run["seeds"]])
        s2s = np.mean([metric_values(y2_true, res[s]["p2"]["social"][hate_rows], len(state.stage2_classes))["macro_f1"]
                       for s in run["seeds"]])
        log(f"\n  Stage 1 macro F1: text {table['text'][0][4]:.3f}, Eq. 2 {table['direct12'][0][4]:.3f}; "
            f"Stage 2 macro F1 on the hateful texts: text {s2t:.3f}, Eq. 4 {s2s:.3f}")
        log(f"\n  {'system':<16}{'precision':>17}{'recall':>17}{'accuracy':>17}{'macro F1':>17}{'Stage 1 F1':>13}"
            f"{'frozen F1':>11}")
        for name in REPORTED:
            mu, sd = table[name]
            fz = f"{frozen[name]:>11.3f}" if name in frozen else f"{'':>11}"
            log(f"  {name:<16}" + "".join(f"{mu[i]:>10.3f} ± {sd[i]:.3f}" for i in range(4)) + f"{mu[4]:>13.3f}" + fz)
        if frozen:
            log("  frozen F1 = Table 6 (frozen encoder, mean over five seeds)")

        log(f"\n  {'comparison':<28}{'diff':>8}{'sd':>7}{'seeds>0':>9}{'95% CI (users)':>20}{'p':>8}   meaning")
        for a, b, meaning in COMPARISONS:
            c = compare(res, run["seeds"], a, b, state, Cboot)
            rows_cmp.append({"encoder": enc, "comparison": f"{a} vs {b}", "meaning": meaning, **c})
            log(f"  {a + ' vs ' + b:<28}{c['diff_mean']:>+8.3f}{c['diff_sd']:>7.3f}{c['seeds_positive']:>5}/{c['n_seeds']:<3}"
                f"   [{c['ci_low']:+.3f}, {c['ci_high']:+.3f}]{fmt_p(c['p_boot'], run['n_boot']):>8}   {meaning}")

        rew_d12, rew_fin = [], []
        for g in tqdm(range(run["n_rewire"]), desc=f"{enc} rewired graphs", mininterval=5):
            Ag = adjacency_from_edges(rewired[g], n_users)
            rg = evaluate_graph(state, stores, run["seeds"], Ag, exact_two_hop(Ag), ["direct12", "final"], cfgs,
                                run, calibrate=False)
            rew_d12.append(np.mean([metric_values(state.y_final9, rg[s]["p9"]["direct12"], 9)["macro_f1"]
                                    for s in run["seeds"]]))
            rew_fin.append(np.mean([metric_values(state.y_final9, rg[s]["p9"]["final"], 9)["macro_f1"]
                                    for s in run["seeds"]]))
            rows_rew.append({"encoder": enc, "graph": g, "direct12_macro_f1": rew_d12[-1],
                             "final_macro_f1": rew_fin[-1]})
        rew_d12, rew_fin = np.array(rew_d12), np.array(rew_fin)
        real_fin = table["final"][0][3]
        real_prior = table["prior"][0][3]
        p_fin = (1 + np.sum(rew_fin >= real_fin - 1e-12)) / (len(rew_fin) + 1)
        removed = removed_text(real_fin, rew_fin.mean(), real_prior)
        log(f"\n  rewired graphs ({len(rew_fin)}): final {rew_fin.mean():.3f} ± {rew_fin.std():.3f} vs real "
            f"{real_fin:.3f} (p {p_fin:.3f}); prior {real_prior:.3f}; {removed}"
            + (f"; frozen rewired {frozen['rewired']:.3f}" if frozen else ""))

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
                os.path.join(run["out"], f"confusion_{enc}_finetuned_{name}.csv"))
        for i, cname in enumerate(class_names):
            log(f"  {cname:<30}{per['text'][i]:>8.3f}{per['final'][i]:>8.3f}{int(sup[i]):>9}")

        for name in REPORTED:
            for s in run["seeds"]:
                preds_to_save[f"{enc}__{name}__seed{s}"] = res[s]["p9"][name].astype(np.int8)

        cmp = {r["comparison"]: r for r in rows_cmp if r["encoder"] == enc}

        def line(a, b):
            c = cmp[f"{a} vs {b}"]
            return (f"{c['diff_mean']:+.3f} (95% CI {c['ci_low']:+.3f} to {c['ci_high']:+.3f}, p "
                    f"{fmt_p(c['p_boot'], run['n_boot'])})")

        def fz(name):
            return f" (frozen {frozen[name]:.3f})" if name in frozen else ""

        summary_lines += [
            f"  [{enc}] fine-tuning, {run['epochs']} epoch(s), {layers_text(run)}, seed(s) {run['seeds']}. "
            f"Final9 macro F1: "
            f"text {table['text'][0][3]:.3f}{fz('text')}, prior {table['prior'][0][3]:.3f}{fz('prior')}, "
            f"text+network {table['final'][0][3]:.3f}{fz('final')}",
            f"  [{enc}] text+network vs text {line('final', 'text')}",
            f"  [{enc}] prior shift alone {line('prior', 'text')}; network beyond the prior {line('final', 'prior')}",
            f"  [{enc}] both calibrated: {table['final_cal'][0][3]:.3f} vs {table['text_cal'][0][3]:.3f}, "
            f"{line('final_cal', 'text_cal')}",
            f"  [{enc}] rewired: final {rew_fin.mean():.3f} ± {rew_fin.std():.3f} vs real {real_fin:.3f} "
            f"(p {p_fin:.3f} over {len(rew_fin)} graphs); {removed}",
            f"  [{enc}] Stage 1 macro F1 text {table['text'][0][4]:.3f}; Stage 2 macro F1 text {s2t:.3f}, Eq. 4 {s2s:.3f}; "
            f"epoch kept per fold (Stage 1, Stage 2): {kept}",
        ]
        out = run["out"]
        pd.DataFrame(rows_seed).to_csv(os.path.join(out, "results_by_seed.csv"), index=False)
        pd.DataFrame(rows_sum).to_csv(os.path.join(out, "results_summary.csv"), index=False)
        pd.DataFrame(rows_cmp).to_csv(os.path.join(out, "comparisons.csv"), index=False)
        pd.DataFrame(rows_rew).to_csv(os.path.join(out, "rewired.csv"), index=False)
        pd.DataFrame(rows_cls).to_csv(os.path.join(out, "per_class.csv"), index=False)
        pd.DataFrame(log_rows).to_csv(os.path.join(out, "finetune_log.csv"), index=False)
        np.savez_compressed(os.path.join(out, "oof_predictions.npz"), **preds_to_save)

    log("\n=== Summary ===")
    log("  Metrics pool the out of fold predictions of all texts. Differences are paired on the same fine-tuned")
    log("  text model; the CI resamples users with replacement (user level bootstrap). Frozen numbers: Table 6.")
    for s_line in summary_lines:
        log(s_line)


if __name__ == "__main__":
    main()
