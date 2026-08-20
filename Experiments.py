#!/usr/bin/env python3

from __future__ import annotations

import copy
import gc
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Set, Tuple
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.optimize import minimize
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer


ENCODERS = {
    "mbert": "bert-base-multilingual-cased",
    "xlmr": "FacebookAI/xlm-roberta-base",
    "parsbert": "HooshvareLab/bert-base-parsbert-uncased",
}

COARSE_CLASSES = ["normal", "offensive", "hateful"]

DATA_PATH = "/content/phond_PaperVersion.csv"
EDGES_PATH = "/content/User_Edges (1).csv"
OUTPUT_DIR = "/content/phond_final_paper_results"

USER_COLUMN = "User_SId"
TEXT_COLUMN = "Text"
STAGE1_COLUMN = "classification"
STAGE2_COLUMN = "hate_class"
EDGE_SOURCE_COLUMN = "source"
EDGE_TARGET_COLUMN = "target"
TIME_COLUMN = None

NORMAL_LABEL = "عادی"
OFFENSIVE_LABEL = "توهین‌آمیز"
HATEFUL_LABEL = "تنفرآمیز"

RUN_ENCODERS = ["mbert", "xlmr", "parsbert"]
MODEL_SEEDS = [11, 29, 47, 71, 101]


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


def load_data(args: SimpleNamespace, cfg: Config) -> DataState:
    df = pd.read_csv(args.data, encoding="utf-8-sig")
    required = [args.user_col, args.text_col, args.stage1_col, args.stage2_col]
    if args.time_col:
        required.append(args.time_col)
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing data columns: {missing}")

    rename = {
        args.user_col: "user",
        args.text_col: "text",
        args.stage1_col: "stage1",
        args.stage2_col: "stage2",
    }
    if args.time_col:
        rename[args.time_col] = "time"
    df = df.rename(columns=rename).copy()

    stage1_map = {
        str(args.normal_label).strip(): "normal",
        str(args.offensive_label).strip(): "offensive",
        str(args.hateful_label).strip(): "hateful",
    }
    raw_stage1 = df["stage1"].astype(str).str.strip()
    invalid = raw_stage1[~raw_stage1.isin(stage1_map)]
    if len(invalid):
        raise ValueError("Unexpected Stage-1 labels:\n" + invalid.value_counts().to_string())
    df["stage1"] = raw_stage1.map(stage1_map)
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

    if args.time_col:
        df = df.sort_values(["user", "time"]).reset_index(drop=True)
    else:
        df = df.reset_index(drop=True)
    df["row_id"] = np.arange(len(df), dtype=np.int64)

    users = sorted(df["user"].unique(), key=str)
    user_index = {u: i for i, u in enumerate(users)}
    user_posts = {u: g.index.values.astype(np.int64) for u, g in df.groupby("user")}

    edges = pd.read_csv(args.edges, encoding="utf-8-sig")
    missing_edge_cols = [c for c in [args.edge_source_col, args.edge_target_col] if c not in edges.columns]
    if missing_edge_cols:
        raise ValueError(f"Missing edge columns: {missing_edge_cols}")

    edge_set = set()
    for s, t in zip(edges[args.edge_source_col], edges[args.edge_target_col]):
        if s not in user_index or t not in user_index:
            continue
        a, b = user_index[s], user_index[t]
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

    def dominant_user_class(user) -> int:
        values = df.loc[df["user"] == user, "stage1_id"].values.astype(int)
        return int(np.bincount(values, minlength=3).argmax())

    def sample_user_posts(user) -> np.ndarray:
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


def stage2_social_prior(
    state: DataState,
    profiles: np.ndarray,
    source_mask: np.ndarray,
    global_prior: np.ndarray,
    cfg: Config,
) -> np.ndarray:
    weights = state.adjacency * source_mask[None, :].astype(np.float32)
    counts = weights.sum(axis=1, keepdims=True)
    sums = weights @ profiles
    p = (
        sums + cfg.stage2_social_strength * global_prior[None, :]
    ) / np.maximum(counts + cfg.stage2_social_strength, 1e-8)
    return normalize_probs(p, global_prior)


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


def estimate_stage2_alpha(
    state: DataState,
    embeddings: np.ndarray,
    fold: Fold,
    cfg: Config,
    device: torch.device,
) -> float:
    users = np.array(sorted(fold.model_users, key=str), dtype=object)
    strata = np.array([user_has_hateful(state, u) for u in users], dtype=int)
    n_splits = cfg.inner_alpha_folds
    if len(np.unique(strata)) > 1:
        n_splits = min(n_splits, max(2, int(np.bincount(strata).min())))
    n_splits = min(n_splits, len(users))
    if n_splits < 2:
        return 1.0

    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=cfg.model_seed * 1000 + fold.fold)
    try:
        splits = list(splitter.split(users, strata))
    except Exception:
        splits = list(splitter.split(users, np.zeros(len(users), dtype=int)))

    all_text, all_residual, all_y = [], [], []
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
        p_social = stage2_social_prior(state, profiles, source_mask, pi7, cfg)
        residual = safe_log(p_social[state.post_user_index[held_h]]) - safe_log(pi7)[None, :]
        all_text.append(text_logits)
        all_residual.append(residual.astype(np.float32))
        all_y.append(state.y_stage2[held_h].astype(int))

    if not all_y:
        return 1.0

    return fit_alpha(
        np.concatenate(all_text),
        np.concatenate(all_residual),
        np.concatenate(all_y),
        cfg,
        len(state.stage2_classes),
    )


def final9(stage1_pred: np.ndarray, stage2_pred: np.ndarray) -> np.ndarray:
    out = np.asarray(stage1_pred, dtype=int).copy()
    hateful = stage1_pred == 2
    out[hateful] = np.asarray(stage2_pred, dtype=int)[hateful] + 2
    return out


def add_predictions(
    rows: List[dict],
    encoder: str,
    fold: int,
    arm: str,
    stage: str,
    row_ids: np.ndarray,
    users: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> None:
    for i, row_id in enumerate(row_ids.tolist()):
        rows.append({
            "encoder": encoder,
            "fold": int(fold),
            "arm": arm,
            "stage": stage,
            "row_id": int(row_id),
            "user": users[i],
            "y_true": int(y_true[i]),
            "y_pred": int(y_pred[i]),
        })


def stage_class_names(state: DataState, stage: str) -> List[str]:
    if stage == "stage1":
        return COARSE_CLASSES
    if stage == "stage2":
        return state.stage2_classes
    return state.final_classes


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


def summarize(state: DataState, predictions: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pooled_rows, fold_rows, class_rows = [], [], []

    for (encoder, arm, stage), g in predictions.groupby(["encoder", "arm", "stage"]):
        names = stage_class_names(state, stage)
        m = metric_values(g.y_true.values, g.y_pred.values, len(names))
        pooled_rows.append({
            "encoder": encoder,
            "stage": stage,
            "arm": arm,
            "macro_precision": m["macro_precision"],
            "macro_recall": m["macro_recall"],
            "accuracy": m["accuracy"],
            "macro_f1": m["macro_f1"],
            "n_posts": len(g),
            "n_users": g.user.nunique(),
        })
        for c, name in enumerate(names):
            class_rows.append({
                "encoder": encoder,
                "stage": stage,
                "arm": arm,
                "class_id": c,
                "class_name": name,
                "precision": float(m["precision"][c]),
                "recall": float(m["recall"][c]),
                "f1": float(m["f1"][c]),
                "support": int(m["support"][c]),
            })

    for (encoder, fold, arm, stage), g in predictions.groupby(["encoder", "fold", "arm", "stage"]):
        names = stage_class_names(state, stage)
        m = metric_values(g.y_true.values, g.y_pred.values, len(names))
        fold_rows.append({
            "encoder": encoder,
            "fold": int(fold),
            "stage": stage,
            "arm": arm,
            "macro_precision": m["macro_precision"],
            "macro_recall": m["macro_recall"],
            "accuracy": m["accuracy"],
            "macro_f1": m["macro_f1"],
            "n_posts": len(g),
            "n_users": g.user.nunique(),
        })

    return pd.DataFrame(pooled_rows), pd.DataFrame(fold_rows), pd.DataFrame(class_rows)


def aggregate_across_seeds(pooled: pd.DataFrame) -> pd.DataFrame:
    metrics = ["macro_precision", "macro_recall", "accuracy", "macro_f1"]
    groups = ["encoder", "encoder_id", "stage", "arm"]
    mean_df = pooled.groupby(groups, as_index=False)[metrics].mean().rename(
        columns={m: f"{m}_mean" for m in metrics}
    )
    std_df = pooled.groupby(groups, as_index=False)[metrics].std(ddof=1).rename(
        columns={m: f"{m}_std" for m in metrics}
    )
    out = mean_df.merge(std_df, on=groups, how="left")
    nseeds = pooled.groupby(groups, as_index=False)["seed"].nunique().rename(columns={"seed": "n_seeds"})
    out = out.merge(nseeds, on=groups, how="left")
    for m in metrics:
        out[f"{m}_mean_std"] = out.apply(
            lambda r, mm=m: f'{r[f"{mm}_mean"]:.4f} ± {r[f"{mm}_std"]:.4f}', axis=1
        )
    return out.sort_values(groups).reset_index(drop=True)


def best_single_seed(pooled: pd.DataFrame) -> pd.DataFrame:
    idx = pooled.groupby(["encoder", "encoder_id", "stage", "arm"])["macro_f1"].idxmax()
    cols = [
        "encoder", "encoder_id", "stage", "arm", "seed",
        "macro_precision", "macro_recall", "accuracy", "macro_f1",
        "n_posts", "n_users",
    ]
    return pooled.loc[idx, cols].sort_values(["encoder", "stage", "arm"]).reset_index(drop=True)


def print_paper_tables(summary: pd.DataFrame, best: pd.DataFrame) -> None:
    labels = {
        ("final9", "text"): "Text-only",
        ("final9", "direct12"): "Direct12",
        ("final9", "final_hierarchical"): "Final Hierarchical",
        ("stage1", "text"): "Text",
        ("stage1", "direct12"): "Direct12",
        ("stage2", "text"): "Text",
        ("stage2", "scalar_social1"): "Scalar Social1",
    }
    wanted = set(labels)
    x = summary[summary.apply(lambda r: (r.stage, r.arm) in wanted, axis=1)].copy()
    x["System"] = x.apply(lambda r: labels[(r.stage, r.arm)], axis=1)
    x["Precision"] = x["macro_precision_mean_std"]
    x["Recall"] = x["macro_recall_mean_std"]
    x["Accuracy"] = x["accuracy_mean_std"]
    x["Macro-F1"] = x["macro_f1_mean_std"]

    for stage, title in [
        ("final9", "FINAL9 — PRIMARY PAPER RESULTS (mean ± std across seeds)"),
        ("stage1", "STAGE 1 — NORMAL / OFFENSIVE / HATEFUL"),
        ("stage2", "STAGE 2 — 7 HATE SUBTYPES"),
    ]:
        print("\n" + "=" * 120)
        print(title)
        print("=" * 120)
        z = x[x.stage == stage][["encoder", "System", "Precision", "Recall", "Accuracy", "Macro-F1"]]
        print(z.to_string(index=False))

    print("\n" + "=" * 120)
    print("BEST SINGLE-SEED RUNS — DIAGNOSTIC ONLY; DO NOT USE AS THE PRIMARY PAPER ESTIMATE")
    print("=" * 120)
    b = best[(best.stage == "final9") & (best.arm == "final_hierarchical")].copy()
    if len(b):
        b = b[["encoder", "seed", "macro_precision", "macro_recall", "accuracy", "macro_f1"]]
        print(b.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    primary = summary[(summary.stage == "final9") & (summary.arm == "final_hierarchical")].copy()
    if len(primary):
        top = primary.sort_values("macro_f1_mean", ascending=False).iloc[0]
        print("\nPRIMARY BEST MEAN RESULT")
        print(
            f'{top.encoder}: Precision={top.macro_precision_mean:.4f} ± {top.macro_precision_std:.4f} | '
            f'Recall={top.macro_recall_mean:.4f} ± {top.macro_recall_std:.4f} | '
            f'Accuracy={top.accuracy_mean:.4f} ± {top.accuracy_std:.4f} | '
            f'Macro-F1={top.macro_f1_mean:.4f} ± {top.macro_f1_std:.4f}'
        )

    if len(b):
        topb = b.sort_values("macro_f1", ascending=False).iloc[0]
        print("\nBEST SINGLE-SEED DIAGNOSTIC")
        print(
            f'{topb.encoder}, seed={int(topb.seed)}: Precision={topb.macro_precision:.4f} | '
            f'Recall={topb.macro_recall:.4f} | Accuracy={topb.accuracy:.4f} | Macro-F1={topb.macro_f1:.4f}'
        )


def run_seed(
    state: DataState,
    embeddings: np.ndarray,
    encoder_name: str,
    cfg: Config,
    device: torch.device,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    prediction_rows = []
    alpha_rows = []

    for fold in state.folds:
        logits1, logits2 = fold_text_logits(state, embeddings, fold, cfg, device)

        profiles3, source3, pi3 = stage1_user_profiles(state, fold)
        p1 = graph_prior(state.adjacency, profiles3, source3, pi3, cfg.stage1_prior_strength)
        p2 = graph_prior(state.adjacency_exact2, profiles3, source3, pi3, cfg.stage1_prior_strength)

        profiles7, source7, pi7 = stage2_user_profiles(state, fold.model_rows, fold.model_users, cfg)
        p7 = stage2_social_prior(state, profiles7, source7, pi7, cfg)
        alpha = estimate_stage2_alpha(state, embeddings, fold, cfg, device)
        alpha_rows.append({"encoder": encoder_name, "fold": fold.fold, "alpha_social1": alpha})

        test_rows = fold.test_rows
        test_users_idx = state.post_user_index[test_rows]

        y1_text = logits1[test_rows].argmax(axis=1)
        z1_direct12 = logits1[test_rows] + safe_log(p1[test_users_idx]) + safe_log(p2[test_users_idx])
        y1_direct12 = z1_direct12.argmax(axis=1)

        hateful_mask = (state.y_stage1[test_rows] == 2) & pd.notna(state.y_stage2[test_rows])
        hateful_rows = test_rows[hateful_mask]
        hateful_users_idx = state.post_user_index[hateful_rows]
        y2_true = state.y_stage2[hateful_rows].astype(int)
        y2_text = logits2[hateful_rows].argmax(axis=1)
        z2_social = logits2[hateful_rows] + alpha * (
            safe_log(p7[hateful_users_idx]) - safe_log(pi7)[None, :]
        )
        y2_social = z2_social.argmax(axis=1)

        z2_all = logits2[test_rows] + alpha * (
            safe_log(p7[test_users_idx]) - safe_log(pi7)[None, :]
        )
        y2_text_all = logits2[test_rows].argmax(axis=1)
        y2_social_all = z2_all.argmax(axis=1)

        y9_true = state.y_final9[test_rows].astype(int)
        y9_text = final9(y1_text, y2_text_all)
        y9_direct12 = final9(y1_direct12, y2_text_all)
        y9_final = final9(y1_direct12, y2_social_all)

        test_user_values = state.df.iloc[test_rows]["user"].values
        hateful_user_values = state.df.iloc[hateful_rows]["user"].values

        add_predictions(prediction_rows, encoder_name, fold.fold, "text", "stage1", test_rows, test_user_values, state.y_stage1[test_rows], y1_text)
        add_predictions(prediction_rows, encoder_name, fold.fold, "direct12", "stage1", test_rows, test_user_values, state.y_stage1[test_rows], y1_direct12)
        add_predictions(prediction_rows, encoder_name, fold.fold, "text", "stage2", hateful_rows, hateful_user_values, y2_true, y2_text)
        add_predictions(prediction_rows, encoder_name, fold.fold, "scalar_social1", "stage2", hateful_rows, hateful_user_values, y2_true, y2_social)
        add_predictions(prediction_rows, encoder_name, fold.fold, "text", "final9", test_rows, test_user_values, y9_true, y9_text)
        add_predictions(prediction_rows, encoder_name, fold.fold, "direct12", "final9", test_rows, test_user_values, y9_true, y9_direct12)
        add_predictions(prediction_rows, encoder_name, fold.fold, "final_hierarchical", "final9", test_rows, test_user_values, y9_true, y9_final)

    predictions = pd.DataFrame(prediction_rows)
    pooled, folds, per_class = summarize(state, predictions)
    alpha = pd.DataFrame(alpha_rows)

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return pooled, folds, per_class, alpha


def main() -> None:
    args = SimpleNamespace(
        data=DATA_PATH,
        edges=EDGES_PATH,
        output=OUTPUT_DIR,
        user_col=USER_COLUMN,
        text_col=TEXT_COLUMN,
        stage1_col=STAGE1_COLUMN,
        stage2_col=STAGE2_COLUMN,
        edge_source_col=EDGE_SOURCE_COLUMN,
        edge_target_col=EDGE_TARGET_COLUMN,
        normal_label=NORMAL_LABEL,
        offensive_label=OFFENSIVE_LABEL,
        hateful_label=HATEFUL_LABEL,
        time_col=TIME_COLUMN,
        encoders=RUN_ENCODERS,
    )

    invalid_encoders = [name for name in args.encoders if name not in ENCODERS]
    if invalid_encoders:
        raise ValueError(f"Unknown encoders: {invalid_encoders}")

    base_cfg = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    state = load_data(args, base_cfg)

    pooled_all = []
    folds_all = []
    per_class_all = []
    alpha_all = []

    for encoder_name in args.encoders:
        model_id = ENCODERS[encoder_name]
        print(f"\nEncoding texts once: {encoder_name} ({model_id})")
        embeddings = encode_texts(state, model_id, base_cfg, device)

        for seed in MODEL_SEEDS:
            print(f"  seed={seed}")
            cfg = Config(model_seed=int(seed))
            seed_everything(seed)
            pooled, folds, per_class, alpha = run_seed(
                state, embeddings, encoder_name, cfg, device
            )
            for df_ in (pooled, folds, per_class, alpha):
                df_.insert(1, "encoder_id", model_id)
                df_.insert(2, "seed", int(seed))
            pooled_all.append(pooled)
            folds_all.append(folds)
            per_class_all.append(per_class)
            alpha_all.append(alpha)

        del embeddings
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    pooled_df = pd.concat(pooled_all, ignore_index=True)
    folds_df = pd.concat(folds_all, ignore_index=True)
    per_class_df = pd.concat(per_class_all, ignore_index=True)
    alpha_df = pd.concat(alpha_all, ignore_index=True)

    summary_df = aggregate_across_seeds(pooled_df)
    best_df = best_single_seed(pooled_df)

    pooled_df.to_csv(output / "paper_seed_results.csv", index=False)
    folds_df.to_csv(output / "paper_seed_fold_results.csv", index=False)
    per_class_df.to_csv(output / "paper_seed_per_class_results.csv", index=False)
    alpha_df.to_csv(output / "paper_stage2_alpha.csv", index=False)
    summary_df.to_csv(output / "paper_results_mean_std_across_seeds.csv", index=False)
    summary_df[summary_df["stage"] == "final9"].to_csv(
        output / "paper_final9_mean_std_across_seeds.csv", index=False
    )
    best_df.to_csv(output / "paper_best_single_seed_diagnostic.csv", index=False)

    print_paper_tables(summary_df, best_df)


if __name__ == "__main__":
    main()
