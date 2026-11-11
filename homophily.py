#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import hashlib
import os
import re
import sys
import time
import unicodedata

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

try:
    from tqdm import tqdm
except Exception:
    def tqdm(it, **kw):
        return it

CONFIG = {
    "csv": "phond.csv",
    "edges": "edges.csv",
    "out": "homophily_out",
    "user_col": "User_SId",
    "label_col": "classification",
    "text_col": "Text",
    "edge_cols": "",
    "fold_file": "",
    "text_mode": "tfidf",
    "encoder": "HooshvareLab/bert-fa-base-uncased",
    "max_length": 128,
    "k_list": "5,10,20",
    "k_pred": "20,10,5",
    "n_perm": 10000,
    "n_rewire": 1000,
    "n_rewire_pred": 1000,
    "n_randtest": 10000,
    "swaps_per_edge": 10,
    "n_bins": 10,
    "n_bins_joint": 5,
    "smooth": 2.0,
    "n_folds": 5,
    "split_seed": 42,
    "seed": 2026,
    "quick": 0,
}

PAPER = {
    "texts": 35723, "normal": 25273, "offensive": 9026, "hateful": 1424,
    "users": 599, "edges": 7626, "mean_degree": 25.46, "max_degree": 215,
    "density": 0.0426, "components": 1,
    "any_hate_assort": 0.438, "any_harm_assort": 0.473,
    "any_hate_same": 0.731, "any_harm_same": 0.871,
    "any_hate_null": 0.507, "any_harm_null": 0.546,
}

NULLS = ["shuffle", "act_strat", "deg_strat", "actxdeg", "rewire"]


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


def parse_config():
    ap = argparse.ArgumentParser(description="PHOND: homophily under activity, degree and sampling controls")
    for k, v in CONFIG.items():
        ap.add_argument(f"--{k}", type=type(v), default=v)
    args, _unknown = ap.parse_known_args()
    cfg = vars(args)
    if cfg["quick"]:
        cfg["n_perm"] = min(cfg["n_perm"], 300)
        cfg["n_rewire"] = min(cfg["n_rewire"], 50)
        cfg["n_rewire_pred"] = min(cfg["n_rewire_pred"], 20)
        cfg["n_randtest"] = min(cfg["n_randtest"], 1000)
    cfg["k_list"] = [int(x) for x in str(cfg["k_list"]).split(",") if x.strip()]
    cfg["k_pred"] = [int(x) for x in str(cfg["k_pred"]).split(",") if x.strip()]
    cfg["text_mode"] = cfg["text_mode"].strip().lower()
    return cfg


def ensure_inputs(cfg):
    for key in ("csv", "edges"):
        p = cfg[key]
        if not os.path.exists(p):
            sys.exit(f"[error] file not found: {p}\n  Pass the paths with --csv and --edges.")


_FA_FIX = str.maketrans({"ي": "ی", "ى": "ی", "ك": "ک", "‌": "", "‍": "",
                         "‏": "", "‎": "", "ـ": "", " ": ""})


def map_label(v):
    s = unicodedata.normalize("NFKC", str(v)).translate(_FA_FIX).strip().lower()
    if not s or s in ("nan", "none"):
        return -1
    if "تنفر" in s or s.startswith("hate"):
        return 2
    if "توهین" in s or s.startswith("offens"):
        return 1
    if "عادی" in s or s in ("normal", "neutral", "0"):
        return 0
    return -1


def norm_id(x):
    s = str(x).strip()
    if re.fullmatch(r"-?\d+\.0+", s):
        s = s.split(".")[0]
    return s


def read_csv_any(path, **kw):
    base = dict(dtype=str, keep_default_na=False, encoding="utf-8")
    base.update(kw)
    try:
        return pd.read_csv(path, encoding_errors="replace", **base)
    except TypeError:
        return pd.read_csv(path, **base)


def load_dataset(cfg):
    df = read_csv_any(cfg["csv"])
    df.columns = [str(c).strip() for c in df.columns]
    for key in ("user_col", "label_col", "text_col"):
        col = cfg[key]
        if col in df.columns:
            continue
        alt = [c for c in df.columns if c.lower() == col.lower()]
        if alt:
            cfg[key] = alt[0]
        elif key == "text_col" and cfg["text_mode"] == "none":
            continue
        else:
            sys.exit(f"[error] column '{col}' not found in {cfg['csv']}.\n  Columns: {list(df.columns)}")
    users = df[cfg["user_col"]].map(norm_id).to_numpy()
    y3 = df[cfg["label_col"]].map(map_label).to_numpy()
    bad = (y3 < 0) | np.isin(users, ["", "nan", "None"])
    if bad.any():
        vals = df.loc[bad, cfg["label_col"]].value_counts().head(10).to_dict()
        log(f"[warn] {int(bad.sum())} rows have an unknown label or an empty user id and are dropped. Values: {vals}")
    keep = ~bad
    texts = None
    if cfg["text_col"] in df.columns:
        texts = df[cfg["text_col"]].astype(str).to_numpy()[keep]
    return users[keep], y3[keep].astype(int), texts, len(df)


def _pick_edge_cols(df, user_set, forced):
    if forced:
        cols = [c.strip() for c in forced.split(",")]
        if len(cols) != 2 or any(c not in df.columns for c in cols):
            sys.exit(f"[error] --edge_cols {forced} not found. Columns: {list(df.columns)}")
        return cols
    scores = []
    for c in df.columns:
        vals = df[c].map(norm_id)
        scores.append((float(vals.isin(user_set).mean()) if len(vals) else 0.0, c))
    good = [c for frac, c in sorted(scores, key=lambda t: -t[0]) if frac >= 0.5]
    return good[:2] if len(good) >= 2 else None


def load_edges(cfg, user_index):
    user_set = set(user_index)
    df = read_csv_any(cfg["edges"])
    cols = _pick_edge_cols(df, user_set, cfg["edge_cols"])
    header_is_data = cols is not None and all(norm_id(c) in user_set for c in cols)
    if (cols is None or header_is_data) and not cfg["edge_cols"]:
        df = read_csv_any(cfg["edges"], header=None)
        cols = _pick_edge_cols(df, user_set, "")
    if cols is None:
        sys.exit("[error] could not find two user-id columns in the edge file. "
                 f"Columns: {list(df.columns)}. Pass them with --edge_cols colA,colB")
    a = df[cols[0]].map(norm_id).to_numpy()
    b = df[cols[1]].map(norm_id).to_numpy()
    self_loop = a == b
    known = np.array([x in user_set for x in a]) & np.array([x in user_set for x in b])
    ok = known & ~self_loop
    ia = np.array([user_index[x] for x in a[ok]], dtype=np.int64)
    ib = np.array([user_index[x] for x in b[ok]], dtype=np.int64)
    pairs = np.unique(np.stack([np.minimum(ia, ib), np.maximum(ia, ib)], 1), axis=0)
    report = {
        "columns": [str(c) for c in cols],
        "raw_rows": int(len(a)),
        "self_loops": int(self_loop.sum()),
        "rows_with_unknown_user": int((~known & ~self_loop).sum()),
        "duplicate_rows": int(ok.sum() - len(pairs)),
        "unique_edges": int(len(pairs)),
    }
    return np.ascontiguousarray(pairs.astype(np.int32)), report


def get_folds(cfg, users, counts):
    n = len(users)
    if cfg["fold_file"]:
        if not os.path.exists(cfg["fold_file"]):
            sys.exit(f"[error] fold file not found: {cfg['fold_file']}")
        ff = read_csv_any(cfg["fold_file"])
        uset = set(users)
        ucol = max(ff.columns, key=lambda c: ff[c].map(norm_id).isin(uset).mean())
        fcols = [c for c in ff.columns if c != ucol and pd.to_numeric(ff[c], errors="coerce").notna().all()]
        if not fcols:
            sys.exit("[error] the fold file needs a user column and an integer fold column")
        m = dict(zip(ff[ucol].map(norm_id), pd.to_numeric(ff[fcols[0]]).astype(int)))
        missing = [u for u in users if u not in m]
        if missing:
            sys.exit(f"[error] {len(missing)} users are missing from the fold file, e.g. {missing[:3]}")
        raw = np.array([m[u] for u in users])
        _, folds = np.unique(raw, return_inverse=True)
        source = f"fold file {cfg['fold_file']} (user column '{ucol}', fold column '{fcols[0]}')"
    else:
        dom = 2 - np.argmax(counts[:, ::-1], axis=1)
        skf = StratifiedKFold(n_splits=cfg["n_folds"], shuffle=True, random_state=cfg["split_seed"])
        folds = np.empty(n, dtype=int)
        for f, (_, te) in enumerate(skf.split(np.zeros(n), dom)):
            folds[te] = f
        source = (f"rebuilt: StratifiedKFold({cfg['n_folds']}, shuffle, seed {cfg['split_seed']}) over the "
                  "dominant label of each user, users sorted by id. Pass --fold_file to reuse the Table 5 folds exactly.")
    pd.DataFrame({"user": users, "fold": folds}).to_csv(os.path.join(cfg["out"], "user_folds_used.csv"), index=False)
    return folds.astype(int), source


def same_share(xb, e):
    return float(np.mean(xb[e[:, 0]] == xb[e[:, 1]]))


def assort(x, e):
    a = x[e[:, 0]].astype(float)
    b = x[e[:, 1]].astype(float)
    u = np.concatenate([a, b])
    v = np.concatenate([b, a])
    mu = u.mean()
    u = u - mu
    v = v - mu
    den = float((u * u).sum())
    return float((u * v).sum() / den) if den > 0 else float("nan")


def _assort_cols(X0, X1):
    mu = (X0.sum(0) + X1.sum(0)) / (2.0 * len(X0))
    D0 = X0 - mu
    D1 = X1 - mu
    den = (D0 * D0).sum(0) + (D1 * D1).sum(0)
    num = 2.0 * (D0 * D1).sum(0)
    with np.errstate(invalid="ignore", divide="ignore"):
        r = num / den
    r[den <= 0] = np.nan
    return r


def stat_vector(Xb, Xn, e):
    a0 = e[:, 0]
    a1 = e[:, 1]
    out = []
    if Xb.shape[1]:
        B0 = Xb[a0].astype(float)
        B1 = Xb[a1].astype(float)
        same = (B0 == B1).mean(0)
        ab = _assort_cols(B0, B1)
        out.append(np.column_stack([same, ab]).ravel())
    if Xn.shape[1]:
        out.append(_assort_cols(Xn[a0], Xn[a1]))
    return np.concatenate(out) if out else np.zeros(0)


def stat_names(names_b, names_n):
    out = []
    for a in names_b:
        out += [(a, "same-attribute edge share"), (a, "assortativity")]
    for a in names_n:
        out += [(a, "assortativity (numeric)")]
    return out


def rank_bins(values, n_bins):
    r = pd.Series(values).rank(method="first").to_numpy() - 1.0
    return np.minimum((r * n_bins / len(values)).astype(int), n_bins - 1)


def groups_from(bins):
    return [np.where(bins == b)[0] for b in np.unique(bins)]


def strata_perm(groups, n, rng):
    perm = np.arange(n)
    for g in groups:
        if len(g) > 1:
            perm[g] = g[rng.permutation(len(g))]
    return perm


def p_greater(null, obs):
    if not np.isfinite(obs):
        return float("nan")
    null = null[np.isfinite(null)]
    return float((1 + np.sum(null >= obs - 1e-12)) / (len(null) + 1))


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


def rewire_many(e0, n, n_rep, swaps_per_edge, rng, desc):
    E = len(e0)
    out = np.empty((n_rep, E, 2), dtype=np.int32)
    if E < 2:
        out[:] = e0
        return out, 0.0
    n_try = int(swaps_per_edge * E)
    acc = 0.0
    for r in tqdm(range(n_rep), desc=desc, mininterval=2):
        e = np.ascontiguousarray(e0.copy())
        adj = np.zeros((n, n), dtype=np.uint8)
        adj[e[:, 0], e[:, 1]] = 1
        adj[e[:, 1], e[:, 0]] = 1
        ia = rng.integers(0, E, n_try)
        ib = rng.integers(0, E, n_try)
        fl = rng.integers(0, 2, n_try).astype(np.uint8)
        acc += _SWAP(e, adj, ia, ib, fl) / n_try
        out[r] = e
    d0 = np.bincount(e0.ravel(), minlength=n)
    d1 = np.bincount(out[0].ravel(), minlength=n)
    lo = np.minimum(out[0][:, 0], out[0][:, 1])
    hi = np.maximum(out[0][:, 0], out[0][:, 1])
    assert np.array_equal(d0, d1), "rewiring changed the degree sequence"
    assert len(np.unique(lo.astype(np.int64) * n + hi)) == E and (lo != hi).all(), "rewiring broke the simple graph"
    return out, acc / n_rep


def homophily_block(tag, nodes, e_full, n_full, bin_attrs, num_attrs, act_full, cfg, rng):
    pos = -np.ones(n_full, dtype=np.int64)
    pos[nodes] = np.arange(len(nodes))
    keep = (pos[e_full[:, 0]] >= 0) & (pos[e_full[:, 1]] >= 0)
    e = np.ascontiguousarray(np.stack([pos[e_full[keep, 0]], pos[e_full[keep, 1]]], 1).astype(np.int32))
    n = len(nodes)
    deg = np.bincount(e.ravel(), minlength=n)
    act = act_full[nodes]
    names_b = list(bin_attrs)
    names_n = list(num_attrs)
    Xb = np.column_stack([bin_attrs[a][nodes] for a in names_b]).astype(np.int8)
    Xn = (np.column_stack([num_attrs[a][nodes] for a in names_n]).astype(float)
          if names_n else np.zeros((n, 0)))
    obs = stat_vector(Xb, Xn, e)

    groups = {
        "shuffle": [np.arange(n)],
        "act_strat": groups_from(rank_bins(act, cfg["n_bins"])),
        "deg_strat": groups_from(rank_bins(deg, cfg["n_bins"])),
        "actxdeg": groups_from(rank_bins(act, cfg["n_bins_joint"]) * cfg["n_bins_joint"]
                               + rank_bins(deg, cfg["n_bins_joint"])),
    }
    nulls = {}
    for name, g in groups.items():
        m = np.empty((cfg["n_perm"], len(obs)))
        for r in tqdm(range(cfg["n_perm"]), desc=f"{tag} {name}", mininterval=2):
            p = strata_perm(g, n, rng)
            m[r] = stat_vector(Xb[p], Xn[p], e)
        nulls[name] = m
    rew, acc = rewire_many(e, n, cfg["n_rewire"], cfg["swaps_per_edge"], rng, f"{tag} rewire")
    m = np.empty((len(rew), len(obs)))
    for r in range(len(rew)):
        m[r] = stat_vector(Xb, Xn, rew[r])
    nulls["rewire"] = m

    rows = []
    for i, (attr, stat) in enumerate(stat_names(names_b, names_n)):
        row = {"subset": tag, "n_users": n, "n_edges": len(e), "attribute": attr, "statistic": stat,
               "observed": obs[i]}
        for name in NULLS:
            col = nulls[name][:, i]
            row[f"{name}_mean"] = float(np.nanmean(col))
            row[f"{name}_sd"] = float(np.nanstd(col))
            row[f"{name}_p"] = p_greater(col, obs[i])
        rows.append(row)
    info = {"n": n, "E": len(e), "rewire_acceptance": acc, "e": e, "rewired": rew}
    return rows, info


def print_block(rows):
    hdr = f"  {'attribute':<22}{'statistic':<27}{'obs':>7}" + "".join(f"{n:>17}" for n in NULLS)
    log(hdr)
    log("  " + " " * 56 + "".join(f"{'(null mean, p)':>17}" for _ in NULLS))
    for r in rows:
        line = f"  {r['attribute']:<22}{r['statistic']:<27}{r['observed']:>7.3f}"
        for n in NULLS:
            p = r[f"{n}_p"]
            ps = "  nan" if not np.isfinite(p) else ("<.001" if p < 0.001 else f"{p:.3f}")
            line += f"{r[f'{n}_mean']:>10.3f} {ps:>6}"
        log(line)


def encode_first_token(texts, model_id, max_length, cache_dir):
    h = hashlib.md5(("\n".join(texts) + f"|{model_id}|{max_length}").encode("utf-8", "replace")).hexdigest()[:10]
    slug = re.sub(r"[^A-Za-z0-9]+", "_", model_id).strip("_")
    path = os.path.join(cache_dir, f"emb_{slug}_len{max_length}_{h}.npy")
    if os.path.exists(path):
        arr = np.load(path)
        if arr.shape[0] == len(texts):
            log(f"  loaded cached embeddings: {path}")
            return arr
    import torch
    from transformers import AutoModel, AutoTokenizer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"  encoding {len(texts):,} texts with {model_id} on {device} (first token, max_length {max_length})")
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id).to(device).eval()
    bs = 256 if device == "cuda" else 64
    order = np.argsort([len(t) for t in texts], kind="stable")
    out = np.zeros((len(texts), model.config.hidden_size), dtype=np.float32)
    with torch.no_grad():
        for s in tqdm(range(0, len(texts), bs), desc="encode", mininterval=5):
            idx = order[s:s + bs]
            enc = tok([texts[i] for i in idx], padding=True, truncation=True,
                      max_length=max_length, return_tensors="pt").to(device)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(device == "cuda")):
                hid = model(**enc).last_hidden_state[:, 0, :]
            out[idx] = hid.float().cpu().numpy()
    np.save(path, out)
    log(f"  saved embeddings: {path}")
    return out


def user_mean_embeddings(emb, uidx, n_users):
    M = sparse.csr_matrix((np.ones(len(uidx), dtype=np.float32), (uidx, np.arange(len(uidx)))),
                          shape=(n_users, len(uidx)))
    cnt = np.asarray(M.sum(1)).ravel()
    return np.asarray(M @ emb) / np.maximum(cnt, 1)[:, None]


def graph_mats(e, n):
    A = np.zeros((n, n), dtype=np.float32)
    A[e[:, 0], e[:, 1]] = 1.0
    A[e[:, 1], e[:, 0]] = 1.0
    A2 = (A @ A) > 0
    A2 &= (A == 0)
    np.fill_diagonal(A2, False)
    return A, A2.astype(np.float32)


def fold_targets(rate, kmask, train):
    med = float(np.median(rate[train & kmask]))
    return med, (rate > med).astype(int)


def network_arms(A, A2, kmask, folds, rate, any_hate, off_rate, smooth):
    n = len(rate)
    p_vote = np.full(n, -1)
    p_net = np.full(n, -1)
    for f in np.unique(folds):
        train = folds != f
        test = folds == f
        tr = train & kmask
        te = test & kmask
        med, y = fold_targets(rate, kmask, train)
        T = train.astype(np.float32)
        lam = smooth
        rbar = float(rate[train].mean())
        abar = float(any_hate[train].mean())
        obar = float(off_rate[train].mean())
        c1 = A @ T
        c2 = A2 @ T
        X = np.column_stack([
            (A @ (T * rate) + lam * rbar) / (c1 + lam),
            (A @ (T * any_hate) + lam * abar) / (c1 + lam),
            (A @ (T * off_rate) + lam * obar) / (c1 + lam),
            (A2 @ (T * rate) + lam * rbar) / (c2 + lam),
        ])
        if len(np.unique(y[tr])) == 2:
            clf = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000))
            clf.fit(X[tr], y[tr])
            p_net[te] = clf.predict(X[te])
        else:
            p_net[te] = y[tr][0] if tr.any() else 0
        voters = (train & kmask).astype(np.float32)
        above = (rate > med).astype(np.float32)
        v1 = A @ (voters * above)
        v0 = A @ (voters * (1.0 - above))
        maj = int(y[tr].mean() >= 0.5) if tr.any() else 0
        pv = np.where(v1 > v0, 1, np.where(v0 > v1, 0, maj))
        p_vote[te] = pv[te]
    return p_vote, p_net


def plain_arm(X, kmask, folds, rate, cv_text=False, tfidf_texts=None):
    n = len(rate)
    pred = np.full(n, -1)
    for f in np.unique(folds):
        train = folds != f
        test = folds == f
        tr = train & kmask
        te = test & kmask
        _, y = fold_targets(rate, kmask, train)
        if len(np.unique(y[tr])) < 2:
            pred[te] = y[tr][0] if tr.any() else 0
            continue
        inner = StratifiedKFold(5, shuffle=True, random_state=0)
        if tfidf_texts is not None:
            from sklearn.feature_extraction.text import TfidfVectorizer
            vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True,
                                  max_features=100000)
            Xtr = vec.fit_transform(tfidf_texts[tr])
            Xte = vec.transform(tfidf_texts[te])
            clf = GridSearchCV(LogisticRegression(class_weight="balanced", solver="liblinear", max_iter=5000,
                                                  random_state=0),
                               {"C": np.logspace(-2, 2, 7)}, scoring="f1_macro", cv=inner)
            clf.fit(Xtr, y[tr])
            pred[te] = clf.predict(Xte)
            continue
        if cv_text:
            pipe = make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=5000))
            clf = GridSearchCV(pipe, {"logisticregression__C": np.logspace(-4, 0, 9)}, scoring="f1_macro", cv=inner)
        else:
            clf = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000))
        clf.fit(X[tr], y[tr])
        pred[te] = clf.predict(X[te])
    return pred


def pooled_truth(kmask, folds, rate):
    y = np.full(len(rate), -1)
    for f in np.unique(folds):
        _, yf = fold_targets(rate, kmask, folds != f)
        te = (folds == f) & kmask
        y[te] = yf[te]
    return y


def macro_f1_batch(y, P):
    y1 = (y == 1)[None, :]
    y0 = ~y1
    p1 = P == 1
    p0 = ~p1
    tp1 = (p1 & y1).sum(1)
    fp1 = (p1 & y0).sum(1)
    fn1 = (p0 & y1).sum(1)
    tp0 = (p0 & y0).sum(1)
    fp0 = (p0 & y1).sum(1)
    fn0 = (p1 & y0).sum(1)
    d1 = 2 * tp1 + fp1 + fn1
    d0 = 2 * tp0 + fp0 + fn0
    f1 = np.where(d1 > 0, 2 * tp1 / np.maximum(d1, 1), 0.0)
    f0 = np.where(d0 > 0, 2 * tp0 / np.maximum(d0, 1), 0.0)
    return (f1 + f0) / 2.0


def paired_randomisation(y, pa, pb, n_iter, rng):
    obs = float(macro_f1_batch(y, pa[None, :])[0] - macro_f1_batch(y, pb[None, :])[0])
    out = np.empty(n_iter)
    chunk = 2000
    for s in range(0, n_iter, chunk):
        m = min(chunk, n_iter - s)
        S = rng.random((m, len(y))) < 0.5
        PA = np.where(S, pb[None, :], pa[None, :])
        PB = np.where(S, pa[None, :], pb[None, :])
        out[s:s + m] = macro_f1_batch(y, PA) - macro_f1_batch(y, PB)
    p = float((1 + np.sum(np.abs(out) >= abs(obs) - 1e-12)) / (n_iter + 1))
    return obs, p


def per_fold_f1(y, p, folds, kmask):
    vals = []
    for f in np.unique(folds):
        m = (folds == f) & kmask
        vals.append(f1_score(y[m], p[m], average="macro", zero_division=0) if m.sum() else float("nan"))
    return vals


def run_prediction(k, n, e, rew_full, folds, rate, any_hate, off_rate, n_texts, deg, user_emb, user_docs, cfg, rng):
    kmask = n_texts >= k
    S = np.where(kmask)[0]
    y = pooled_truth(kmask, folds, rate)
    meds = [fold_targets(rate, kmask, folds != f)[0] for f in np.unique(folds)]
    log(f"\n  k = {k}: {len(S)} users with at least {k} texts. Fold medians of the hate rate: "
        + ", ".join(f"{m:.4f}" for m in meds) + f". Share above the median: {y[S].mean():.3f}")
    if any(m == 0 for m in meds):
        log("  [note] a fold median is 0, so in that fold 'above the median' means 'has any hateful text'.")

    A, A2 = graph_mats(e, n)
    preds = {}
    preds["vote"], preds["network"] = network_arms(A, A2, kmask, folds, rate, any_hate, off_rate, cfg["smooth"])
    preds["activity"] = plain_arm(np.log1p(n_texts)[:, None], kmask, folds, rate)
    preds["degree"] = plain_arm(np.log1p(deg)[:, None], kmask, folds, rate)
    preds["activity+degree"] = plain_arm(np.column_stack([np.log1p(n_texts), np.log1p(deg)]), kmask, folds, rate)
    if user_emb is not None:
        preds["text"] = plain_arm(user_emb, kmask, folds, rate, cv_text=True)
    elif user_docs is not None:
        preds["text"] = plain_arm(None, kmask, folds, rate, tfidf_texts=user_docs)

    rew_vote, rew_net = [], []
    for r in tqdm(range(min(cfg["n_rewire_pred"], len(rew_full))), desc=f"k={k} rewired controls", mininterval=2):
        Ar, A2r = graph_mats(rew_full[r], n)
        pv, pn = network_arms(Ar, A2r, kmask, folds, rate, any_hate, off_rate, cfg["smooth"])
        rew_vote.append(f1_score(y[S], pv[S], average="macro", zero_division=0))
        rew_net.append(f1_score(y[S], pn[S], average="macro", zero_division=0))
    rew_vote = np.array(rew_vote)
    rew_net = np.array(rew_net)

    rows = []
    for arm, p in preds.items():
        rows.append({
            "k": k, "n_users": len(S), "arm": arm,
            "macro_f1": f1_score(y[S], p[S], average="macro", zero_division=0),
            "accuracy": accuracy_score(y[S], p[S]),
            "fold_f1": ";".join(f"{v:.3f}" for v in per_fold_f1(y, p, folds, kmask)),
        })
    rows.append({"k": k, "n_users": len(S), "arm": "network on rewired graphs",
                 "macro_f1": float(rew_net.mean()), "accuracy": float("nan"),
                 "fold_f1": f"sd {rew_net.std():.3f} over {len(rew_net)} graphs"})
    rows.append({"k": k, "n_users": len(S), "arm": "vote on rewired graphs",
                 "macro_f1": float(rew_vote.mean()), "accuracy": float("nan"),
                 "fold_f1": f"sd {rew_vote.std():.3f} over {len(rew_vote)} graphs"})

    f1_of = {r["arm"]: r["macro_f1"] for r in rows}
    tests = []
    pairs = [("network", "activity"), ("network", "degree"), ("network", "activity+degree"),
             ("vote", "activity"), ("vote", "activity+degree")]
    if "text" in preds:
        pairs += [("network", "text"), ("vote", "text"), ("text", "activity")]
    for a, b in pairs:
        d, p = paired_randomisation(y[S], preds[a][S], preds[b][S], cfg["n_randtest"], rng)
        fa = per_fold_f1(y, preds[a], folds, kmask)
        fb = per_fold_f1(y, preds[b], folds, kmask)
        wins = sum(x > z for x, z in zip(fa, fb))
        tests.append({"k": k, "comparison": f"{a} vs {b}", "diff_macro_f1": d, "p_paired_randomisation": p,
                      "folds_won": f"{wins}/{len(fa)}"})
    for arm, rew in (("network", rew_net), ("vote", rew_vote)):
        tests.append({"k": k, "comparison": f"{arm} vs its rewired graphs",
                      "diff_macro_f1": f1_of[arm] - float(rew.mean()),
                      "p_paired_randomisation": p_greater(rew, f1_of[arm]),
                      "folds_won": f"rank test over {len(rew)} rewired graphs"})
    return rows, tests


def main():
    cfg = parse_config()
    os.makedirs(cfg["out"], exist_ok=True)
    tee = Tee(os.path.join(cfg["out"], "report.txt"))
    sys.stdout = tee
    t0 = time.time()
    try:
        _main(cfg)
    finally:
        log(f"\nTotal time: {(time.time() - t0) / 60:.1f} min. Outputs in {cfg['out']}")
        sys.stdout = tee.out
        tee.close()


def _main(cfg):
    ensure_inputs(cfg)
    rng = np.random.default_rng(cfg["seed"])
    log("PHOND: homophily under activity, degree and sampling controls")
    log(f"csv   : {cfg['csv']}\nedges : {cfg['edges']}\nout   : {cfg['out']}")
    log(f"n_perm {cfg['n_perm']}, n_rewire {cfg['n_rewire']}, n_rewire_pred {cfg['n_rewire_pred']}, "
        f"numba {'yes' if HAVE_NUMBA else 'no'}, text_mode {cfg['text_mode']}")
    if not HAVE_NUMBA and cfg["n_rewire"] > 200:
        cfg["n_rewire"] = 200
        cfg["n_rewire_pred"] = min(cfg["n_rewire_pred"], 100)
        log("[note] numba not found, so rewiring is slow; n_rewire set to 200 and n_rewire_pred to 100.")

    user_of_text, y3, texts, n_rows_file = load_dataset(cfg)
    users = np.unique(user_of_text)
    user_index = {u: i for i, u in enumerate(users)}
    uidx = np.array([user_index[u] for u in user_of_text], dtype=np.int64)
    n = len(users)
    counts = np.zeros((n, 3), dtype=np.int64)
    np.add.at(counts, (uidx, y3), 1)
    n_texts = counts.sum(1)
    n_off = counts[:, 1]
    n_hate = counts[:, 2]
    e, erep = load_edges(cfg, user_index)
    deg = np.bincount(e.ravel(), minlength=n)

    log("\n=== A. Sanity check against the paper ===")
    A_sp = sparse.coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(n, n))
    ncomp, _ = connected_components(A_sp + A_sp.T, directed=False)
    any_hate = (n_hate > 0).astype(int)
    any_harm = ((n_hate + n_off) > 0).astype(int)
    h1, h0 = any_hate.sum(), n - any_hate.sum()
    m1, m0 = any_harm.sum(), n - any_harm.sum()
    exp_null_hate = (h1 * (h1 - 1) + h0 * (h0 - 1)) / (n * (n - 1))
    exp_null_harm = (m1 * (m1 - 1) + m0 * (m0 - 1)) / (n * (n - 1))
    checks = [
        ("rows in file", n_rows_file, PAPER["texts"]),
        ("texts used", len(y3), PAPER["texts"]),
        ("normal", int((y3 == 0).sum()), PAPER["normal"]),
        ("offensive", int((y3 == 1).sum()), PAPER["offensive"]),
        ("hateful", int((y3 == 2).sum()), PAPER["hateful"]),
        ("users", n, PAPER["users"]),
        ("edges (unique, undirected)", len(e), PAPER["edges"]),
        ("mean degree", round(2 * len(e) / n, 2), PAPER["mean_degree"]),
        ("max degree", int(deg.max()), PAPER["max_degree"]),
        ("density", round(2 * len(e) / (n * (n - 1)), 4), PAPER["density"]),
        ("connected components", int(ncomp), PAPER["components"]),
        ("assortativity, any hateful", round(assort(any_hate, e), 3), PAPER["any_hate_assort"]),
        ("assortativity, any offensive or hateful", round(assort(any_harm, e), 3), PAPER["any_harm_assort"]),
        ("same-attribute edges, any hateful", round(same_share(any_hate, e), 3), PAPER["any_hate_same"]),
        ("same-attribute edges, any offensive or hateful", round(same_share(any_harm, e), 3), PAPER["any_harm_same"]),
        ("shuffle null (exact), any hateful", round(exp_null_hate, 3), PAPER["any_hate_null"]),
        ("shuffle null (exact), any offensive or hateful", round(exp_null_harm, 3), PAPER["any_harm_null"]),
    ]
    bad = 0
    for name, got, want in checks:
        ok = abs(float(got) - float(want)) <= (0.0015 if isinstance(want, float) else 0)
        bad += not ok
        log(f"  {name:<48}{got!s:>10}   paper {want!s:>8}   {'ok' if ok else '<-- differs'}")
    log(f"  edge file: columns {erep['columns']}, {erep['raw_rows']} rows, {erep['duplicate_rows']} duplicate rows, "
        f"{erep['self_loops']} self loops, {erep['rows_with_unknown_user']} rows with a user not in the dataset")
    if bad:
        log(f"  [warn] {bad} values differ from the paper. Check that this is the paper's file (35,723 rows).")

    log("\n=== B. Activity and degree ===")
    rho, prho = spearmanr(n_texts, deg)
    log(f"  Spearman(number of texts, degree) = {rho:.3f} (p = {prho:.2g})")
    log(f"  degree assortativity              = {assort(deg.astype(float), e):.3f}")
    log(f"  activity assortativity (log #texts) = {assort(np.log1p(n_texts), e):.3f}")
    for name, a in (("any hateful", any_hate), ("any offensive or hateful", any_harm)):
        log(f"  {name:<26} yes: n={int(a.sum()):>3}, median texts {np.median(n_texts[a == 1]):>6.1f}, "
            f"median degree {np.median(deg[a == 1]):>6.1f}   |   no: n={int((a == 0).sum()):>3}, "
            f"median texts {np.median(n_texts[a == 0]):>6.1f}, median degree {np.median(deg[a == 0]):>6.1f}")

    log("\n=== C. Homophily under five null models ===")
    log("  p = one sided share of null draws at or above the observed value; '<.001' means none in the draws.")
    log("  shuffle = plain label shuffle (as in the paper); act_strat / deg_strat = shuffle within deciles of")
    log("  activity / degree; actxdeg = within 5 x 5 activity-by-degree cells; rewire = degree preserving rewiring.")
    hate_rate = n_hate / np.maximum(n_texts, 1)
    harm_rate = (n_hate + n_off) / np.maximum(n_texts, 1)
    off_rate = n_off / np.maximum(n_texts, 1)
    all_rows = []
    rewired_full = None
    blocks = [("all users", np.arange(n), 1)] + [(f"users with >= {k} texts", np.where(n_texts >= k)[0], k)
                                                  for k in cfg["k_list"]]
    for tag, nodes, k in blocks:
        bin_attrs = {"any hateful": any_hate, "any off. or hateful": any_harm}
        num_attrs = {}
        if k > 1:
            mh = float(np.median(hate_rate[nodes]))
            mm = float(np.median(harm_rate[nodes]))
            bin_attrs["hate rate > median"] = (hate_rate > mh).astype(int)
            bin_attrs["harm rate > median"] = (harm_rate > mm).astype(int)
            num_attrs = {"hate rate": hate_rate, "harm rate": harm_rate}
        log(f"\n  [{tag}]")
        if k > 1:
            log(f"  median hate rate {mh:.4f} (share above {np.mean(hate_rate[nodes] > mh):.3f}), "
                f"median offensive-or-hateful rate {mm:.4f} (share above {np.mean(harm_rate[nodes] > mm):.3f})")
        rows, info = homophily_block(tag, nodes, e, n, bin_attrs, num_attrs, n_texts, cfg, rng)
        log(f"  {info['n']} users, {info['E']} edges, rewiring acceptance {info['rewire_acceptance']:.2f}")
        print_block(rows)
        all_rows += rows
        if k == 1:
            rewired_full = info["rewired"]
    pd.DataFrame(all_rows).to_csv(os.path.join(cfg["out"], "homophily_tests.csv"), index=False)

    log("\n=== D. User level prediction: is the user's hate rate above the training median? ===")
    folds, source = get_folds(cfg, users, counts)
    log(f"  folds: {source}")
    user_emb, user_docs = None, None
    if cfg["text_mode"] == "parsbert":
        if texts is None:
            log("  [warn] no text column, text arm skipped")
        else:
            try:
                emb = encode_first_token(list(texts), cfg["encoder"], cfg["max_length"], cfg["out"])
                user_emb = user_mean_embeddings(emb, uidx, n)
            except Exception as ex:
                log(f"  [warn] ParsBERT encoding failed ({type(ex).__name__}: {ex}). Using TF-IDF for the text arm.")
                cfg["text_mode"] = "tfidf"
    if cfg["text_mode"] == "tfidf" and texts is not None:
        docs = pd.Series(texts).groupby(uidx).agg(lambda s: " \n ".join(s))
        user_docs = docs.reindex(range(n)).fillna("").to_numpy()
    if len(rewired_full) < cfg["n_rewire_pred"]:
        extra, _ = rewire_many(e, n, cfg["n_rewire_pred"] - len(rewired_full), cfg["swaps_per_edge"], rng, "extra rewires")
        rewired_full = np.concatenate([rewired_full, extra], 0)
    pred_rows, test_rows = [], []
    for k in cfg["k_pred"]:
        rows, tests = run_prediction(k, n, e, rewired_full, folds, hate_rate, any_hate, off_rate,
                                     n_texts, deg, user_emb, user_docs, cfg, rng)
        pred_rows += rows
        test_rows += tests
        log(f"  {'arm':<28}{'macro F1':>9}{'acc':>7}   per fold / spread")
        for r in rows:
            acc = "" if not np.isfinite(r["accuracy"]) else f"{r['accuracy']:.3f}"
            log(f"  {r['arm']:<28}{r['macro_f1']:>9.3f}{acc:>7}   {r['fold_f1']}")
        log(f"  {'comparison':<36}{'diff':>8}{'p':>9}   folds")
        for t in tests:
            p = t["p_paired_randomisation"]
            log(f"  {t['comparison']:<36}{t['diff_macro_f1']:>+8.3f}{p:>9.4f}   {t['folds_won']}")
    pd.DataFrame(pred_rows).to_csv(os.path.join(cfg["out"], "prediction_results.csv"), index=False)
    pd.DataFrame(test_rows).to_csv(os.path.join(cfg["out"], "prediction_tests.csv"), index=False)

    pd.DataFrame({"user": users, "n_texts": n_texts, "n_offensive": n_off, "n_hateful": n_hate,
                  "hate_rate": hate_rate, "degree": deg, "fold": folds}).to_csv(
        os.path.join(cfg["out"], "user_table.csv"), index=False)

    log("\n=== Summary ===")
    df = pd.DataFrame(all_rows)

    def pick(subset, attr, stat):
        r = df[(df.subset == subset) & (df.attribute == attr) & (df.statistic == stat)]
        return r.iloc[0] if len(r) else None

    def fmt_p(p):
        return "<0.001" if p < 0.001 else f"{p:.3f}"

    for attr in ("any hateful", "any off. or hateful"):
        r = pick("all users", attr, "same-attribute edge share")
        log(f"  all users, {attr}: same-attribute edges {r['observed']:.3f}; null means "
            + ", ".join(f"{nm} {r[f'{nm}_mean']:.3f} (p {fmt_p(r[f'{nm}_p'])})" for nm in NULLS))
    for k in cfg["k_list"]:
        tag = f"users with >= {k} texts"
        r = pick(tag, "hate rate", "assortativity (numeric)")
        if r is not None:
            log(f"  >= {k} texts, hate rate assortativity {r['observed']:.3f}; null means "
                + ", ".join(f"{nm} {r[f'{nm}_mean']:.3f} (p {fmt_p(r[f'{nm}_p'])})" for nm in NULLS))
        r = pick(tag, "hate rate > median", "same-attribute edge share")
        if r is not None:
            log(f"  >= {k} texts, hate rate above median, same-attribute edges {r['observed']:.3f}; null means "
                + ", ".join(f"{nm} {r[f'{nm}_mean']:.3f} (p {fmt_p(r[f'{nm}_p'])})" for nm in NULLS))
    k0 = cfg["k_pred"][0] if cfg["k_pred"] else None
    if k0 is not None:
        pr = pd.DataFrame(pred_rows)
        pr = pr[pr.k == k0]
        log(f"  prediction, users with >= {k0} texts (n = {int(pr.n_users.iloc[0])}): "
            + ", ".join(f"{a} {f:.3f}" for a, f in zip(pr.arm, pr.macro_f1)))
        tr = pd.DataFrame(test_rows)
        tr = tr[tr.k == k0]
        log("  tests: " + "; ".join(f"{c} {d:+.3f} (p {fmt_p(p)})" for c, d, p in
                                    zip(tr.comparison, tr.diff_macro_f1, tr.p_paired_randomisation)))


if __name__ == "__main__":
    main()
