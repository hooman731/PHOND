#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import argparse
import os
import re
import subprocess
import sys
import time
import unicodedata

import numpy as np


def _import_or_install(module, package):
    import importlib
    try:
        return importlib.import_module(module)
    except ImportError:
        print(f"installing {package} ...", flush=True)
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", package], check=False)
        return importlib.import_module(module)


pd = _import_or_install("pandas", "pandas")

try:
    from tqdm import tqdm
except Exception:
    def tqdm(it, **kw):
        return it

CONFIG = {
    "csv": "",
    "edges": "",
    "pps": "",
    "out": "homophily_pps_out",
    "user_col": "User_SId",
    "label_col": "classification",
    "id_col": "Id",
    "text_col": "Text",
    "edge_cols": "",
    "k_list": "5,10,20",
    "n_perm": 10000,
    "n_rewire": 1000,
    "swaps_per_edge": 10,
    "n_bins": 10,
    "n_bins_joint": 5,
    "seed": 2026,
    "quick": 0,
}

EXPECTED = {"texts": 35723, "users": 599, "edges": 7626, "pps_rows": 18000, "pps_users": 584,
            "pps_normal": 16474, "pps_offensive": 1302, "pps_hateful": 224}

NULLS = ["shuffle", "act_strat", "deg_strat", "actxdeg", "rewire"]
MAIN = "PPS labels, users with PPS texts"
REF = "all labels, the same users"


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
    ap = argparse.ArgumentParser(description="PHOND: homophily with user labels from the PPS texts only")
    for k, v in CONFIG.items():
        ap.add_argument(f"--{k}", type=type(v), default=v)
    args, _unknown = ap.parse_known_args()
    cfg = vars(args)
    if cfg["quick"]:
        cfg["n_perm"] = min(cfg["n_perm"], 300)
        cfg["n_rewire"] = min(cfg["n_rewire"], 50)
    cfg["k_list"] = [int(x) for x in str(cfg["k_list"]).split(",") if x.strip()]
    return cfg


_FA_FIX = str.maketrans({"ي": "ی", "ى": "ی", "ك": "ک", "‌": "", "‍": "",
                         "‏": "", "‎": "", "ـ": "", " ": ""})
_INVISIBLE = dict.fromkeys(map(ord, "‌‍‎‏‪‫‬‭‮﻿ـ"), None)
_DIACRITICS = re.compile(r"[ً-ٰٟ]")


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


def norm_fa(value):
    s = unicodedata.normalize("NFKC", str(value))
    s = s.replace("ي", "ی").replace("ى", "ی").replace("ك", "ک").replace("ة", "ه").replace("ۀ", "ه")
    s = s.translate(_INVISIBLE)
    s = _DIACRITICS.sub("", s)
    return re.sub(r"\s+", " ", s).strip()


def read_any(path, header="infer"):
    if os.path.splitext(path)[1].lower() in (".xlsx", ".xlsm", ".xls"):
        _import_or_install("openpyxl", "openpyxl")
        sheets = pd.read_excel(path, sheet_name=None, dtype=str, keep_default_na=False,
                               header=0 if header == "infer" else header)
        return max(sheets.values(), key=len)
    base = dict(dtype=str, keep_default_na=False, encoding="utf-8-sig", header=header)
    try:
        df = pd.read_csv(path, encoding_errors="replace", **base)
    except TypeError:
        df = pd.read_csv(path, **base)
    df.columns = [str(c).strip() for c in df.columns]
    return df


def pick_col(df, wanted, candidates=()):
    low = {str(c).strip().lower(): c for c in df.columns}
    for c in (wanted,) + tuple(candidates):
        if c and c.strip().lower() in low:
            return low[c.strip().lower()]
    return None


def load_final(cfg):
    df = read_any(cfg["csv"])
    ucol = pick_col(df, cfg["user_col"], ("User_SId", "SId"))
    lcol = pick_col(df, cfg["label_col"], ("classification", "label"))
    icol = pick_col(df, cfg["id_col"], ("Id",))
    tcol = pick_col(df, cfg["text_col"], ("Text", "TextContent"))
    if ucol is None or lcol is None:
        sys.exit(f"[error] user or label column not found in {cfg['csv']}. Columns: {list(df.columns)}")
    users = df[ucol].map(norm_id).to_numpy()
    y3 = df[lcol].map(map_label).to_numpy()
    bad = (y3 < 0) | np.isin(users, ["", "nan", "None"])
    if bad.any():
        vals = df.loc[bad, lcol].value_counts().head(10).to_dict()
        log(f"[warn] {int(bad.sum())} rows have an unknown label or an empty user id and are dropped. Values: {vals}")
    keep = ~bad
    return {
        "users": users[keep],
        "y3": y3[keep].astype(int),
        "ids": df[icol].map(norm_id).to_numpy()[keep] if icol else None,
        "texts": df[tcol].map(norm_fa).to_numpy()[keep] if tcol else None,
        "n_file": len(df),
        "cols": (ucol, lcol, icol, tcol),
    }


def match_pps(cfg, fin):
    p = read_any(cfg["pps"])
    pu = pick_col(p, cfg["user_col"], ("User_SId", "SId"))
    pi = pick_col(p, cfg["id_col"], ("Id",))
    pt = pick_col(p, cfg["text_col"], ("Text", "TextContent"))
    n = len(fin["y3"])
    taken = np.zeros(n, dtype=bool)
    pusers = p[pu].map(norm_id).to_numpy() if pu else np.array([""] * len(p), dtype=object)
    by_id = by_text = 0
    left = list(range(len(p)))
    if pi is not None and fin["ids"] is not None:
        index = {}
        for i, k in enumerate(fin["ids"]):
            if k:
                index.setdefault(k, []).append(i)
        pids = p[pi].map(norm_id).to_numpy()
        rest = []
        for r in left:
            for i in index.get(pids[r], ()):
                if not taken[i] and (pu is None or fin["users"][i] == pusers[r]):
                    taken[i] = True
                    by_id += 1
                    break
            else:
                rest.append(r)
        left = rest
    if left and pt is not None and fin["texts"] is not None and pu is not None:
        index = {}
        for i, (u, t) in enumerate(zip(fin["users"], fin["texts"])):
            if t:
                index.setdefault((u, t), []).append(i)
        ptexts = p[pt].map(norm_fa).to_numpy()
        rest = []
        for r in left:
            for i in index.get((pusers[r], ptexts[r]), ()):
                if not taken[i]:
                    taken[i] = True
                    by_text += 1
                    break
            else:
                rest.append(r)
        left = rest
    return taken, {"rows": len(p), "by_id": by_id, "by_text": by_text, "unmatched": len(left),
                   "columns": (pu, pi, pt)}


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
    df = read_any(cfg["edges"])
    cols = _pick_edge_cols(df, user_set, cfg["edge_cols"])
    header_is_data = cols is not None and all(norm_id(c) in user_set for c in cols)
    if (cols is None or header_is_data) and not cfg["edge_cols"]:
        df = read_any(cfg["edges"], header=None)
        cols = _pick_edge_cols(df, user_set, "")
    if cols is None:
        sys.exit("[error] could not find two user id columns in the edge file. "
                 f"Columns: {list(df.columns)}. Pass them with --edge_cols colA,colB")
    a = df[cols[0]].map(norm_id).to_numpy()
    b = df[cols[1]].map(norm_id).to_numpy()
    self_loop = a == b
    known = np.array([x in user_set for x in a]) & np.array([x in user_set for x in b])
    ok = known & ~self_loop
    ia = np.array([user_index[x] for x in a[ok]], dtype=np.int64)
    ib = np.array([user_index[x] for x in b[ok]], dtype=np.int64)
    pairs = np.unique(np.stack([np.minimum(ia, ib), np.maximum(ia, ib)], 1), axis=0)
    report = {"columns": [str(c) for c in cols], "raw_rows": int(len(a)), "self_loops": int(self_loop.sum()),
              "rows_with_unknown_user": int((~known & ~self_loop).sum()),
              "duplicate_rows": int(ok.sum() - len(pairs)), "unique_edges": int(len(pairs))}
    return np.ascontiguousarray(pairs.astype(np.int32)), report


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
        out.append(np.column_stack([same, _assort_cols(B0, B1)]).ravel())
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


def rewired_stats(e0, n, n_rep, swaps_per_edge, rng, stat_fn, desc):
    E = len(e0)
    if E < 2:
        return np.array([stat_fn(e0) for _ in range(n_rep)]), 0.0
    n_try = int(swaps_per_edge * E)
    stats, acc, first = [], 0.0, None
    for _ in tqdm(range(n_rep), desc=desc, mininterval=2):
        e = np.ascontiguousarray(e0.copy())
        adj = np.zeros((n, n), dtype=np.uint8)
        adj[e[:, 0], e[:, 1]] = 1
        adj[e[:, 1], e[:, 0]] = 1
        ia = rng.integers(0, E, n_try)
        ib = rng.integers(0, E, n_try)
        fl = rng.integers(0, 2, n_try).astype(np.uint8)
        acc += _SWAP(e, adj, ia, ib, fl) / n_try
        if first is None:
            first = e.copy()
        stats.append(stat_fn(e))
    d0 = np.bincount(e0.ravel(), minlength=n)
    d1 = np.bincount(first.ravel(), minlength=n)
    lo = np.minimum(first[:, 0], first[:, 1])
    hi = np.maximum(first[:, 0], first[:, 1])
    assert np.array_equal(d0, d1), "rewiring changed the degree sequence"
    assert len(np.unique(lo.astype(np.int64) * n + hi)) == E and (lo != hi).all(), "rewiring broke the simple graph"
    return np.array(stats), acc / n_rep


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
    nulls["rewire"], acc = rewired_stats(e, n, cfg["n_rewire"], cfg["swaps_per_edge"], rng,
                                         lambda ee: stat_vector(Xb, Xn, ee), f"{tag} rewire")
    rows = []
    for i, (attr, stat) in enumerate(stat_names(names_b, names_n)):
        row = {"subset": tag, "n_users": n, "n_edges": len(e), "attribute": attr, "statistic": stat,
               "observed": obs[i]}
        if stat == "same-attribute edge share":
            row["users_with_attribute"] = int(Xb[:, names_b.index(attr)].sum())
        for name in NULLS:
            col = nulls[name][:, i]
            row[f"{name}_mean"] = float(np.nanmean(col)) if np.isfinite(col).any() else float("nan")
            row[f"{name}_p"] = p_greater(col, obs[i])
        rows.append(row)
    return rows, {"n": n, "E": len(e), "rewire_acceptance": acc}


def fmt_p(p):
    if not np.isfinite(p):
        return "  nan"
    return "<.001" if p < 0.001 else f"{p:.3f}"


def print_block(rows):
    log(f"  {'attribute':<22}{'statistic':<27}{'obs':>7}" + "".join(f"{n:>17}" for n in NULLS))
    log("  " + " " * 56 + "".join(f"{'(null mean, p)':>17}" for _ in NULLS))
    for r in rows:
        line = f"  {r['attribute']:<22}{r['statistic']:<27}{r['observed']:>7.3f}"
        for n in NULLS:
            line += f"{r[f'{n}_mean']:>10.3f} {fmt_p(r[f'{n}_p']):>6}"
        log(line)


def main():
    cfg = parse_config()
    for key in ("csv", "edges", "pps"):
        if not cfg[key] or not os.path.exists(cfg[key]):
            sys.exit(f"[error] --{key} file not found: {cfg[key]!r}")
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
    rng = np.random.default_rng(cfg["seed"])
    log("PHOND: homophily with user labels from the PPS texts only")
    log(f"csv   : {cfg['csv']}\nedges : {cfg['edges']}\npps   : {cfg['pps']}\nout   : {cfg['out']}")
    if not HAVE_NUMBA and cfg["n_rewire"] > 200:
        cfg["n_rewire"] = 200
        log("[note] numba not found, so rewiring is slow; n_rewire set to 200.")
    log(f"n_perm {cfg['n_perm']}, n_rewire {cfg['n_rewire']}, numba {'yes' if HAVE_NUMBA else 'no'}")

    fin = load_final(cfg)
    users = np.unique(fin["users"])
    user_index = {u: i for i, u in enumerate(users)}
    uidx = np.array([user_index[u] for u in fin["users"]], dtype=np.int64)
    n = len(users)
    in_pps, prep = match_pps(cfg, fin)
    e, erep = load_edges(cfg, user_index)

    all_counts = np.zeros((n, 3), dtype=np.int64)
    np.add.at(all_counts, (uidx, fin["y3"]), 1)
    pps_counts = np.zeros((n, 3), dtype=np.int64)
    np.add.at(pps_counts, (uidx[in_pps], fin["y3"][in_pps]), 1)
    n_all = all_counts.sum(1)
    n_pps = pps_counts.sum(1)
    pps_users = np.where(n_pps > 0)[0]

    log("\n=== Checks ===")
    checks = [
        ("texts in the final dataset", len(fin["y3"]), EXPECTED["texts"]),
        ("users", n, EXPECTED["users"]),
        ("edges (unique, undirected)", len(e), EXPECTED["edges"]),
        ("rows in the PPS file", prep["rows"], EXPECTED["pps_rows"]),
        ("PPS texts found in the final dataset", int(in_pps.sum()), EXPECTED["pps_rows"]),
        ("users with PPS texts", len(pps_users), EXPECTED["pps_users"]),
        ("PPS texts labeled normal", int(pps_counts[:, 0].sum()), EXPECTED["pps_normal"]),
        ("PPS texts labeled offensive", int(pps_counts[:, 1].sum()), EXPECTED["pps_offensive"]),
        ("PPS texts labeled hateful", int(pps_counts[:, 2].sum()), EXPECTED["pps_hateful"]),
    ]
    bad = 0
    for name, got, want in checks:
        ok = got == want
        bad += not ok
        log(f"  {name:<40}{got:>8,}   expected {want:>8,}   {'ok' if ok else '<-- differs'}")
    log(f"  final dataset columns: user {fin['cols'][0]}, label {fin['cols'][1]}, Id {fin['cols'][2]}, "
        f"text {fin['cols'][3]}")
    log(f"  PPS file: matched {prep['by_id']:,} by Id and {prep['by_text']:,} by user and text, "
        f"{prep['unmatched']:,} not found (columns user {prep['columns'][0]}, Id {prep['columns'][1]}, "
        f"text {prep['columns'][2]})")
    log(f"  edge file: columns {erep['columns']}, {erep['raw_rows']:,} rows, {erep['duplicate_rows']} duplicate rows, "
        f"{erep['self_loops']} self loops, {erep['rows_with_unknown_user']} rows with a user not in the dataset")
    if bad:
        log(f"  [warn] {bad} values differ from the expected counts. Check the three files before using "
            "the results.")

    any_hate_pps = (pps_counts[:, 2] > 0).astype(int)
    any_harm_pps = ((pps_counts[:, 1] + pps_counts[:, 2]) > 0).astype(int)
    any_hate_all = (all_counts[:, 2] > 0).astype(int)
    any_harm_all = ((all_counts[:, 1] + all_counts[:, 2]) > 0).astype(int)
    S = pps_users
    log(f"\n  Among the {len(S)} users with PPS texts (median {np.median(n_pps[S]):.0f} PPS texts per user):")
    log(f"    any hateful text:              {int(any_hate_pps[S].sum()):>4} users with PPS labels, "
        f"{int(any_hate_all[S].sum()):>4} with all labels")
    log(f"    any offensive or hateful text: {int(any_harm_pps[S].sum()):>4} users with PPS labels, "
        f"{int(any_harm_all[S].sum()):>4} with all labels")

    log("\n=== Homophily under the five null models of Table 4 ===")
    log("  p = one sided share of null draws at or above the observed value; '<.001' means none in the draws.")
    log("  shuffle = plain label shuffle; act_strat / deg_strat = shuffle within deciles of activity / degree;")
    log("  actxdeg = within 5 x 5 activity by degree cells; rewire = degree preserving rewiring, labels fixed.")
    log("  Activity is the number of PPS texts with PPS labels, and the number of all texts with all labels.")
    hate_rate = pps_counts[:, 2] / np.maximum(n_pps, 1)
    harm_rate = (pps_counts[:, 1] + pps_counts[:, 2]) / np.maximum(n_pps, 1)
    blocks = [(MAIN, S, 1)] + [(f"PPS labels, users with >= {k} PPS texts", np.where(n_pps >= k)[0], k)
                               for k in cfg["k_list"]]
    all_rows = []
    for tag, nodes, k in blocks:
        log(f"\n  [{tag}]")
        if len(nodes) < 20:
            log(f"  only {len(nodes)} users, skipped")
            continue
        bin_attrs = {"any hateful": any_hate_pps, "any off. or hateful": any_harm_pps}
        num_attrs = {}
        if k > 1:
            mh = float(np.median(hate_rate[nodes]))
            mm = float(np.median(harm_rate[nodes]))
            bin_attrs["hate rate > median"] = (hate_rate > mh).astype(int)
            bin_attrs["harm rate > median"] = (harm_rate > mm).astype(int)
            num_attrs = {"hate rate": hate_rate, "harm rate": harm_rate}
            log(f"  median hate rate {mh:.4f} (share above {np.mean(hate_rate[nodes] > mh):.3f}), "
                f"median offensive-or-hateful rate {mm:.4f} (share above {np.mean(harm_rate[nodes] > mm):.3f})")
        rows, info = homophily_block(tag, nodes, e, n, bin_attrs, num_attrs, n_pps, cfg, rng)
        log(f"  {info['n']} users, {info['E']} edges, rewiring acceptance {info['rewire_acceptance']:.2f}")
        print_block(rows)
        all_rows += rows
    log(f"\n  [{REF}]")
    rows, info = homophily_block(REF, S, e, n, {"any hateful": any_hate_all, "any off. or hateful": any_harm_all},
                                 {}, n_all, cfg, rng)
    log(f"  {info['n']} users, {info['E']} edges, rewiring acceptance {info['rewire_acceptance']:.2f}")
    print_block(rows)
    all_rows += rows
    pd.DataFrame(all_rows).to_csv(os.path.join(cfg["out"], "homophily_pps.csv"), index=False)

    df = pd.DataFrame(all_rows)

    def get(subset, attr, stat):
        r = df[(df.subset == subset) & (df.attribute == attr) & (df.statistic == stat)]
        return r.iloc[0] if len(r) else None

    log("\n=== Summary ===")
    held = {}
    for attr, words in (("any hateful", "a hateful text"), ("any off. or hateful", "an offensive or hateful text")):
        r = get(MAIN, attr, "same-attribute edge share")
        a = get(MAIN, attr, "assortativity")
        ref = get(REF, attr, "same-attribute edge share")
        means = [r[f"{nm}_mean"] for nm in NULLS]
        ps = [r[f"{nm}_p"] for nm in NULLS]
        worst = max(ps)
        failed = [nm for nm, p in zip(NULLS, ps) if not (p < 0.05)]
        held[attr] = (r, a, worst, failed, words)
        log(f"  {attr}, PPS labels: {100 * r['observed']:.1f}% same-attribute edges "
            f"({int(r['users_with_attribute'])} of {int(r['n_users'])} users have {words}); "
            f"null means {100 * min(means):.1f}% to {100 * max(means):.1f}%; largest p {fmt_p(worst).strip()}")
        log(f"      assortativity {a['observed']:.3f}, null means {min(a[f'{nm}_mean'] for nm in NULLS):.3f} to "
            f"{max(a[f'{nm}_mean'] for nm in NULLS):.3f}; with all labels on the same users "
            f"{100 * ref['observed']:.1f}% same-attribute edges")
        log("      -> " + ("above all five nulls (p < 0.05 for each)" if not failed
                           else "not above: " + ", ".join(failed)))
    rate_ok = []
    for k in cfg["k_list"]:
        tag = f"PPS labels, users with >= {k} PPS texts"
        r = get(tag, "hate rate", "assortativity (numeric)")
        if r is None:
            continue
        ps = [r[f"{nm}_p"] for nm in NULLS]
        means = [r[f"{nm}_mean"] for nm in NULLS]
        failed = [nm for nm, p in zip(NULLS, ps) if not (p < 0.05)]
        log(f"  >= {k} PPS texts ({int(r['n_users'])} users), share of hateful PPS texts: assortativity "
            f"{r['observed']:.3f}, null means {min(means):.3f} to {max(means):.3f}; largest p {fmt_p(max(ps)).strip()}")
        m = get(tag, "hate rate > median", "same-attribute edge share")
        if m is not None:
            mm = [m[f"{nm}_mean"] for nm in NULLS]
            log(f"      same side of the median: {100 * m['observed']:.1f}% of edges, null means "
                f"{100 * min(mm):.1f}% to {100 * max(mm):.1f}%")
        log("      -> " + ("above all five nulls (p < 0.05 for each)" if not failed else "not above: " + ", ".join(failed)))
        if not failed:
            rate_ok.append((k, r, max(ps), min(means), max(means)))


if __name__ == "__main__":
    main()
