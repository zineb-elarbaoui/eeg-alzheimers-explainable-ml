

import argparse
import importlib.util
import re
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import hypergeom, spearmanr


def load_module(name, filename):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).parent / filename)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


f4 = load_module("loocv4", "04_loocv_framework.py")

K_MAX = 16
K_RANGE = list(range(4, 17))
LEVELS = ("family_region", "region", "family", "band")


# ----------------------------------------------------------------------------------------------
# selection (fold-wise, training data only; paper's full pipeline)
# ----------------------------------------------------------------------------------------------
def select_names(X_raw, y01, names, k):
    """impute (training medians) -> drop zero-variance -> f4.select_features. Names in selection order."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        med = np.nanmedian(X_raw, axis=0)
    Xi = np.where(np.isnan(X_raw), med, X_raw)
    keep = Xi.var(axis=0) > 0
    Xi, kept = Xi[:, keep], np.asarray(names)[keep]
    idx = f4.select_features(Xi, y01, k=k, mode="full")
    return list(kept[idx])


def loocv_selections(X, y01, names, k=K_MAX):
    n = X.shape[0]
    return [select_names(X[np.arange(n) != i], y01[np.arange(n) != i], names, k) for i in range(n)]


def subsample_selections(X, y01, names, n_sub, frac, seed, k=K_MAX):
    rng = np.random.default_rng(seed)
    pos, neg = np.where(y01 == 1)[0], np.where(y01 == 0)[0]
    out = []
    for _ in range(n_sub):
        idx = np.concatenate([rng.choice(pos, int(round(frac * len(pos))), replace=False),
                              rng.choice(neg, int(round(frac * len(neg))), replace=False)])
        out.append(select_names(X[idx], y01[idx], names, k))
    return out


# ----------------------------------------------------------------------------------------------
# stability statistics
# ----------------------------------------------------------------------------------------------
def pairwise_jaccard(S):
    S = S.astype(float)
    inter = S @ S.T
    sz = S.sum(axis=1)
    union = sz[:, None] + sz[None, :] - inter
    iu = np.triu_indices(len(S), 1)
    return inter[iu] / np.maximum(union[iu], 1e-12)


def chance_jaccard(k, p):
    """Exact expected Jaccard of two independent random k-subsets of p features."""
    m = np.arange(0, k + 1)
    return float((hypergeom.pmf(m, p, k, k) * m / (2 * k - m)).sum())


def membership(sels, k, index, n_items):
    S = np.zeros((len(sels), n_items), dtype=bool)
    for i, s in enumerate(sels):
        S[i, [index[n] for n in s[:k]]] = True
    return S


def feature_stats(sels, k, index, p):
    S = membership(sels, k, index, p)
    J = pairwise_jaccard(S)
    f = S.mean(axis=0)
    return {"mean_jaccard": J.mean(), "sd_jaccard": J.std(), "min_jaccard": J.min(),
            "n_unique": int((f > 0).sum()), "n_ge90": int((f >= 0.90).sum()), "n_ge95": int((f >= 0.95).sum()),
            "chance_jaccard": chance_jaccard(k, p)}, f


def mean_position(sels, k, names):
    tot, cnt = {}, {}
    for s in sels:
        for pos, n in enumerate(s[:k], 1):
            tot[n] = tot.get(n, 0) + pos
            cnt[n] = cnt.get(n, 0) + 1
    return np.array([tot[n] / cnt[n] if n in cnt else np.nan for n in names])


# ----------------------------------------------------------------------------------------------
# feature-name parsing (heuristic: edit here if your column names differ)
# ----------------------------------------------------------------------------------------------
BANDS = ("delta", "theta", "alpha", "beta")
CHANNEL_REGION = {"fp1": "frontal", "fp2": "frontal", "f3": "frontal", "f4": "frontal", "f7": "frontal",
                  "f8": "frontal", "t3": "temporal", "t4": "temporal", "t5": "temporal", "t6": "temporal",
                  "c3": "central", "c4": "central", "p3": "parietal", "p4": "parietal",
                  "o1": "occipital", "o2": "occipital", "fz": "midline", "cz": "midline", "pz": "midline"}
REGION_PREFIX = (("front", "frontal"), ("temp", "temporal"), ("cent", "central"), ("pari", "parietal"),
                 ("occ", "occipital"), ("mid", "midline"), ("glob", "global"), ("all", "global"))


def parse_feature(name):
    s = name.lower()
    toks = [t for t in re.split(r"[^a-z0-9]+", s) if t]
    band = next((b for b in BANDS if b in s), "none")

    if "asym" in s:
        fam = "asymmetry"
    elif "sfr" in toks or "slow" in s:
        fam, band = "slow/fast", "ratio"
    elif "tar" in toks or ("theta" in s and "alpha" in s):
        fam, band = "theta/alpha", "ratio"
    elif "tbr" in toks or ("theta" in s and "beta" in s):
        fam, band = "theta/beta", "ratio"
    elif ("entropy" in s and "perm" in s) or "pe" in toks:
        fam = "perm_entropy"
    elif ("entropy" in s and "diff" in s) or "de" in toks or "diffent" in s:
        fam = "diff_entropy"
    elif "rel" in s:
        fam = "rel_power"
    elif "abs" in s:
        fam = "abs_power"
    else:
        fam = "other"

    region = "unknown"
    for t in toks:
        if t in CHANNEL_REGION:
            region = CHANNEL_REGION[t]
            break
        hit = next((r for pre, r in REGION_PREFIX if t.startswith(pre)), None)
        if hit:
            region = hit
            break
    return fam, band, region


def build_meta(names):
    meta = pd.DataFrame([(n,) + parse_feature(n) for n in names], columns=["feature", "family", "band", "region"])
    meta["family_region"] = meta["family"] + "|" + meta["region"]
    return meta


def group_membership(sels, k, f2g, n_groups):
    G = np.zeros((len(sels), n_groups), dtype=bool)
    C = np.zeros((len(sels), n_groups))
    for i, s in enumerate(sels):
        for n in s[:k]:
            G[i, f2g[n]] = True
            C[i, f2g[n]] += 1
    return G, C


def group_chance_jaccard(group_of_feature, n_groups, k, n_sets=1000, seed=0):
    """Mean pairwise Jaccard of the GROUP sets reached by random k-feature draws (chance baseline)."""
    rng = np.random.default_rng(seed)
    p = len(group_of_feature)
    G = np.zeros((n_sets, n_groups), dtype=bool)
    for i in range(n_sets):
        G[i, group_of_feature[rng.choice(p, k, replace=False)]] = True
    return float(pairwise_jaccard(G).mean())


# ----------------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="outputs/features.csv")
    ap.add_argument("--k", type=int, default=12, help="reference k for the headline tables (paper: 12)")
    ap.add_argument("--n_subsamples", type=int, default=100)
    ap.add_argument("--subsample_frac", type=float, default=0.80)
    ap.add_argument("--subsample_seed", type=int, default=2026)
    ap.add_argument("--skip_subsample", action="store_true")
    ap.add_argument("--plots", action="store_true")
    ap.add_argument("--show_names", action="store_true")
    ap.add_argument("--out_dir", default="outputs/step7_stability")
    args = ap.parse_args()

    df = pd.read_csv(args.features)
    cols = [c for c in df.columns if c not in {"subject_id", "label", "n_epochs_used"}]
    X = df[cols].to_numpy(dtype=float)
    y01 = (df["label"].to_numpy() == "AD").astype(int)
    p = len(cols)
    index = {c: i for i, c in enumerate(cols)}
    meta = build_meta(cols)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    meta.to_csv(out / "feature_groups.csv", index=False)

    print(f"Loaded {args.features}: {X.shape[0]} subjects x {p} features ({y01.sum()} AD, {(1 - y01).sum()} HC)")
    print(f"Name parsing: {int((meta.family == 'other').sum())} features with unparsed family, "
          f"{int((meta.region == 'unknown').sum())} with unparsed region (see {out / 'feature_groups.csv'})")
    if args.show_names:
        print(meta.groupby(["family", "band"]).size().to_string())
        print(meta.groupby("region").size().to_string())
        print(meta.sample(min(15, p), random_state=0).to_string(index=False))
        return
    print()

    # ---- prefix property: selection at k == first k of the selection at K_MAX ----
    tr = np.arange(len(y01)) != 0
    a = select_names(X[tr], y01[tr], cols, args.k)
    b = select_names(X[tr], y01[tr], cols, K_MAX)[:args.k]
    assert a == b, "prefix property violated"
    print(f"Check passed: selection at k={args.k} equals the first {args.k} of the k={K_MAX} selection "
          f"(so every k in {K_RANGE[0]}..{K_RANGE[-1]} comes from one run).\n")

    # ---- selections ----
    sels = {}
    t0 = time.time()
    sels["LOOCV"] = loocv_selections(X, y01, cols)
    print(f"  LOOCV selections ({len(y01)} folds, {time.time() - t0:.1f}s)")
    if not args.skip_subsample:
        t0 = time.time()
        sels["SUBSAMPLE"] = subsample_selections(X, y01, cols, args.n_subsamples, args.subsample_frac,
                                                 args.subsample_seed)
        print(f"  Subsample selections ({args.n_subsamples} x {args.subsample_frac:.0%}, {time.time() - t0:.1f}s)")
    print()

    # ---- feature level, all k ----
    rows, freq = [], {}
    for scope, ss in sels.items():
        for k in K_RANGE:
            st, f = feature_stats(ss, k, index, p)
            rows.append({"scope": scope, "k": k, **st})
            if k == args.k:
                freq[scope] = f
    stab = pd.DataFrame(rows)
    stab.to_csv(out / "stability_vs_k.csv", index=False)

    ft = meta[["feature", "family", "band", "region"]].copy()
    for scope, f in freq.items():
        ft[f"freq_{scope}"] = f
    ft["mean_position_LOOCV"] = mean_position(sels["LOOCV"], args.k, cols)
    ft = ft.sort_values("freq_LOOCV", ascending=False)
    ft.to_csv(out / f"selection_frequency_k{args.k}.csv", index=False)

    # ---- group level, reference k ----
    grows, lvl_rows = [], []
    for level in LEVELS:
        groups = sorted(meta[level].unique())
        gidx = {g: i for i, g in enumerate(groups)}
        f2g = {n: gidx[g] for n, g in zip(meta["feature"], meta[level])}
        g_of_f = np.array([gidx[g] for g in meta[level]])
        chance = group_chance_jaccard(g_of_f, len(groups), args.k)
        row = {"level": level, "n_groups": len(groups), "chance_jaccard": chance}
        for scope, ss in sels.items():
            G, C = group_membership(ss, args.k, f2g, len(groups))
            row[f"jaccard_{scope}"] = pairwise_jaccard(G).mean()
            for g, fr, cn in zip(groups, G.mean(axis=0), C.mean(axis=0)):
                grows.append({"scope": scope, "level": level, "group": g, "freq": fr, "mean_count_per_set": cn})
        lvl_rows.append(row)
    gdf = pd.DataFrame(grows)
    gdf.to_csv(out / f"group_frequency_k{args.k}.csv", index=False)
    lvl = pd.DataFrame(lvl_rows)
    lvl.to_csv(out / f"group_level_jaccard_k{args.k}.csv", index=False)

    # ---------------------------------------------------------------------------------------
    # report
    # ---------------------------------------------------------------------------------------
    pd.set_option("display.width", 200)
    fm = lambda v: f"{v:.3f}"
    print(f"=== Feature-level stability, full pipeline, k={args.k} ===")
    ref = stab[stab.k == args.k][["scope", "mean_jaccard", "sd_jaccard", "min_jaccard", "chance_jaccard",
                                  "n_unique", "n_ge90", "n_ge95"]]
    print(ref.to_string(index=False, float_format=fm))
    print("Paper (63 subjects, k=12): 38 unique features, 3 selected in >95% of folds.")

    print(f"\n=== Most consistently selected features, k={args.k} ===")
    show = ["feature", "freq_LOOCV", "mean_position_LOOCV"] + (["freq_SUBSAMPLE"] if "SUBSAMPLE" in sels else [])
    print(ft[show].head(15).to_string(index=False, float_format=fm))

    if "SUBSAMPLE" in sels:
        a_, b_ = ft["freq_LOOCV"].to_numpy(), ft["freq_SUBSAMPLE"].to_numpy()
        top = lambda f_: set(np.argsort(-f_)[:args.k])
        print(f"\nLOOCV vs subsample frequencies: Spearman rho = {spearmanr(a_, b_)[0]:.3f}, "
              f"top-{args.k} overlap = {len(top(a_) & top(b_))}/{args.k}")

    print(f"\n=== Mean pairwise Jaccard by grouping level, k={args.k} (feature level shown for reference) ===")
    feat_row = {"level": "feature", "n_groups": p, "chance_jaccard": chance_jaccard(args.k, p)}
    for scope in sels:
        feat_row[f"jaccard_{scope}"] = stab[(stab.scope == scope) & (stab.k == args.k)].mean_jaccard.iloc[0]
    print(pd.concat([pd.DataFrame([feat_row]), lvl]).to_string(index=False, float_format=fm))

    for level in ("family_region", "family", "region"):
        print(f"\n=== Group selection frequency: {level} (LOOCV | subsample), k={args.k} ===")
        g = gdf[gdf.level == level].pivot(index="group", columns="scope", values="freq")
        print(g.sort_values("LOOCV", ascending=False).head(10 if level == "family_region" else 12)
              .to_string(float_format=fm))

    print("\n=== Mean pairwise Jaccard vs k ===")
    pv = stab.pivot(index="k", columns="scope", values="mean_jaccard")
    pv["chance"] = stab[stab.scope == "LOOCV"].set_index("k")["chance_jaccard"]
    print(pv.to_string(float_format=fm))

    print("\nHow to read this:")
    print("  * LOOCV Jaccard is optimistic (training sets share 63/64 subjects); compare with the subsample row.")
    print("  * Group-level Jaccard is mechanically higher than feature-level; compare each level with its own chance value.")
    print("  * If group-level overlap is well above its chance level while feature-level overlap is low,")
    print("    correlated features are substituting for each other (same signal, different feature).")
    print(f"\nAll outputs written to {out}/")

    if args.plots:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib not available; skipping plots")
            return
        fig, ax = plt.subplots(figsize=(6, 4))
        for scope, d in stab.groupby("scope"):
            ax.plot(d.k, d.mean_jaccard, marker="o", ms=3, label=scope.lower(),
                    ls="-" if scope == "LOOCV" else "--")
        ax.plot(K_RANGE, [chance_jaccard(k, p) for k in K_RANGE], "k:", label="random selection")
        ax.set_xlabel("k"); ax.set_ylabel("mean pairwise Jaccard"); ax.legend(fontsize=8); ax.set_ylim(0, 1)
        fig.tight_layout(); fig.savefig(out / "jaccard_vs_k.png", dpi=200); plt.close(fig)

        top15 = ft.head(15).iloc[::-1]
        fig, ax = plt.subplots(figsize=(7, 5))
        yy = np.arange(len(top15))
        ax.barh(yy + 0.2, top15["freq_LOOCV"], 0.4, label="LOOCV folds")
        if "freq_SUBSAMPLE" in top15:
            ax.barh(yy - 0.2, top15["freq_SUBSAMPLE"], 0.4, label=f"{args.subsample_frac:.0%} subsamples")
        ax.set_yticks(yy); ax.set_yticklabels(top15["feature"], fontsize=7)
        ax.set_xlabel("selection frequency"); ax.legend(fontsize=8)
        fig.tight_layout(); fig.savefig(out / "top_feature_frequency.png", dpi=200); plt.close(fig)


if __name__ == "__main__":
    main()