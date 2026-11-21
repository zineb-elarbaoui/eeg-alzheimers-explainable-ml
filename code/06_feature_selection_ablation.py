

import argparse
import importlib.util
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.exceptions import ConvergenceWarning


def load_module(name, filename):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).parent / filename)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


f4 = load_module("loocv4", "04_loocv_framework.py")
f5 = load_module("loocv5", "05_run_loocv.py")     # reuse the paper's four model definitions (MODELS)

VARIANTS = [
    {"name": "full",       "use_sel": True,  "mode": "full",       "seeded": False},
    {"name": "no_pruning", "use_sel": True,  "mode": "no_pruning", "seeded": False},
    {"name": "no_ranking", "use_sel": True,  "mode": "no_ranking", "seeded": True},
    {"name": "none",       "use_sel": False, "mode": "full",       "seeded": False},
]


# ----------------------------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------------------------
def run_one(X, y, ids, cols, make_clf, k, variant, seed):
    """One full LOOCV run; returns (fold results, number of solver convergence warnings)."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        results = f4.run_loocv(X, y, ids, cols, make_clf, k=k,
                               use_feature_selection=variant["use_sel"], selection_mode=variant["mode"],
                               selection_seed=2026 if seed is None else seed)
    n_conv = sum(issubclass(w.category, ConvergenceWarning) for w in caught)
    return results, n_conv


def redundancy_stats(results, R_abs, name2idx):
    """Mean / max pairwise |Spearman rho| among each fold's selected features (descriptive, all subjects)."""
    means, maxs, uniq = [], [], set()
    for r in results:
        idx = [name2idx[n] for n in r.selected_features]
        uniq.update(idx)
        if len(idx) < 2:
            continue
        vals = R_abs[np.ix_(idx, idx)][np.triu_indices(len(idx), 1)]
        means.append(vals.mean())
        maxs.append(vals.max())
    return (float(np.mean(means)) if means else np.nan, float(np.mean(maxs)) if maxs else np.nan, len(uniq))


class Bootstrap:
    """Stratified subject bootstrap on out-of-fold scores; one shared index matrix => paired comparisons."""

    def __init__(self, y01, B=2000, seed=0):
        rng = np.random.default_rng(seed)
        pos, neg = np.where(y01 == 1)[0], np.where(y01 == 0)[0]
        self.n1, self.n0 = len(pos), len(neg)
        self.idx = np.concatenate([rng.choice(pos, (B, self.n1)), rng.choice(neg, (B, self.n0))], axis=1)

    def aucs(self, score):
        r = rankdata(np.asarray(score)[self.idx], axis=1)          # positives occupy the first n1 columns
        return (r[:, :self.n1].sum(axis=1) - self.n1 * (self.n1 + 1) / 2) / (self.n1 * self.n0)


def verdict(lo, hi):
    return "better" if lo > 0 else ("worse" if hi < 0 else "within noise")


# ----------------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="outputs/features.csv")
    ap.add_argument("--ks", type=int, nargs="+", default=[12], help="number(s) of selected features k (paper: 12)")
    ap.add_argument("--seeds", type=int, default=5, help="independent random orders for the no_ranking variant")
    ap.add_argument("--boot", type=int, default=2000, help="bootstrap resamples")
    ap.add_argument("--models", nargs="+", default=None, help="subset of model names (default: all four)")
    ap.add_argument("--step5_dir", default="outputs/step5")
    ap.add_argument("--out_dir", default="outputs/step6_ablation")
    args = ap.parse_args()

    df = pd.read_csv(args.features)
    cols = [c for c in df.columns if c not in {"subject_id", "label", "n_epochs_used"}]
    X = df[cols].to_numpy(dtype=float)
    y = df["label"].to_numpy()
    y01 = (y == "AD").astype(int)
    ids = df["subject_id"].tolist()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    print(f"Loaded {args.features}: {X.shape[0]} subjects x {X.shape[1]} features "
          f"({(y == 'AD').sum()} AD, {(y == 'HC').sum()} HC)\n")

    models = {m: f for m, f in f5.MODELS.items() if args.models is None or m in args.models}
    boot = Bootstrap(y01, B=args.boot)
    X_imp = np.where(np.isnan(X), np.nanmedian(X, axis=0), X)
    R_abs = np.abs(np.nan_to_num(f4.spearman_corr_matrix(X_imp), nan=0.0))     # descriptive only
    name2idx = {c: i for i, c in enumerate(cols)}
    seeds = [2026 + s for s in range(args.seeds)]

    rows = []
    for mname, make_clf in models.items():
        print(f"=== {mname} ===")
        none_cache = None
        for k in args.ks:
            full_boot = None
            for v in VARIANTS:
                if v["name"] == "none" and none_cache is not None:          # k-independent: run once per model
                    runs, convs = none_cache
                else:
                    runs, convs = [], 0
                    for seed in (seeds if v["seeded"] else [None]):
                        t0 = time.time()
                        res, nc = run_one(X, y, ids, cols, make_clf, k, v, seed)
                        runs.append(res)
                        convs += nc
                        tag = f"seed {seed}" if seed else ""
                        auc_i = f4.summarize(res)["AUC"]
                        print(f"  k={k:<3d} {v['name']:<11s} {tag:<10s} AUC={auc_i:.3f}  [{time.time() - t0:.1f}s]"
                              + (f"  ({nc} solver convergence warnings)" if nc else ""))
                        stem = f"{mname.replace(' ', '_')}_{v['name']}" + ("" if v["name"] == "none" else f"_k{k}") \
                               + (f"_seed{seed}" if seed else "")
                        pd.DataFrame([{"subject_id": r.subject_id, "y_true": r.y_true, "y_pred": r.y_pred,
                                       "score": r.proba, "selected_features": "|".join(r.selected_features)}
                                      for r in res]).to_csv(out / f"predictions_{stem}.csv", index=False)
                        if seed in (None, seeds[0]):                          # coefficients: first run of each variant
                            pd.DataFrame([{"subject_id": r.subject_id, "feature": f, "coef": c}
                                          for r in res for f, c in r.coefficients.items()]
                                         ).to_csv(out / f"coefficients_{stem}.csv", index=False)
                    if v["name"] == "none":
                        none_cache = (runs, convs)

                mets = [f4.summarize(r) for r in runs]
                aucs = np.array([m["AUC"] for m in mets])
                red = np.array([redundancy_stats(r, R_abs, name2idx) for r in runs])
                n_feat = float(np.mean([np.mean([len(fr.selected_features) for fr in r]) for r in runs]))
                bvec = np.mean([boot.aucs([fr.proba for fr in r]) for r in runs], axis=0)   # seed-averaged for no_ranking
                if v["name"] == "full":
                    full_boot, full_auc = bvec, float(aucs.mean())
                dlt = bvec - full_boot
                lo, hi = np.percentile(dlt, [2.5, 97.5])
                alo, ahi = np.percentile(bvec, [2.5, 97.5])
                rows.append({
                    "Model": mname, "k": ("all" if v["name"] == "none" else k), "k_ref": k, "variant": v["name"],
                    "n_runs": len(runs), "AUC": aucs.mean(), "AUC_seed_sd": aucs.std(ddof=1) if len(runs) > 1 else np.nan,
                    "AUC_ci_lo": alo, "AUC_ci_hi": ahi,
                    "delta_vs_full": aucs.mean() - full_auc, "delta_lo": lo, "delta_hi": hi,
                    "verdict": "reference" if v["name"] == "full" else verdict(lo, hi),
                    "ACC": np.mean([m["ACC"] for m in mets]), "SEN": np.mean([m["SEN"] for m in mets]),
                    "SPE": np.mean([m["SPE"] for m in mets]), "F1": np.mean([m["F1"] for m in mets]),
                    "n_features": n_feat, "mean_abs_rho": red[:, 0].mean(), "mean_max_rho": red[:, 1].mean(),
                    "n_unique_features": red[:, 2].mean(), "convergence_warnings": convs,
                })
        print()

    summary = pd.DataFrame(rows)
    summary.to_csv(out / "ablation_summary.csv", index=False)

    # ---- integrity check: the `full` variant must reproduce Step 5 exactly (same code path, deterministic) ----
    s5 = Path(args.step5_dir) / "loocv_summary_k12.csv"
    if s5.exists() and 12 in args.ks:
        ref = pd.read_csv(s5).set_index("Model")["AUC"]
        chk = summary[(summary.variant == "full") & (summary.k == 12)].set_index("Model")["AUC"]
        diffs = {m: abs(chk[m] - ref[m]) for m in chk.index if m in ref.index}
        ok = all(d < 1e-9 for d in diffs.values())
        print("Integrity check vs Step 5:", "OK - the `full` variant reproduces Step 5 exactly." if ok else
              f"WARNING - `full` differs from Step 5 ({ {m: round(d, 4) for m, d in diffs.items()} }); "
              "was features.csv regenerated since Step 5?")
        print()

    # ---- report ----
    show = ["variant", "AUC", "AUC_seed_sd", "delta_vs_full", "delta_lo", "delta_hi", "verdict",
            "ACC", "SEN", "SPE", "n_features", "mean_abs_rho", "n_unique_features"]
    for (mname, k), g in summary.groupby(["Model", "k_ref"], sort=False):
        print(f"--- {mname}, k={k} (95% interval = paired bootstrap on delta-AUC vs the full pipeline) ---")
        print(g[show].to_string(index=False, float_format=lambda v: f"{v:.3f}", na_rep="-"))
        print()

    half = ((summary[summary.variant == "full"]["AUC_ci_hi"] - summary[summary.variant == "full"]["AUC_ci_lo"]) / 2).mean()
    print(f"Resolution of this cohort: the full pipeline's AUC has a 95% bootstrap half-width of about +/-{half:.3f}.")
    print("Differences far smaller than that cannot be told apart from noise with 65 subjects.\n")

    nz = summary[(summary.variant != "full")]
    for vname, g in nz.groupby("variant", sort=False):
        c = g["verdict"].value_counts().to_dict()
        print(f"{vname:<11s} vs full over {len(g)} model/k combinations: "
              + ", ".join(f"{n} {lab}" for lab, n in c.items()) + f"  (mean delta-AUC {g['delta_vs_full'].mean():+.3f})")
    print(f"\nAll outputs written to {out}/  (summary CSV + per-run predictions and coefficients for Steps 7-8)")


if __name__ == "__main__":
    main()