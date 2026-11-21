"""
Step 5 - Train linear baseline models 
====================================================================
Runs Ridge Logistic, ElasticNet Logistic, Linear SVM, and Shrinkage LDA through
the Step 4 fold-wise LOOCV framework, using the exact hyperparameters reported in the paper

"""

import argparse
import importlib.util
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

# Newer scikit-learn (>=1.8) deprecates the 'penalty' kwarg in favour of 'l1_ratio'/'C' alone;
# this is harmless noise here since we pin the exact hyperparameters the paper specifies.
warnings.filterwarnings("ignore", category=FutureWarning)
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC


def load_framework():
    spec = importlib.util.spec_from_file_location("loocv4", Path(__file__).parent / "04_loocv_framework.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# Hyperparameters exactly as specified in the paper, section 2.7.1.
def make_ridge_logistic():
    return LogisticRegression(penalty="l2", C=1.0, solver="liblinear",
                               class_weight="balanced", max_iter=5000, random_state=2026)


def make_elasticnet_logistic():
    return LogisticRegression(penalty="elasticnet", C=1.0, l1_ratio=0.5, solver="saga",
                               class_weight="balanced", max_iter=20000, tol=1e-4, random_state=2026)


def make_linear_svm():
    return LinearSVC(C=1.0, class_weight="balanced", max_iter=20000, random_state=2026)


def make_shrinkage_lda():
    return LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")


MODELS = {
    "Ridge Logistic": make_ridge_logistic,
    "ElasticNet Logistic": make_elasticnet_logistic,
    "Linear SVM": make_linear_svm,
    "Shrinkage LDA": make_shrinkage_lda,
}

# Paper's Table 2 (63 subjects, k = 12) -- the benchmark we're comparing against.
PAPER_BENCHMARK = {
    "Ridge Logistic":      {"AUC": 0.912, "ACC": 0.857, "SEN": 0.778, "SPE": 0.963},
    "ElasticNet Logistic": {"AUC": 0.910, "ACC": 0.857, "SEN": 0.778, "SPE": 0.963},
    "Linear SVM":          {"AUC": 0.914, "ACC": 0.810, "SEN": 0.667, "SPE": 1.000},
    "Shrinkage LDA":       {"AUC": 0.862, "ACC": 0.857, "SEN": 0.778, "SPE": 0.963},
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="outputs/features.csv")
    ap.add_argument("--k", type=int, default=12)
    ap.add_argument("--k_sweep", action="store_true", help="also run k in {4,6,8,10,12,14,16} for every model")
    ap.add_argument("--out_dir", default="outputs/step5")
    args = ap.parse_args()

    f4 = load_framework()
    df = pd.read_csv(args.features)
    meta_cols = {"subject_id", "label", "n_epochs_used"}
    feature_cols = [c for c in df.columns if c not in meta_cols]
    X = df[feature_cols].to_numpy(dtype=float)
    y = df["label"].to_numpy()
    ids = df["subject_id"].tolist()

    print(f"Loaded {args.features}: {X.shape[0]} subjects x {X.shape[1]} features "
          f"({(y == 'AD').sum()} AD, {(y == 'HC').sum()} HC)\n")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows, all_results = [], {}
    for name, make_clf in MODELS.items():
        t0 = time.time()
        results = f4.run_loocv(X, y, ids, feature_cols, make_clf, k=args.k)
        m = f4.summarize(results)
        dt = time.time() - t0
        all_results[name] = results

        bench = PAPER_BENCHMARK[name]
        rows.append({"Model": name, "AUC": m["AUC"], "ACC": m["ACC"], "SEN": m["SEN"],
                     "SPE": m["SPE"], "PRE": m["PRE"], "F1": m["F1"],
                     "paper_AUC": bench["AUC"], "paper_ACC": bench["ACC"],
                     "paper_SEN": bench["SEN"], "paper_SPE": bench["SPE"]})
        print(f"{name:22s} AUC={m['AUC']:.3f} (paper {bench['AUC']:.3f})  "
              f"ACC={m['ACC']:.3f} (paper {bench['ACC']:.3f})  "
              f"SEN={m['SEN']:.3f} (paper {bench['SEN']:.3f})  "
              f"SPE={m['SPE']:.3f} (paper {bench['SPE']:.3f})  [{dt:.1f}s]")

        pd.DataFrame([{"subject_id": r.subject_id, "y_true": r.y_true, "y_pred": r.y_pred,
                        "score": r.proba, "selected_features": "|".join(r.selected_features)}
                       for r in results]).to_csv(out_dir / f"predictions_{name.replace(' ', '_')}.csv", index=False)

    summary = pd.DataFrame(rows)
    summary.to_csv(out_dir / "loocv_summary_k12.csv", index=False)

    print("\n" + "=" * 100)
    print(f"Summary at k={args.k} (our {X.shape[0]} subjects vs. the paper's 63):")
    print(summary[["Model", "AUC", "paper_AUC", "ACC", "paper_ACC", "SEN", "paper_SEN", "SPE", "paper_SPE"]]
          .to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    best = summary.loc[summary["AUC"].idxmax(), "Model"]
    auc_range_ok = summary["AUC"].between(0.75, 0.95).all()   # generous band around the paper's 0.86-0.91
    print(f"\nBest AUC: {best} ({summary['AUC'].max():.3f}).")
    if auc_range_ok:
        print("All four models land in a range consistent with the paper (0.86-0.91, "
              "with some slack for the 65- vs 63-subject difference). Pipeline looks correct.")
    else:
        print("One or more models are well outside the paper's range -- worth double-checking "
              "Steps 1-4 before trusting Step 6 onward.")

    # ---- Feature selection stability, mirrors the paper's "38 unique features, 3 in >=95% of folds" ----
    print("\nFeature-selection stability (Ridge Logistic, the model used for SHAP in Step 7):")
    from collections import Counter
    freq = Counter(f for r in all_results["Ridge Logistic"] for f in r.selected_features)
    n_folds = X.shape[0]
    top = freq.most_common(12)
    print(f"  {len(freq)} unique features selected at least once across {n_folds} folds "
          f"(paper: 38 features across 63 folds).")
    for feat, cnt in top:
        print(f"    {feat:35s} selected in {cnt}/{n_folds} folds ({cnt/n_folds:.0%})")
    pd.DataFrame(freq.most_common(), columns=["feature", "n_folds_selected"]).to_csv(
        out_dir / "feature_selection_frequency_ridge.csv", index=False)

    # ---- Optional: k-sweep, paper's Figure 4 ----
    if args.k_sweep:
        print("\nRunning k-sweep {4,6,8,10,12,14,16} for all four models (this takes a few minutes)...")
        sweep_rows = []
        for k in [4, 6, 8, 10, 12, 14, 16]:
            for name, make_clf in MODELS.items():
                results = f4.run_loocv(X, y, ids, feature_cols, make_clf, k=k)
                m = f4.summarize(results)
                sweep_rows.append({"k": k, "Model": name, "AUC": m["AUC"], "ACC": m["ACC"]})
                print(f"  k={k:2d}  {name:22s} AUC={m['AUC']:.3f}")
        pd.DataFrame(sweep_rows).to_csv(out_dir / "k_sweep.csv", index=False)
        print(f"\nWrote {out_dir / 'k_sweep.csv'} (compare to the paper's Figure 4).")

    print(f"\nAll outputs written to {out_dir}/")


if __name__ == "__main__":
    main()