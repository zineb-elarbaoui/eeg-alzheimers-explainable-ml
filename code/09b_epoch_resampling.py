"""
Step 9b - Epoch-resampling sensitivity  (robustness extension, revised roadmap)
===============================================================================
Question: how much do the results depend on WHICH 60 epochs were randomly drawn per subject?

The paper (and Step 3) average every feature over at most 60 randomly drawn accepted epochs per subject
(seed 2026). This script repeats the whole analysis for independent draws:

  * seeds 2026 ... 2025+N (default N = 30; seed 2026 must reproduce outputs/features.csv exactly)
  * one extra run using EVERY accepted epoch (no cap)

For every draw it recomputes all 373 features (reusing 03_extract_features.py: same sampling code and the same
rng-per-pass convention, so seed 2026 is the Step 3 draw), then runs the paper's leakage-safe LOOCV pipeline
at k = 12 (selection once per fold, shared by the models; code reused from 09_sensitivity_analyses.py).
Default model: Ridge Logistic only, as in the paper. Use --models for others.

Reported: mean / SD / range of the AUC over draws, where the seed-2026 result and the all-epochs result sit,
the paper's reference (Ridge: 0.887 +/- 0.020, range 0.829-0.921, all epochs 0.917), and how consistently the
same features are selected across draws.

This measures a different source of noise from the bootstrap intervals of Steps 6 and 9: the bootstrap asks
"which subjects are in the cohort", this asks "which epochs were drawn for each subject".

Each draw is saved to its own JSON file, so an interrupted run continues where it stopped (rerun the same
command). Expect roughly a minute or more per draw (feature extraction + selection over 65 folds).

Usage (project root):
  python code/09b_epoch_resampling.py                    # 30 draws + all epochs, Ridge Logistic
  python code/09b_epoch_resampling.py --n_draws 5        # quick trial
  python code/09b_epoch_resampling.py --models "Ridge Logistic" "ElasticNet Logistic" --plots
"""

import argparse
import importlib.util
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def load_module(name, filename):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).parent / filename)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


st3 = load_module("feat3", "03_extract_features.py")
s9 = load_module("sens9", "09_sensitivity_analyses.py")     # build_folds, select_all, loocv_scores, MODELS

K_REF = 12
PAPER_REF = {"mean": 0.887, "sd": 0.020, "lo": 0.829, "hi": 0.921, "all_epochs": 0.917}   # Ridge, Shen et al. 2026


def draw_features(manifest, pre, ch_names, max_epochs, seed):
    """Same loop as 03_extract_features.py: one rng per pass, consumed subject by subject in manifest order."""
    rng = np.random.default_rng(seed)
    rows = []
    for _, r in manifest.iterrows():
        sid = r["subject_id"]
        epochs = np.load(pre / "epochs" / f"{sid}_epochs.npy").astype(np.float64)
        epochs = st3.sample_epochs(epochs, max_epochs, rng)
        f, _ = st3.subject_features(epochs, ch_names)
        rows.append({"subject_id": sid, "label": r["label"], **f})
    return pd.DataFrame(rows)


def run_draw(df, models):
    cols = [c for c in df.columns if c not in ("subject_id", "label")]
    X = df[cols].to_numpy(dtype=float)
    y01 = (df["label"].to_numpy() == "AD").astype(int)
    folds = s9.build_folds(X, y01)
    sels = s9.select_all(folds, "full", k=K_REF)
    aucs = {m: float(roc_auc_score(y01, s9.loocv_scores(folds, make, sels=sels, k=K_REF)))
            for m, make in models.items()}
    counts = {}
    for i, f in enumerate(folds):
        for j in f["kept"][sels[i][:K_REF]]:
            counts[cols[j]] = counts.get(cols[j], 0) + 1
    return aucs, {k: v / len(folds) for k, v in counts.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pre_dir", default="outputs/preprocessed")
    ap.add_argument("--manifest", default="outputs/cohort_manifest.csv")
    ap.add_argument("--features", default="outputs/features.csv", help="Step 3 output, used for the seed-2026 check")
    ap.add_argument("--n_draws", type=int, default=30)
    ap.add_argument("--first_seed", type=int, default=2026)
    ap.add_argument("--max_epochs", type=int, default=60)
    ap.add_argument("--models", nargs="+", default=["Ridge Logistic"])
    ap.add_argument("--skip_all_epochs", action="store_true")
    ap.add_argument("--plots", action="store_true")
    ap.add_argument("--out_dir", default="outputs/step9_epochs")
    args = ap.parse_args()

    pre = Path(args.pre_dir)
    ch_names = (pre / "channel_names.txt").read_text().split()
    manifest = pd.read_csv(args.manifest)
    models = {m: f for m, f in s9.f5.MODELS.items() if m in args.models}
    if not models:
        raise SystemExit(f"None of {args.models} found in MODELS: {list(s9.f5.MODELS)}")
    out = Path(args.out_dir)
    (out / "draws").mkdir(parents=True, exist_ok=True)

    jobs = [(f"seed{s}", s, args.max_epochs) for s in range(args.first_seed, args.first_seed + args.n_draws)]
    if not args.skip_all_epochs:
        jobs.append(("all_epochs", args.first_seed, 0))
    print(f"{len(manifest)} subjects, models: {', '.join(models)}, {len(jobs)} runs "
          f"({args.n_draws} draws of <= {args.max_epochs} epochs" + ("" if args.skip_all_epochs else " + all epochs") + ")\n")

    t_start = time.time()
    for label, seed, max_ep in jobs:
        path = out / "draws" / f"{label}.json"
        if path.exists():
            print(f"  {label}: already done, skipping")
            continue
        t0 = time.time()
        df = draw_features(manifest, pre, ch_names, max_ep, seed)
        if label == f"seed{args.first_seed}" and Path(args.features).exists():
            ref = pd.read_csv(args.features).set_index("subject_id")
            cols = [c for c in df.columns if c not in ("subject_id", "label")]
            same = set(cols) <= set(ref.columns) and set(df.subject_id) <= set(ref.index) and \
                np.allclose(df.set_index("subject_id")[cols].to_numpy(),
                            ref.loc[df.subject_id, cols].to_numpy(), rtol=1e-9, atol=0, equal_nan=True)
            print("  check: seed", seed, "reproduces", args.features, "->",
                  "OK" if same else "WARNING - differs (was features.csv made with other settings or code?)")
        aucs, freq = run_draw(df, models)
        path.write_text(json.dumps({"label": label, "seed": seed, "max_epochs": max_ep, "AUC": aucs, "sel_freq": freq}))
        print(f"  {label}: " + ", ".join(f"{m} AUC={a:.3f}" for m, a in aucs.items())
              + f"  [{time.time() - t0:.0f}s, elapsed {(time.time() - t_start) / 60:.1f} min]")

    # ---- summary (from all saved draws, so a resumed run summarises everything) ----
    res = [json.loads(p.read_text()) for p in sorted((out / "draws").glob("*.json"))]
    draws = [r for r in res if r["label"].startswith("seed")]
    allep = next((r for r in res if r["label"] == "all_epochs"), None)
    rows = []
    for m in models:
        a = np.array([r["AUC"][m] for r in draws if m in r["AUC"]])
        s0 = next((r["AUC"][m] for r in draws if r["seed"] == args.first_seed and m in r["AUC"]), np.nan)
        rows.append({"model": m, "n_draws": len(a), "AUC_mean": a.mean(), "AUC_sd": a.std(ddof=1) if len(a) > 1 else np.nan,
                     "AUC_min": a.min(), "AUC_max": a.max(), "AUC_range": a.max() - a.min(),
                     f"AUC_seed{args.first_seed}": s0,
                     "rank_of_first_seed": int((a < s0).sum()) + 1 if not np.isnan(s0) else np.nan,
                     "AUC_all_epochs": allep["AUC"][m] if allep and m in allep["AUC"] else np.nan})
    sm = pd.DataFrame(rows)
    sm.to_csv(out / "epoch_resampling_summary.csv", index=False)
    long = pd.DataFrame([{"run": r["label"], "model": m, "AUC": a} for r in res for m, a in r["AUC"].items()])
    long.to_csv(out / "epoch_resampling_runs.csv", index=False)

    pd.set_option("display.width", 220)
    print("\n" + "=" * 100)
    print(f"EPOCH RESAMPLING: AUC over {len(draws)} independent draws of <= {args.max_epochs} epochs (k = {K_REF})")
    print("=" * 100)
    print(sm.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    if "Ridge Logistic" in models:
        print(f"\nPaper, Ridge Logistic: mean {PAPER_REF['mean']:.3f} +/- {PAPER_REF['sd']:.3f}, range "
              f"{PAPER_REF['lo']:.3f}-{PAPER_REF['hi']:.3f}, all epochs {PAPER_REF['all_epochs']:.3f}")
    print("Bootstrap half-width over subjects (Step 6/9) for comparison: about +/-0.084")

    # ---- consistency of the selected features across draws ----
    if draws:
        fr = pd.DataFrame([r["sel_freq"] for r in draws]).fillna(0.0)
        summ = pd.DataFrame({"mean_freq": fr.mean(), "min_freq": fr.min(),
                             "draws_ge90": (fr >= 0.90).sum()}).sort_values("mean_freq", ascending=False)
        summ.to_csv(out / "feature_selection_across_draws.csv")
        print(f"\n--- Features most consistently selected across the {len(draws)} draws "
              f"(fraction of LOOCV folds, averaged over draws) ---")
        print(summ.head(12).to_string(float_format=lambda x: f"{x:.2f}"))
        print(f"Features selected in >=90% of folds in at least half the draws: "
              f"{int((summ.draws_ge90 >= len(draws) / 2).sum())}")

    print("\nHow to read this:")
    print("  * The SD over draws is the epoch-sampling noise. If the seed-2026 AUC lies inside the range, the")
    print("    Step 5 number is an ordinary draw, and gaps to the paper's 0.912 are within this noise.")
    print("  * Epoch noise and subject (bootstrap) noise are different: the first is cheap to average away,")
    print("    the second is a property of having only 65 subjects.")
    print(f"\nAll outputs written to {out}/")

    if args.plots and draws:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib not available; skipping plot")
            return
        fig, axes = plt.subplots(1, len(models), figsize=(4.5 * len(models), 3.6), squeeze=False)
        for ax, m in zip(axes[0], models):
            a = np.array([r["AUC"][m] for r in draws if m in r["AUC"]])
            ax.hist(a, bins=10, color="C0", alpha=0.8)
            s0 = next((r["AUC"][m] for r in draws if r["seed"] == args.first_seed), None)
            if s0 is not None:
                ax.axvline(s0, color="k", lw=1.5, label=f"seed {args.first_seed}")
            if allep and m in allep["AUC"]:
                ax.axvline(allep["AUC"][m], color="C3", ls="--", label="all epochs")
            if m == "Ridge Logistic":
                ax.axvspan(PAPER_REF["lo"], PAPER_REF["hi"], color="grey", alpha=0.15, label="paper range")
            ax.set_xlabel("LOOCV AUC"); ax.set_title(m, fontsize=9); ax.legend(fontsize=7)
        fig.tight_layout(); fig.savefig(out / "epoch_resampling.png", dpi=200); plt.close(fig)


if __name__ == "__main__":
    main()