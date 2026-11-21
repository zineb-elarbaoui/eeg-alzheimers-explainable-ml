import argparse
import importlib.util
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact, mannwhitneyu, rankdata, spearmanr
from sklearn.metrics import roc_auc_score


def load_module(name, filename):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).parent / filename)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


f4 = load_module("loocv4", "04_loocv_framework.py")
f5 = load_module("loocv5", "05_run_loocv.py")      # MODELS: the paper's four model definitions

K_RANGE = list(range(4, 17))
K_MAX = 16
K_REF = 12


# ----------------------------------------------------------------------------------------------
# fold machinery (fitted on the training fold only)
# ----------------------------------------------------------------------------------------------
def build_folds(X, y01):
    """Per fold: imputed + zero-variance-filtered training matrix, held-out row, labels, kept column ids."""
    n, folds = X.shape[0], []
    for i in range(n):
        tr = np.arange(n) != i
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            med = np.nanmedian(X[tr], axis=0)
        Xtr = np.where(np.isnan(X[tr]), med, X[tr])
        xte = np.where(np.isnan(X[i]), med, X[i])
        keep = Xtr.var(axis=0) > 0
        folds.append({"Xtr": Xtr[:, keep], "xte": xte[keep], "ytr": y01[tr], "kept": np.where(keep)[0], "tr": tr})
    return folds


def select_all(folds, mode, seed=None, k=K_MAX):
    out = []
    for i, f in enumerate(folds):
        rng = np.random.default_rng([seed, i]) if mode == "no_ranking" else None
        out.append(f4.select_features(f["Xtr"], f["ytr"], k=k, mode=mode, rng=rng))
    return out


def zscore(Atr, arow):
    mu, sd = Atr.mean(axis=0), Atr.std(axis=0)
    sd = np.where(sd == 0, 1.0, sd)
    return (Atr - mu) / sd, (arow - mu) / sd


def loocv_scores(folds, make_clf, sels=None, k=K_REF, demo=None, solver_seed=None):
    """Out-of-fold scores.  sels=None -> demographic-only model (needs demo).  demo is appended to the
    selected EEG features when both are given (forced in, standardised on the training fold)."""
    score = np.zeros(len(folds))
    for i, f in enumerate(folds):
        if sels is None:
            Atr, arow = demo[f["tr"]], demo[i]
        else:
            idx = sels[i][:k]
            Atr, arow = f["Xtr"][:, idx], f["xte"][idx]
            if demo is not None:
                Atr, arow = np.hstack([Atr, demo[f["tr"]]]), np.concatenate([arow, demo[i]])
        Ztr, zte = zscore(Atr, arow)
        clf = make_clf(f["ytr"]) if f4._wants_y_train(make_clf) else make_clf()
        if solver_seed is not None and "random_state" in clf.get_params():
            clf.set_params(random_state=solver_seed)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clf.fit(Ztr, f["ytr"])
        score[i] = float(clf.predict_proba(zte[None])[0, 1]) if hasattr(clf, "predict_proba") \
            else float(clf.decision_function(zte[None])[0])
    return score


class Bootstrap:
    """Stratified subject bootstrap on out-of-fold scores; one shared index matrix => paired comparisons."""

    def __init__(self, y01, B=2000, seed=0):
        rng = np.random.default_rng(seed)
        pos, neg = np.where(y01 == 1)[0], np.where(y01 == 0)[0]
        self.n1, self.n0 = len(pos), len(neg)
        self.idx = np.concatenate([rng.choice(pos, (B, self.n1)), rng.choice(neg, (B, self.n0))], axis=1)

    def aucs(self, score):
        r = rankdata(np.asarray(score)[self.idx], axis=1)
        return (r[:, :self.n1].sum(axis=1) - self.n1 * (self.n1 + 1) / 2) / (self.n1 * self.n0)


def verdict(lo, hi):
    return "better" if lo > 0 else ("worse" if hi < 0 else "within noise")


# ----------------------------------------------------------------------------------------------
# part 1: demographics
# ----------------------------------------------------------------------------------------------
def part_demographics(X, y01, ids, cols, folds, sels_full, models, boot, args, out):
    print("=" * 100)
    print("DEMOGRAPHICS: is age or sex doing part of the work?")
    print("=" * 100)
    man = pd.read_csv(args.manifest)
    man.columns = [c.lower() for c in man.columns]
    man = man.set_index("subject_id").loc[ids]
    age = pd.to_numeric(man["age"], errors="coerce").to_numpy(dtype=float)
    sex_raw = man["sex"].astype(str)
    sex = sex_raw.str.upper().str.startswith("M").to_numpy(dtype=float)
    print(f"sex values found in the manifest: {sorted(sex_raw.unique())}  (coded 1 = starts with 'M', else 0)")
    if np.isnan(age).any():
        raise SystemExit("age has missing values in the manifest; fill or drop them before running this part")
    D = np.column_stack([age, sex])
    mmse = pd.to_numeric(man["mmse"], errors="coerce").to_numpy(dtype=float) if "mmse" in man.columns else None

    # ---- description ----
    print("\n--- Cohort description ---")
    rows = []
    for lab, g in (("AD", y01 == 1), ("HC", y01 == 0)):
        r = {"group": lab, "n": int(g.sum()), "age_mean": age[g].mean(), "age_sd": age[g].std(ddof=1),
             "pct_male": 100 * sex[g].mean()}
        if mmse is not None:
            r.update(mmse_mean=np.nanmean(mmse[g]), mmse_sd=np.nanstd(mmse[g], ddof=1))
        rows.append(r)
    desc = pd.DataFrame(rows)
    print(desc.to_string(index=False, float_format=lambda x: f"{x:.1f}"))
    p_age = mannwhitneyu(age[y01 == 1], age[y01 == 0]).pvalue
    tab = [[int(sex[y01 == 1].sum()), int((1 - sex[y01 == 1]).sum())],
           [int(sex[y01 == 0].sum()), int((1 - sex[y01 == 0]).sum())]]
    p_sex = fisher_exact(tab)[1]
    print(f"Age difference AD vs HC: Mann-Whitney p = {p_age:.3f}.  Sex distribution: Fisher exact p = {p_sex:.3f}.")
    print(f"AUC of age alone (1 = older -> AD): {roc_auc_score(y01, age):.3f}   "
          f"AUC of sex alone: {roc_auc_score(y01, sex):.3f}")
    desc.to_csv(out / "demographics_description.csv", index=False)

    # ---- models: demo-only / EEG-only / EEG + demo ----
    print("\n--- Out-of-fold AUC (k=12 EEG features; demographics forced in for EEG+demo) ---")
    rows, eeg_scores = [], {}
    for m, make in models.items():
        s_demo = loocv_scores(folds, make, sels=None, demo=D)
        s_eeg = loocv_scores(folds, make, sels=sels_full, k=K_REF)
        s_both = loocv_scores(folds, make, sels=sels_full, k=K_REF, demo=D)
        eeg_scores[m] = s_eeg
        b_eeg, b_demo, b_both = boot.aucs(s_eeg), boot.aucs(s_demo), boot.aucs(s_both)
        for name, s, b in (("demographics only", s_demo, b_demo), ("EEG only (paper)", s_eeg, b_eeg),
                           ("EEG + age + sex", s_both, b_both)):
            d = b - b_eeg
            lo, hi = np.percentile(d, [2.5, 97.5])
            alo, ahi = np.percentile(b, [2.5, 97.5])
            rows.append({"model": m, "input": name, "AUC": roc_auc_score(y01, s), "AUC_lo": alo, "AUC_hi": ahi,
                         "delta_vs_EEG_only": roc_auc_score(y01, s) - roc_auc_score(y01, s_eeg),
                         "delta_lo": lo, "delta_hi": hi,
                         "verdict": "reference" if name.startswith("EEG only") else verdict(lo, hi)})
    res = pd.DataFrame(rows)
    res.to_csv(out / "demographics_models.csv", index=False)
    print(res.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    # ---- offered as candidate features: does the selection ever pick them? ----
    print("\n--- Age and sex offered as candidate features to the paper's selection (k=12) ---")
    X_aug = np.hstack([X, D])
    folds_aug = build_folds(X_aug, y01)
    sels_aug = select_all(folds_aug, "full", k=K_REF)
    p = X.shape[1]
    picked = np.zeros(X_aug.shape[1])
    for i, f in enumerate(folds_aug):
        picked[f["kept"][sels_aug[i][:K_REF]]] += 1
    print(f"age selected in {int(picked[p])}/{len(folds)} folds, sex selected in {int(picked[p + 1])}/{len(folds)} folds")
    pd.DataFrame({"feature": ["age", "sex"], "folds_selected": [int(picked[p]), int(picked[p + 1])],
                  "n_folds": len(folds)}).to_csv(out / "demographics_selection.csv", index=False)

    # ---- does the EEG score track age within each group? ----
    print("\n--- Does the out-of-fold EEG score track age within diagnostic groups? (Spearman rho, p) ---")
    crow = []
    for m, s in eeg_scores.items():
        r = {"model": m}
        for lab, g in (("AD", y01 == 1), ("HC", y01 == 0)):
            rho, pv = spearmanr(s[g], age[g])
            r[f"rho_age_{lab}"], r[f"p_{lab}"] = rho, pv
        crow.append(r)
    cdf = pd.DataFrame(crow)
    cdf.to_csv(out / "demographics_score_vs_age.csv", index=False)
    print(cdf.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    print("\nHow to read this:")
    print("  * If the demographic-only AUC is near 0.5 and EEG+demo is within noise of EEG-only, age and sex are not")
    print("    driving the result. A demographic AUC well above 0.5 would mean the cohort is confounded.")
    print("  * If age/sex are never selected, the paper's pipeline does not use them even when offered.")
    print("  * A within-group correlation of the EEG score with age would suggest the score partly encodes ageing.")
    print("    With ~30 subjects per group, only large correlations can be detected.")
    print("  * This is a check, not an adjustment: it cannot rule out confounds that are not in the manifest.\n")


# ----------------------------------------------------------------------------------------------
# part 2: k sweep
# ----------------------------------------------------------------------------------------------
def part_k(y01, folds, models, boot, args, out):
    print("=" * 100)
    print("K SWEEP: number of selected features, k = 4..16")
    print("=" * 100)
    variants = {"full": [None], "no_pruning": [None], "no_ranking": [2026 + s for s in range(args.seeds)]}
    t0 = time.time()
    sels = {v: [select_all(folds, v, seed=sd) for sd in seeds] for v, seeds in variants.items()}
    print(f"selections computed for {', '.join(variants)} ({time.time() - t0:.0f}s)\n")

    rows, bvecs = [], {}
    for m, make in models.items():
        for v, seeds in variants.items():
            for k in K_RANGE:
                scs = [loocv_scores(folds, make, sels=s, k=k) for s in sels[v]]
                aucs = np.array([roc_auc_score(y01, s) for s in scs])
                b = np.mean([boot.aucs(s) for s in scs], axis=0)
                bvecs[(m, v, k)] = b
                lo, hi = np.percentile(b, [2.5, 97.5])
                rows.append({"model": m, "variant": v, "k": k, "AUC": aucs.mean(),
                             "AUC_seed_sd": aucs.std(ddof=1) if len(aucs) > 1 else np.nan,
                             "AUC_lo": lo, "AUC_hi": hi})
        print(f"  {m}: done ({time.time() - t0:.0f}s)")
    df = pd.DataFrame(rows)
    # paired comparison with k = 12 (same variant, same model)
    dl, dh, vd, dd = [], [], [], []
    for r in df.itertuples():
        d = bvecs[(r.model, r.variant, r.k)] - bvecs[(r.model, r.variant, K_REF)]
        lo, hi = np.percentile(d, [2.5, 97.5])
        ref = df[(df.model == r.model) & (df.variant == r.variant) & (df.k == K_REF)].AUC.iloc[0]
        dd.append(r.AUC - ref); dl.append(lo); dh.append(hi)
        vd.append("reference" if r.k == K_REF else verdict(lo, hi))
    df["delta_vs_k12"], df["delta_lo"], df["delta_hi"], df["verdict"] = dd, dl, dh, vd
    df.to_csv(out / "k_sweep.csv", index=False)

    # optional reference: no selection at all (Step 6)
    none_ref = {}
    s6 = Path(args.step6_dir) / "ablation_summary.csv"
    if s6.exists():
        d6 = pd.read_csv(s6)
        none_ref = d6[d6.variant == "none"].set_index("Model")["AUC"].to_dict()

    pd.set_option("display.width", 220)
    for v in variants:
        print(f"\n--- AUC vs k, variant `{v}` ---")
        pv = df[df.variant == v].pivot(index="k", columns="model", values="AUC")
        print(pv.to_string(float_format=lambda x: f"{x:.3f}"))
    print("\n--- Summary per model / variant ---")
    sm = []
    for (m, v), g in df.groupby(["model", "variant"], sort=False):
        best = g.loc[g.AUC.idxmax()]
        sm.append({"model": m, "variant": v, "AUC_k12": g[g.k == K_REF].AUC.iloc[0], "min": g.AUC.min(),
                   "max": g.AUC.max(), "range": g.AUC.max() - g.AUC.min(), "best_k": int(best.k),
                   "k_worse_than_12": int((g.verdict == "worse").sum()),
                   "k_better_than_12": int((g.verdict == "better").sum())})
    sm = pd.DataFrame(sm)
    sm.to_csv(out / "k_sweep_summary.csv", index=False)
    print(sm.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    half = ((df[(df.variant == "full") & (df.k == K_REF)].AUC_hi - df[(df.variant == "full") & (df.k == K_REF)].AUC_lo) / 2).mean()
    print(f"\nResolution: a single AUC here has a 95% bootstrap half-width of about +/-{half:.3f}.")
    print("The best k is chosen from the same LOOCV profile it is evaluated on: do not present it as an improved result.\n")

    if args.plots:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib not available; skipping plot")
            return
        fig, axes = plt.subplots(1, len(models), figsize=(4 * len(models), 3.6), sharey=True)
        for ax, m in zip(np.atleast_1d(axes), models):
            for v, c in (("full", "C0"), ("no_pruning", "C1"), ("no_ranking", "C2")):
                g = df[(df.model == m) & (df.variant == v)]
                ax.plot(g.k, g.AUC, marker="o", ms=3, color=c, label=v)
                if v == "full":
                    ax.fill_between(g.k, g.AUC_lo, g.AUC_hi, color=c, alpha=0.12)
            if m in none_ref:
                ax.axhline(none_ref[m], color="k", ls=":", lw=1, label="no selection")
            ax.axvline(K_REF, color="grey", lw=0.5)
            ax.set_title(m, fontsize=9); ax.set_xlabel("k"); ax.set_ylim(0.65, 0.95)
        np.atleast_1d(axes)[0].set_ylabel("LOOCV AUC")
        np.atleast_1d(axes)[0].legend(fontsize=6, loc="lower right")
        fig.tight_layout(); fig.savefig(out / "k_sweep.png", dpi=200); plt.close(fig)


# ----------------------------------------------------------------------------------------------
# part 3: solver seeds
# ----------------------------------------------------------------------------------------------
def part_seeds(y01, folds, sels_full, models, args, out):
    print("=" * 100)
    print("SOLVER SEEDS (k=12, paper's pipeline)")
    print("=" * 100)
    rows = []
    for m, make in models.items():
        probe = make(np.zeros(2, dtype=int)) if f4._wants_y_train(make) else make()
        if "random_state" not in probe.get_params():
            rows.append({"model": m, "n_seeds": 1, "AUC_mean": np.nan, "AUC_sd": 0.0, "AUC_min": np.nan,
                         "AUC_max": np.nan, "note": "no random_state: deterministic"})
            continue
        aucs = [roc_auc_score(y01, loocv_scores(folds, make, sels=sels_full, k=K_REF, solver_seed=2026 + s))
                for s in range(args.solver_seeds)]
        rows.append({"model": m, "n_seeds": len(aucs), "AUC_mean": np.mean(aucs), "AUC_sd": np.std(aucs, ddof=1),
                     "AUC_min": np.min(aucs), "AUC_max": np.max(aucs), "note": ""})
    df = pd.DataFrame(rows)
    df.to_csv(out / "solver_seeds.csv", index=False)
    print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\nA solver-seed SD far below the bootstrap half-width (about 0.08) means seeds are not a source of concern.\n")


# ----------------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="outputs/features.csv")
    ap.add_argument("--manifest", default="outputs/cohort_manifest.csv")
    ap.add_argument("--part", choices=["all", "demographics", "k", "seeds"], default="all")
    ap.add_argument("--seeds", type=int, default=5, help="random orders for no_ranking in the k sweep")
    ap.add_argument("--solver_seeds", type=int, default=10)
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--plots", action="store_true")
    ap.add_argument("--step5_dir", default="outputs/step5")
    ap.add_argument("--step6_dir", default="outputs/step6_ablation")
    ap.add_argument("--out_dir", default="outputs/step9_sensitivity")
    args = ap.parse_args()

    df = pd.read_csv(args.features)
    cols = [c for c in df.columns if c not in {"subject_id", "label", "n_epochs_used"}]
    X = df[cols].to_numpy(dtype=float)
    y01 = (df["label"].to_numpy() == "AD").astype(int)
    ids = df["subject_id"].tolist()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    models = dict(f5.MODELS)
    print(f"Loaded {args.features}: {X.shape[0]} subjects x {X.shape[1]} features ({y01.sum()} AD, {(1 - y01).sum()} HC)\n")

    t0 = time.time()
    folds = build_folds(X, y01)
    sels_full = select_all(folds, "full")
    print(f"Paper's selection computed for all folds ({time.time() - t0:.0f}s)")

    # integrity check: EEG-only k=12 must reproduce Step 5
    s5 = Path(args.step5_dir) / "loocv_summary_k12.csv"
    if s5.exists():
        ref = pd.read_csv(s5).set_index("Model")["AUC"]
        diffs = {m: abs(roc_auc_score(y01, loocv_scores(folds, make, sels=sels_full, k=K_REF)) - ref[m])
                 for m, make in models.items() if m in ref.index}
        ok = all(d < 1e-6 for d in diffs.values())
        print("Integrity check vs Step 5:", "OK - same pipeline reproduced." if ok else
              f"WARNING - differs from Step 5 ({ {m: round(d, 4) for m, d in diffs.items()} })")
    print()

    boot = Bootstrap(y01, B=args.boot)
    if args.part in ("all", "demographics"):
        part_demographics(X, y01, ids, cols, folds, sels_full, models, boot, args, out)
    if args.part in ("all", "k"):
        part_k(y01, folds, models, boot, args, out)
    if args.part in ("all", "seeds"):
        part_seeds(y01, folds, sels_full, models, args, out)
    print(f"All outputs written to {out}/")


if __name__ == "__main__":
    main()