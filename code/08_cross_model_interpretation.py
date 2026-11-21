

import argparse
import importlib.util
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score


def load_module(name, filename):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).parent / filename)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


f4 = load_module("loocv4", "04_loocv_framework.py")
f5 = load_module("loocv5", "05_run_loocv.py")            # MODELS: the paper's four model definitions
st7 = load_module("stab7", "07_feature_stability.py")    # parse_feature / build_meta (family, band, region)

UNI = "Univariate AUC"
TOP_OVERLAP = 12
TOP_SIGN = 20


# ----------------------------------------------------------------------------------------------
# fold preparation (identical to run_loocv, fitted on the training fold only)
# ----------------------------------------------------------------------------------------------
def prep_fold(X, y01, i, k, use_sel):
    n = X.shape[0]
    tr = np.arange(n) != i
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        med = np.nanmedian(X[tr], axis=0)
    Xtr = np.where(np.isnan(X[tr]), med, X[tr])
    Xte = np.where(np.isnan(X[[i]]), med, X[[i]])
    keep = Xtr.var(axis=0) > 0
    Xtr, Xte, kept_idx = Xtr[:, keep], Xte[:, keep], np.where(keep)[0]
    ytr = y01[tr]
    sel = f4.select_features(Xtr, ytr, k=k, mode="full") if use_sel else np.arange(Xtr.shape[1])
    Xtr, Xte, orig_idx = Xtr[:, sel], Xte[:, sel], kept_idx[sel]
    mu, sd = Xtr.mean(axis=0), Xtr.std(axis=0)
    sd = np.where(sd == 0, 1.0, sd)
    return (Xtr - mu) / sd, ((Xte - mu) / sd)[0], ytr, orig_idx


def fit_predict(make_clf, Ztr, ytr, zte):
    clf = make_clf(ytr) if f4._wants_y_train(make_clf) else make_clf()
    clf.fit(Ztr, ytr)
    score = float(clf.predict_proba(zte[None])[0, 1]) if hasattr(clf, "predict_proba") \
        else float(clf.decision_function(zte[None])[0])
    return clf, score, int(clf.predict(zte[None])[0])


# ----------------------------------------------------------------------------------------------
# agreement measures
# ----------------------------------------------------------------------------------------------
def top_set(v, n):
    idx = np.argsort(-np.asarray(v))[:n]
    return {int(j) for j in idx if v[j] > 0}


def agreement(imp, coef, top=TOP_OVERLAP, sign_top=TOP_SIGN):
    """imp / coef: dict name -> vector over features. Returns Spearman, top-overlap, sign-agreement, n_shared."""
    names = list(imp)
    mk = lambda: pd.DataFrame(np.nan, index=names, columns=names)
    sp, ov, sg, ns = mk(), mk(), mk(), mk()
    for a in names:
        for b in names:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                sp.loc[a, b] = spearmanr(imp[a], imp[b])[0]
            ov.loc[a, b] = len(top_set(imp[a], top) & top_set(imp[b], top))
            shared = sorted(top_set(imp[a], sign_top) & top_set(imp[b], sign_top))
            ns.loc[a, b] = len(shared)
            if shared:
                sg.loc[a, b] = float(np.mean([np.sign(coef[a][j]) == np.sign(coef[b][j]) for j in shared]))
    return sp, ov, sg, ns


def offdiag_mean(M, names):
    vals = [M.loc[a, b] for i, a in enumerate(names) for b in names[i + 1:]]
    return float(np.nanmean(vals)) if vals else np.nan


def group_agreement(share, meta, level, names, top=5):
    gs = {m: pd.Series(share[m]).groupby(meta[level].to_numpy()).sum() for m in names}
    groups = gs[names[0]].index
    sp = pd.DataFrame(np.nan, index=names, columns=names)
    ov = pd.DataFrame(np.nan, index=names, columns=names)
    for a in names:
        for b in names:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                sp.loc[a, b] = spearmanr(gs[a].loc[groups], gs[b].loc[groups])[0]
            ov.loc[a, b] = len(set(gs[a].nlargest(top).index) & set(gs[b].nlargest(top).index))
    return sp, ov, gs


# ----------------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="outputs/features.csv")
    ap.add_argument("--k", type=int, default=12)
    ap.add_argument("--skip_none", action="store_true", help="run the full pipeline only")
    ap.add_argument("--plots", action="store_true")
    ap.add_argument("--verify_shap", action="store_true", help="check fold 0 against shap.LinearExplainer")
    ap.add_argument("--step5_dir", default="outputs/step5")
    ap.add_argument("--step6_dir", default="outputs/step6_ablation")
    ap.add_argument("--out_dir", default="outputs/step8_interpretation")
    args = ap.parse_args()

    df = pd.read_csv(args.features)
    cols = [c for c in df.columns if c not in {"subject_id", "label", "n_epochs_used"}]
    X = df[cols].to_numpy(dtype=float)
    y01 = (df["label"].to_numpy() == "AD").astype(int)
    ids = df["subject_id"].tolist()
    n, p = X.shape
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    meta = st7.build_meta(cols)
    models = dict(f5.MODELS)
    variants = ["full"] + ([] if args.skip_none else ["none"])
    print(f"Loaded {args.features}: {n} subjects x {p} features ({y01.sum()} AD, {(1 - y01).sum()} HC)")
    print(f"Models: {', '.join(models)} | variants: {', '.join(variants)} | k = {args.k}\n")

    # model-free reference (descriptive, all subjects)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        Xi_all = np.where(np.isnan(X), np.nanmedian(X, axis=0), X)
    uni_auc = np.array([0.5 if np.all(Xi_all[:, j] == Xi_all[0, j]) else roc_auc_score(y01, Xi_all[:, j])
                        for j in range(p)])
    uni_signed = uni_auc - 0.5                     # > 0: higher in AD
    uni_imp = np.abs(uni_signed)

    # ---- one LOOCV pass: out-of-fold scores + out-of-fold Linear SHAP ----
    R = {v: {m: {"score": np.zeros(n), "pred": np.zeros(n, int), "phi": np.zeros((n, p)),
                 "coef": np.zeros((n, p))} for m in models} for v in variants}
    selmask = {v: np.zeros((n, p), dtype=bool) for v in variants}
    t0 = time.time()
    for i in range(n):
        for v in variants:
            Ztr, zte, ytr, oidx = prep_fold(X, y01, i, args.k, v == "full")
            selmask[v][i, oidx] = True
            for m, make in models.items():
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    clf, score, pred = fit_predict(make, Ztr, ytr, zte)
                w = clf.coef_.ravel()
                r = R[v][m]
                r["score"][i], r["pred"][i] = score, pred
                r["coef"][i, oidx] = w
                r["phi"][i, oidx] = w * zte                  # Linear SHAP, standardised inputs
                if args.verify_shap and i == 0 and m == next(iter(models)) and v == "full":
                    try:
                        import shap
                        sv = shap.LinearExplainer(clf, Ztr).shap_values(zte[None])
                        sv = np.asarray(sv[1] if isinstance(sv, list) else sv).ravel()
                        print(f"shap check (fold 0, {m}): max |difference| vs coef*z = "
                              f"{np.max(np.abs(sv - w * zte)):.2e}\n")
                    except Exception as e:                     # shap missing or API difference
                        print(f"shap check skipped: {e}\n")
        if (i + 1) % 13 == 0 or i == n - 1:
            print(f"  LOOCV fold {i + 1}/{n}  [{time.time() - t0:.0f}s]")
    print()

    # ---- integrity check against Step 5 / Step 6 ----
    refs = {}
    s5 = Path(args.step5_dir) / "loocv_summary_k12.csv"
    if s5.exists() and args.k == 12:
        refs["full"] = pd.read_csv(s5).set_index("Model")["AUC"]
    s6 = Path(args.step6_dir) / "ablation_summary.csv"
    if s6.exists():
        d = pd.read_csv(s6)
        d = d[d.variant == "none"]
        if len(d):
            refs["none"] = d.set_index("Model")["AUC"]
    print("=== Out-of-fold AUC (integrity check against earlier steps) ===")
    aucs = pd.DataFrame({v: {m: roc_auc_score(y01, R[v][m]["score"]) for m in models} for v in variants})
    for v in variants:
        if v in refs:
            diff = max(abs(aucs.loc[m, v] - refs[v][m]) for m in models if m in refs[v].index)
            print(f"  {v}: max |AUC difference| vs earlier step = {diff:.2e} "
                  + ("(OK, same pipeline)" if diff < 1e-6 else "(WARNING: does not reproduce - was features.csv regenerated?)"))
    print(aucs.to_string(float_format=lambda x: f"{x:.3f}"))
    print()

    pd.set_option("display.width", 220)
    fm = lambda x: f"{x:.2f}"
    summary_rows = []

    for v in variants:
        print("=" * 100)
        print(f"VARIANT: {v}" + ("  (paper's pipeline, same 12 features per fold for every model)" if v == "full"
                                   else "  (no feature selection: each model's regularisation decides)"))
        print("=" * 100)
        imp, share, mcoef, consist = {}, {}, {}, {}
        for m in models:
            r = R[v][m]
            imp[m] = np.abs(r["phi"]).mean(axis=0)
            share[m] = imp[m] / imp[m].sum() if imp[m].sum() > 0 else imp[m]
            mcoef[m] = r["coef"].mean(axis=0)
            pos = ((r["coef"] > 0) & selmask[v]).sum(axis=0)
            neg = ((r["coef"] < 0) & selmask[v]).sum(axis=0)
            consist[m] = np.where(pos + neg > 0, np.maximum(pos, neg) / np.maximum(pos + neg, 1), np.nan)
        imp_ref, coef_ref = dict(imp), dict(mcoef)
        imp_ref[UNI], coef_ref[UNI] = uni_imp, uni_signed
        share[UNI] = uni_imp / uni_imp.sum()
        names_m = list(models)
        names_all = names_m + [UNI]

        # per-feature table
        ft = meta[["feature", "family", "band", "region"]].copy()
        ft["sel_freq"] = selmask[v].mean(axis=0)
        ft["uni_auc"], ft["uni_dir"] = uni_auc, np.where(uni_signed > 0, "up_in_AD", "down_in_AD")
        for m in models:
            tag = m.replace(" ", "_")
            ft[f"imp_{tag}"], ft[f"share_{tag}"] = imp[m], share[m]
            ft[f"coef_{tag}"], ft[f"signconsist_{tag}"] = mcoef[m], consist[m]
        ft["mean_share"] = np.mean([share[m] for m in models], axis=0)
        ft = ft.sort_values("mean_share", ascending=False)
        ft.to_csv(out / f"importance_{v}.csv", index=False)
        np.savez_compressed(out / f"phi_{v}.npz", subject_ids=np.array(ids), feature_names=np.array(cols),
                            **{m.replace(" ", "_"): R[v][m]["phi"] for m in models})

        # top features per model
        print("\n--- Top 8 features per model (share of total importance; + = higher in AD) ---")
        for m in models:
            top = np.argsort(-imp[m])[:8]
            line = ", ".join(f"{cols[j]} {share[m][j]:.2f}{'+' if mcoef[m][j] > 0 else '-'}" for j in top)
            print(f"  {m:<20s} {line}")
        print(f"  {UNI:<20s} " + ", ".join(f"{cols[j]} {uni_imp[j]:.2f}{'+' if uni_signed[j] > 0 else '-'}"
                                          for j in np.argsort(-uni_imp)[:8]) + "   (|AUC-0.5|)")

        # feature-level agreement
        sp, ov, sg, ns = agreement(imp_ref, coef_ref)
        print(f"\n--- Feature level: Spearman rank correlation of importance (all {p} features) ---")
        print(sp.to_string(float_format=fm))
        print(f"\n--- Feature level: overlap of top-{TOP_OVERLAP} features (count) ---")
        print(ov.to_string(float_format=lambda x: f"{x:.0f}"))
        print(f"\n--- Sign agreement of coefficients on features in both top-{TOP_SIGN} lists "
              f"(n shared in brackets, see CSV) ---")
        print(sg.to_string(float_format=fm, na_rep="-"))
        sp.to_csv(out / f"agreement_spearman_{v}.csv")
        ov.to_csv(out / f"agreement_top{TOP_OVERLAP}_{v}.csv")
        sg.to_csv(out / f"agreement_sign_{v}.csv")
        ns.to_csv(out / f"agreement_nshared_{v}.csv")

        # direction vs univariate direction (descriptive)
        print(f"\n--- Does each model's coefficient sign match the univariate direction? (top-{TOP_OVERLAP} features) ---")
        for m in models:
            tops = sorted(top_set(imp[m], TOP_OVERLAP))
            same = np.mean([np.sign(mcoef[m][j]) == np.sign(uni_signed[j]) for j in tops]) if tops else np.nan
            print(f"  {m:<20s} {same:.2f}  ({len(tops)} features)")

        # group-level agreement
        row = {"variant": v, "feature_spearman_models": offdiag_mean(sp, names_m),
               "feature_spearman_vs_univariate": float(np.nanmean([sp.loc[m, UNI] for m in names_m])),
               f"feature_top{TOP_OVERLAP}_overlap_models": offdiag_mean(ov, names_m),
               "sign_agreement_models": offdiag_mean(sg, names_m)}
        for level in ("family_region", "family", "region"):
            gsp, gov, gs = group_agreement(share, meta, level, names_all)
            gsp.to_csv(out / f"group_agreement_spearman_{level}_{v}.csv")
            pd.DataFrame(gs).to_csv(out / f"group_share_{level}_{v}.csv")
            row[f"{level}_spearman_models"] = offdiag_mean(gsp, names_m)
            row[f"{level}_spearman_vs_univariate"] = float(np.nanmean([gsp.loc[m, UNI] for m in names_m]))
            if level == "family_region":
                print("\n--- Family|region level: Spearman of group shares between models ---")
                print(gsp.to_string(float_format=fm))
                gdf = pd.DataFrame(gs)
                gdf["mean_models"] = gdf[names_m].mean(axis=1)
                print("\n--- Largest family|region groups (mean share across the four models) ---")
                print(gdf.sort_values("mean_models", ascending=False).head(8)[names_m].to_string(float_format=fm))
        summary_rows.append(row)

        if args.plots:
            try:
                import matplotlib
                matplotlib.use("Agg")
                import matplotlib.pyplot as plt
            except ImportError:
                print("matplotlib not available; skipping plots")
                args.plots = False
            else:
                fig, ax = plt.subplots(figsize=(5, 4.2))
                im = ax.imshow(sp.to_numpy(dtype=float), vmin=-0.2, vmax=1, cmap="viridis")
                ax.set_xticks(range(len(names_all))); ax.set_xticklabels(names_all, rotation=45, ha="right", fontsize=7)
                ax.set_yticks(range(len(names_all))); ax.set_yticklabels(names_all, fontsize=7)
                for a in range(len(names_all)):
                    for b in range(len(names_all)):
                        ax.text(b, a, f"{sp.iloc[a, b]:.2f}", ha="center", va="center", fontsize=7, color="w")
                ax.set_title(f"Spearman of importances ({v})", fontsize=9)
                fig.colorbar(im, fraction=0.046); fig.tight_layout()
                fig.savefig(out / f"agreement_{v}.png", dpi=200); plt.close(fig)

                top15 = ft.head(15).iloc[::-1]
                fig, ax = plt.subplots(figsize=(6.5, 5))
                M = top15[[f"share_{m.replace(' ', '_')}" for m in models]].to_numpy()
                im = ax.imshow(M, aspect="auto", cmap="magma_r")
                ax.set_xticks(range(len(models))); ax.set_xticklabels(models, rotation=30, ha="right", fontsize=7)
                ax.set_yticks(range(len(top15))); ax.set_yticklabels(top15["feature"], fontsize=7)
                ax.set_title(f"Share of total importance, top 15 features ({v})", fontsize=9)
                fig.colorbar(im, fraction=0.046); fig.tight_layout()
                fig.savefig(out / f"importance_heatmap_{v}.png", dpi=200); plt.close(fig)
        print()

    summ = pd.DataFrame(summary_rows)
    summ.to_csv(out / "agreement_summary.csv", index=False)
    print("=" * 100)
    print("SUMMARY: mean pairwise agreement between the four models (and with the univariate reference)")
    print("=" * 100)
    print(summ.set_index("variant").T.to_string(float_format=fm))
    print("\nHow to read this:")
    print("  * `full`: all models get the same 12 features per fold, so agreement is partly built in.")
    print("  * `none`: each model's own regularisation decides; agreement here is the real test of convergence.")
    print("  * If feature-level agreement is modest but family|region agreement is high, the models rely on the same")
    print("    signal through different (correlated) features - consistent with the Step 7 substitution result.")
    print("  * Multivariate coefficient signs can differ from univariate directions when features are correlated.")
    print(f"\nAll outputs written to {out}/")


if __name__ == "__main__":
    main()