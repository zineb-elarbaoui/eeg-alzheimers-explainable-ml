

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

MODELS = ["Ridge_Logistic", "ElasticNet_Logistic", "Linear_SVM", "Shrinkage_LDA"]
TOP = 12
TOP_SIGN = 20


def top_idx(v, n):
    idx = np.argsort(-np.asarray(v))[:n]
    return {int(j) for j in idx if v[j] > 0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="outputs/step8_interpretation")
    args = ap.parse_args()
    d = Path(args.dir)

    full = pd.read_csv(d / "importance_full.csv").set_index("feature")
    none = pd.read_csv(d / "importance_none.csv").set_index("feature")
    none = none.loc[full.index]                     # same feature order in both tables
    p = len(full)
    picked = (full["sel_freq"] > 0).to_numpy()      # features the paper's selection ever picked
    print(f"{p} features; the paper's selection picks {int(picked.sum())} of them at least once (k=12, LOOCV)")
    print(f"Chance top-{TOP} overlap = {TOP * TOP / p:.2f} features\n")

    rows = []
    for m in MODELS:
        sf, sn = full[f"share_{m}"].to_numpy(), none[f"share_{m}"].to_numpy()
        cf, cn = full[f"coef_{m}"].to_numpy(), none[f"coef_{m}"].to_numpy()
        tf, tn = top_idx(sf, TOP), top_idx(sn, TOP)
        sh = sorted(top_idx(sf, TOP_SIGN) & top_idx(sn, TOP_SIGN))
        fam = pd.DataFrame({"full": sf, "none": sn, "family": full["family"].to_numpy()}).groupby("family").sum()
        gf = pd.Series(sf, index=full.index).groupby((full["family"] + "|" + full["region"]).to_numpy()).sum()
        gn = pd.Series(sn, index=full.index).groupby((full["family"] + "|" + full["region"]).to_numpy()).sum()
        rows.append({
            "model": m.replace("_", " "),
            "spearman_features": spearmanr(sf, sn)[0],
            f"top{TOP}_overlap": len(tf & tn),
            "none_share_on_picked_features": float(sn[picked].sum()),
            "full_share_complexity_asym": float(fam.loc[[f for f in fam.index if f in ("asymmetry", "perm_entropy", "diff_entropy")], "full"].sum()),
            "none_share_complexity_asym": float(fam.loc[[f for f in fam.index if f in ("asymmetry", "perm_entropy", "diff_entropy")], "none"].sum()),
            "spearman_family_region": spearmanr(gf.loc[gf.index.union(gn.index)].fillna(0),
                                                gn.loc[gf.index.union(gn.index)].fillna(0))[0]
            if len(gf) > 2 else np.nan,
            "sign_agree_top20_both": float(np.mean([np.sign(cf[j]) == np.sign(cn[j]) for j in sh])) if sh else np.nan,
            "n_shared_top20": len(sh),
        })
        fam.to_csv(d / f"family_share_full_vs_none_{m}.csv")

    res = pd.DataFrame(rows)
    res.to_csv(d / "pipeline_comparison.csv", index=False)
    pd.set_option("display.width", 220)
    print(res.to_string(index=False, float_format=lambda x: f"{x:.2f}"))

    print("\nFamily share of importance, mean over the four models:")
    fam_all = pd.concat([pd.read_csv(d / f"family_share_full_vs_none_{m}.csv").set_index("family") for m in MODELS])
    print(fam_all.groupby(level=0).mean().sort_values("full", ascending=False).to_string(float_format=lambda x: f"{x:.2f}"))

    print("\nTop-5 features per pipeline (mean share across the four models):")
    for name, t in (("full", full), ("none", none)):
        ms = t[[f"share_{m}" for m in MODELS]].mean(axis=1).sort_values(ascending=False).head(5)
        print(f"  {name}: " + ", ".join(f"{f} {v:.2f}" for f, v in ms.items()))
    print(f"\nWritten to {d}/pipeline_comparison.csv")


if __name__ == "__main__":
    main()