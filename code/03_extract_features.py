

import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.integrate import trapezoid
from scipy.signal import welch

FS = 100.0
BANDS = {"delta": (1.0, 4.0), "theta": (4.0, 8.0), "alpha": (8.0, 13.0), "beta": (13.0, 30.0)}
BAND_NAMES = list(BANDS)
NPERSEG, NOVERLAP = 256, 128   # 2.56 s segments, 50% overlap

# Lateralised regions exactly as written in the paper (note: F7/F8 appear in both frontal and temporal there).
REGIONS = {
    "frontal_left": ["Fp1", "F3", "F7"],   "frontal_right": ["Fp2", "F4", "F8"],
    "temporal_left": ["T3", "T5", "F7"],   "temporal_right": ["T4", "T6", "F8"],
    "central_left": ["C3"],                "central_right": ["C4"],
    "parietal_left": ["P3"],               "parietal_right": ["P4"],
    "occipital_left": ["O1"],              "occipital_right": ["O2"],
}
PAIRS = {"frontal": ("frontal_left", "frontal_right"), "central": ("central_left", "central_right"),
         "temporal": ("temporal_left", "temporal_right"), "parietal": ("parietal_left", "parietal_right"),
         "occipital": ("occipital_left", "occipital_right")}
MIDLINE = ["Fz", "Cz", "Pz"]


def sample_epochs(epochs, max_epochs, rng):
    if max_epochs and epochs.shape[0] > max_epochs:
        idx = np.sort(rng.choice(epochs.shape[0], size=max_epochs, replace=False))
        return epochs[idx]
    return epochs


def psd_epoch_averaged(epochs):
    """(E, C, T) -> freqs, (C, F): Welch PSD of every epoch, averaged over epochs."""
    freqs, psd = welch(epochs, fs=FS, nperseg=NPERSEG, noverlap=NOVERLAP, window="hann", axis=-1)
    return freqs, psd.mean(axis=0)


def integrate(freqs, psd, lo, hi):
    m = (freqs >= lo) & (freqs <= hi)
    return trapezoid(psd[..., m], freqs[m], axis=-1)


def permutation_entropy(epochs, order=3, delay=1):
    """Normalised permutation entropy in [0, 1] for every epoch and channel -> (E, C)."""
    n = epochs.shape[-1] - (order - 1) * delay
    emb = np.stack([epochs[..., i * delay:i * delay + n] for i in range(order)], axis=-1)
    ranks = np.argsort(emb, axis=-1)
    base = order ** np.arange(order - 1, -1, -1)
    code = (ranks * base).sum(axis=-1)
    pattern_codes = [(np.array(p) * base).sum() for p in itertools.permutations(range(order))]
    counts = np.stack([(code == c).sum(axis=-1) for c in pattern_codes], axis=-1)
    p = counts / counts.sum(axis=-1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        h = -np.where(p > 0, p * np.log(p), 0.0).sum(axis=-1)
    return h / np.log(len(pattern_codes))


def differential_entropy(epochs):
    """Gaussian differential entropy 0.5*ln(2*pi*e*var) per epoch and channel -> (E, C)."""
    return 0.5 * np.log(2 * np.pi * np.e * epochs.var(axis=-1))


def asym(l, r):
    return (r - l) / (r + l)


def subject_features(epochs, ch_names, pe_order=3):
    """epochs (E, C, T) in uV -> dict of 373 features -> (dict name -> value, dict name -> family)."""
    idx = {c: i for i, c in enumerate(ch_names)}
    freqs, psd = psd_epoch_averaged(epochs)
    total = integrate(freqs, psd, 1.0, 30.0)                                   # (C,)
    absp = {b: integrate(freqs, psd, *BANDS[b]) for b in BAND_NAMES}           # band -> (C,)
    relp = {b: absp[b] / total for b in BAND_NAMES}
    PE = permutation_entropy(epochs, order=pe_order).mean(axis=0)              # (C,) epoch-averaged
    DE = differential_entropy(epochs).mean(axis=0)

    # spatial units -> value arrays
    def pick(chs):
        return [idx[c] for c in chs]
    units = {c: [idx[c]] for c in ch_names}
    units.update({r: pick(chs) for r, chs in REGIONS.items()})
    units["midline"] = pick(MIDLINE)
    units["global"] = list(range(len(ch_names)))

    feats, fam = {}, {}
    def put(name, value, family):
        feats[name] = float(value)
        fam[name] = family

    unit_abs, unit_rel = {}, {}
    for u, ii in units.items():
        for b in BAND_NAMES:
            unit_abs[(u, b)] = absp[b][ii].mean()
            unit_rel[(u, b)] = relp[b][ii].mean()
            put(f"{u}_abs_{b}", unit_abs[(u, b)], "absolute_power")
            put(f"{u}_rel_{b}", unit_rel[(u, b)], "relative_power")
        put(f"{u}_PE", PE[ii].mean(), "permutation_entropy")
        put(f"{u}_DE", DE[ii].mean(), "differential_entropy")

    for u in list(REGIONS) + ["global"]:                                       # ratios from regional relative powers
        r = {b: unit_rel[(u, b)] for b in BAND_NAMES}
        put(f"{u}_TAR", r["theta"] / r["alpha"], "theta_alpha_ratio")
        put(f"{u}_TBR", r["theta"] / r["beta"], "theta_beta_ratio")
        put(f"{u}_SFR", (r["delta"] + r["theta"]) / (r["alpha"] + r["beta"]), "slow_fast_ratio")

    for name, (l, r) in PAIRS.items():                                         # hemispheric asymmetry indices
        for b in BAND_NAMES:
            put(f"asym_{name}_{b}", asym(unit_rel[(l, b)], unit_rel[(r, b)]), "spectral_asymmetry")
        put(f"asym_{name}_PE", asym(PE[units[l]].mean(), PE[units[r]].mean()), "other_complexity")
        put(f"asym_{name}_DE", asym(DE[units[l]].mean(), DE[units[r]].mean()), "other_complexity")
    return feats, fam


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pre_dir", default="outputs/preprocessed")
    ap.add_argument("--manifest", default="outputs/cohort_manifest.csv")
    ap.add_argument("--out", default="outputs/features.csv")
    ap.add_argument("--max_epochs", type=int, default=60, help="max accepted epochs per subject; 0 = use all")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--pe_order", type=int, default=3)
    args = ap.parse_args()

    pre = Path(args.pre_dir)
    ch_names = (pre / "channel_names.txt").read_text().split()
    needed = {c for chs in REGIONS.values() for c in chs} | set(MIDLINE)
    if needed - set(ch_names):
        raise SystemExit(f"Channels not found: {sorted(needed - set(ch_names))}; data has {ch_names}")

    manifest = pd.read_csv(args.manifest)
    rng = np.random.default_rng(args.seed)
    rows, fam = [], None
    for _, r in manifest.iterrows():
        sid = r["subject_id"]
        epochs = np.load(pre / "epochs" / f"{sid}_epochs.npy").astype(np.float64)
        n_all = epochs.shape[0]
        epochs = sample_epochs(epochs, args.max_epochs, rng)
        f, fam = subject_features(epochs, ch_names, pe_order=args.pe_order)
        rows.append({"subject_id": sid, "label": r["label"], "n_epochs_used": epochs.shape[0], **f})
        print(f"{sid}: {epochs.shape[0]}/{n_all} epochs used -> {len(f)} features")

    df = pd.DataFrame(rows)
    feat_cols = [c for c in df.columns if c not in ("subject_id", "label", "n_epochs_used")]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    pd.DataFrame({"feature": feat_cols, "family": [fam[c] for c in feat_cols]}).to_csv(
        Path(args.out).with_name(Path(args.out).stem + "_names.csv"), index=False)
    print(f"\nWrote {args.out}: {len(df)} subjects x {len(feat_cols)} features (NaNs: {int(df[feat_cols].isna().sum().sum())})")
    print("Features per family:", pd.Series([fam[c] for c in feat_cols]).value_counts().to_dict())
    print("Paper (Fig. 3a): absolute 124, relative 124, DE 31, PE 31, spectral asym 20, TAR 11, TBR 11, SFR 11, other 10")


if __name__ == "__main__":
    main()