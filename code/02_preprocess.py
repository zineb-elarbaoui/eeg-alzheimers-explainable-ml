import argparse
from pathlib import Path

import mne
import numpy as np
import pandas as pd

mne.set_log_level("ERROR")

L_FREQ, H_FREQ = 1.0, 40.0   # band-pass edges (Hz)
SFREQ_TARGET = 100.0         # resampling rate (Hz)
DISCARD_START_S = 60.0       # seconds dropped at the start of each recording
EPOCH_S = 5.0                # epoch length (s), no overlap
REJECT_EXTREME_UV = 500.0    # any single channel peak-to-peak above this -> reject
REJECT_MULTI_UV = 300.0      # ...or at least MULTI_MIN_CHANNELS channels above this -> reject
MULTI_MIN_CHANNELS = 2       # the paper does not say how many channels 'multi-channel' means (assumption)
REJECT_MEDIAN_UV = 150.0     # ...or the across-channel MEDIAN peak-to-peak above this -> reject


def preprocess_raw(raw: mne.io.BaseRaw):
    """Apply steps 2-7 to a loaded Raw object.

    Returns (data, ch_names, info) where data has shape (n_kept, n_channels, n_times)
    in microvolts, and info is a dict of counts for the summary table.
    """
    raw = raw.copy().pick("eeg")

    # 2. common average reference
    raw.set_eeg_reference("average", projection=False)

    # 3. band-pass filter (done BEFORE resampling so the 40 Hz edge is safely below the new Nyquist of 50 Hz)
    raw.filter(l_freq=L_FREQ, h_freq=H_FREQ, verbose="ERROR")

    # 4. resample
    raw.resample(SFREQ_TARGET)

    # 5. discard the first 60 s
    if raw.times[-1] <= DISCARD_START_S + EPOCH_S:
        raise ValueError("recording too short after discarding the first 60 s")
    raw.crop(tmin=DISCARD_START_S)

    # 6. non-overlapping fixed-length epochs
    epochs = mne.make_fixed_length_epochs(raw, duration=EPOCH_S, overlap=0.0, preload=True, verbose="ERROR")
    data_uv = epochs.get_data() * 1e6  # volts -> microvolts
    n_total = data_uv.shape[0]

    # 7. amplitude-based rejection (peak-to-peak per epoch and channel)
    ptp = data_uv.max(axis=2) - data_uv.min(axis=2)          # (n_epochs, n_channels)
    r_extreme = (ptp > REJECT_EXTREME_UV).any(axis=1)
    r_multi = (ptp > REJECT_MULTI_UV).sum(axis=1) >= MULTI_MIN_CHANNELS
    r_median = np.median(ptp, axis=1) > REJECT_MEDIAN_UV
    keep = ~(r_extreme | r_multi | r_median)
    data_kept = data_uv[keep].astype(np.float32)

    info = {
        "n_epochs_total": int(n_total),
        "n_epochs_kept": int(keep.sum()),
        "n_epochs_rejected": int((~keep).sum()),
        "rej_extreme": int(r_extreme.sum()),
        "rej_multi": int(r_multi.sum()),
        "rej_median": int(r_median.sum()),
        "median_ptp_uv": float(np.median(ptp)),   # sanity check on units
    }
    return data_kept, raw.ch_names, info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bids_root", required=True)
    ap.add_argument("--manifest", default="outputs/cohort_manifest.csv")
    ap.add_argument("--out_dir", default="outputs/preprocessed")
    args = ap.parse_args()

    bids_root = Path(args.bids_root)
    out_dir = Path(args.out_dir)
    (out_dir / "epochs").mkdir(parents=True, exist_ok=True)

    manifest = pd.read_csv(args.manifest)
    rows, ref_channels = [], None

    for _, r in manifest.iterrows():
        sid = r["subject_id"]
        set_path = bids_root / "derivatives" / sid / "eeg" / f"{sid}_task-eyesclosed_eeg.set"
        try:
            raw = mne.io.read_raw_eeglab(str(set_path), preload=True, verbose="ERROR")
            orig_sfreq, orig_dur = raw.info["sfreq"], raw.times[-1]
            data, ch_names, info = preprocess_raw(raw)

            if ref_channels is None:
                ref_channels = ch_names
            elif ch_names != ref_channels:
                raise ValueError(f"channel mismatch with first subject: {ch_names}")

            np.save(out_dir / "epochs" / f"{sid}_epochs.npy", data)
            rows.append({"subject_id": sid, "label": r["label"], "orig_sfreq": orig_sfreq,
                         "orig_duration_s": round(orig_dur, 1), "status": "ok", **info})
            print(f"{sid} ({r['label']}): {info['n_epochs_kept']}/{info['n_epochs_total']} epochs kept, "
                  f"median p-p {info['median_ptp_uv']:.0f} uV")
        except Exception as e:
            print(f"{sid}: FAILED - {e}")
            rows.append({"subject_id": sid, "label": r["label"], "status": f"failed: {e}"})

    if ref_channels:
        (out_dir / "channel_names.txt").write_text("\n".join(ref_channels))

    summary = pd.DataFrame(rows)
    summary.to_csv(out_dir / "preprocessing_summary.csv", index=False)

    ok = summary[summary["status"] == "ok"]
    print(f"\nDone: {len(ok)}/{len(summary)} subjects preprocessed. "
          f"Epochs kept per subject: min {ok['n_epochs_kept'].min()}, "
          f"median {int(ok['n_epochs_kept'].median())}, max {ok['n_epochs_kept'].max()}")
    print(f"Fraction of epochs rejected overall: "
          f"{ok['n_epochs_rejected'].sum() / ok['n_epochs_total'].sum():.1%}")


if __name__ == "__main__":
    main()