import argparse
import sys
from pathlib import Path

import pandas as pd

GROUP_LABELS = {"A": "AD", "C": "HC"}  # F (FTD) is intentionally dropped


def build_cohort(participants_tsv: Path) -> pd.DataFrame:
    df = pd.read_csv(participants_tsv, sep="\t")
    df.columns = [c.strip() for c in df.columns]

    before = len(df)
    group_counts_full = df["Group"].value_counts().to_dict()

    cohort = df[df["Group"].isin(GROUP_LABELS.keys())].copy()
    cohort["label"] = cohort["Group"].map(GROUP_LABELS)
    cohort = cohort.rename(
        columns={"participant_id": "subject_id", "Gender": "sex", "Age": "age", "MMSE": "mmse"}
    )
    cohort = cohort[["subject_id", "label", "Group", "age", "sex", "mmse"]]
    cohort = cohort.sort_values("subject_id").reset_index(drop=True)

    print(f"Full dataset: {before} subjects -> {group_counts_full}")
    print(f"After dropping FTD (F): {len(cohort)} subjects "
          f"({(cohort['label'] == 'AD').sum()} AD, {(cohort['label'] == 'HC').sum()} HC)")

    return cohort


def check_derivative_file(bids_root: Path, subject_id: str) -> tuple[str, str]:
    """Return (path_str, status) where status is one of:
    'ok', 'missing', 'unreadable'."""
    set_path = bids_root / "derivatives" / subject_id / "eeg" / f"{subject_id}_task-eyesclosed_eeg.set"

    if not set_path.exists():
        return str(set_path), "missing"

    try:
        import mne  
        mne.set_log_level("ERROR")
        raw = mne.io.read_raw_eeglab(str(set_path), preload=False, verbose="ERROR")
        n_ch = len(raw.ch_names)
        dur = raw.times[-1]
        if n_ch == 0 or dur <= 0:
            return str(set_path), "unreadable"
        return str(set_path), "ok"
    except Exception as e:
        print(f"  [unreadable] {subject_id}: {e}")
        return str(set_path), "unreadable"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bids_root", type=str, default=None,
                         help="Path to the local ds004504 dataset root (contains participants.tsv and derivatives/)")
    parser.add_argument("--participants_tsv", type=str, default="data/participants.tsv",
                         help="Fallback path to participants.tsv if bids_root is not given")
    parser.add_argument("--out", type=str, default="data/cohort_manifest.csv")
    args = parser.parse_args()

    if args.bids_root:
        bids_root = Path(args.bids_root)
        participants_tsv = bids_root / "participants.tsv"
    else:
        bids_root = None
        participants_tsv = Path(args.participants_tsv)

    if not participants_tsv.exists():
        sys.exit(f"participants.tsv not found at {participants_tsv}")

    cohort = build_cohort(participants_tsv)

    statuses, paths = [], []
    for sid in cohort["subject_id"]:
        if bids_root is None:
            paths.append("")
            statuses.append("not_checked")
        else:
            p, s = check_derivative_file(bids_root, sid)
            paths.append(p)
            statuses.append(s)

    cohort["set_path"] = paths
    cohort["file_status"] = statuses

    n_bad = (cohort["file_status"].isin(["missing", "unreadable"])).sum()
    if n_bad:
        print(f"\n{n_bad} subject(s) flagged missing/unreadable and will be dropped from the final cohort:")
        print(cohort.loc[cohort["file_status"].isin(["missing", "unreadable"]),
                          ["subject_id", "label", "file_status"]].to_string(index=False))
        cohort_clean = cohort[~cohort["file_status"].isin(["missing", "unreadable"])].copy()
    else:
        cohort_clean = cohort.copy()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cohort_clean.to_csv(out_path, index=False)

    print(f"\nFinal cohort written to {out_path}: {len(cohort_clean)} subjects "
          f"({(cohort_clean['label'] == 'AD').sum()} AD, {(cohort_clean['label'] == 'HC').sum()} HC)")


if __name__ == "__main__":
    main()