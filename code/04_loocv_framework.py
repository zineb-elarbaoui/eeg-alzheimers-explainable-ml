"""
Step 4 - Fold-wise LOOCV framework

A reusable, leakage-safe pipeline matching Shen et al. (2026), section 2.4:

For each held-out subject:
  1. Median-impute missing values (fit on training fold only)
  2. Drop zero-variance features (fit on training fold only)
  3. Rank remaining features by |AUC - 0.5| (univariate, training fold only)
  4. Keep the top N_PRESELECT (120) ranked features
  5. Spearman redundancy pruning: walk the ranked list, keep a feature only if
     |correlation| < RHO_THRESH (0.90) with every feature already kept, stop at k features
  6. Standardize (fit on training fold only)
  7. Fit the classifier on the training fold
  8. Transform + predict the one held-out subject


"""

import inspect
from dataclasses import dataclass, field

import numpy as np
from sklearn.metrics import roc_auc_score

RHO_THRESH = 0.90
N_PRESELECT = 120


def univariate_auc_scores(X, y):
    """|AUC - 0.5| for every column of X against binary y (1 = positive class). Returns (n_features,)."""
    scores = np.empty(X.shape[1])
    for j in range(X.shape[1]):
        col = X[:, j]
        if np.all(col == col[0]):          # constant column -> no separating power
            scores[j] = 0.0
            continue
        auc = roc_auc_score(y, col)
        scores[j] = abs(auc - 0.5)
    return scores


def _wants_y_train(make_classifier):
    """True if make_classifier's signature accepts an argument (the fold's y_train)."""
    try:
        return len(inspect.signature(make_classifier).parameters) >= 1
    except (TypeError, ValueError):
        return False


def spearman_corr_matrix(X):
    """Spearman correlation matrix via rank + Pearson (no scipy dependency needed beyond rankdata)."""
    from scipy.stats import rankdata
    R = np.apply_along_axis(rankdata, 0, X)
    return np.corrcoef(R, rowvar=False)


SELECTION_MODES = ("full", "no_pruning", "no_ranking")


def select_features(X_train, y_train, k, n_preselect=N_PRESELECT, rho_thresh=RHO_THRESH,
                    mode="full", rng=None):
    """Fit feature selection on TRAINING DATA ONLY.

    mode (ablation switch for Step 6 of the robustness study):
      "full"       the paper's pipeline: rank by |AUC-0.5|, keep the top n_preselect,
                   Spearman-prune (|rho| >= rho_thresh dropped) down to k features.
      "no_pruning" rank by |AUC-0.5| and keep the top k directly -- NO redundancy control.
      "no_ranking" NO outcome information at all: visit the features in a random order
                   (drawn from rng) and Spearman-prune down to k. Isolates how much the
                   supervised AUC ranking contributes beyond mere de-correlation.
                   Needs an rng (np.random.Generator); labels are never looked at.

    Returns: selected_idx (indices into the ORIGINAL feature array, in selection order)
    """
    if mode not in SELECTION_MODES:
        raise ValueError(f"mode must be one of {SELECTION_MODES}, got {mode!r}")
    n_features = X_train.shape[1]

    if mode == "no_pruning":
        scores = univariate_auc_scores(X_train, y_train)
        return np.argsort(-scores, kind="stable")[:min(k, n_features)]

    if mode == "no_ranking":
        if rng is None:
            raise ValueError("mode='no_ranking' needs rng (a np.random.Generator) for the random feature order")
        preselect = rng.permutation(n_features)         # random visiting order, ALL features, no labels used
    else:                                               # "full"
        scores = univariate_auc_scores(X_train, y_train)
        ranked = np.argsort(-scores)                    # best first
        preselect = ranked[:min(n_preselect, n_features)]

    if len(preselect) == 0:
        return np.array([], dtype=int)

    corr = spearman_corr_matrix(X_train[:, preselect])
    corr = np.nan_to_num(corr, nan=0.0)                 # a zero-variance column -> corr undefined -> treat as uncorrelated

    selected_local = [0]                                # best-ranked feature is always kept
    for i in range(1, len(preselect)):
        if len(selected_local) >= k:
            break
        max_corr = np.max(np.abs(corr[i, selected_local]))
        if max_corr < rho_thresh:
            selected_local.append(i)

    return preselect[selected_local[:k]]


@dataclass
class FoldResult:
    subject_id: str
    y_true: int
    y_pred: int
    proba: float          # predicted probability of the positive class (AD)
    selected_features: list = field(default_factory=list)
    coefficients: dict = field(default_factory=dict)     # feature_name -> fitted coefficient, if the model exposes one


def run_loocv(X, y, subject_ids, feature_names, make_classifier, k=12,
              n_preselect=N_PRESELECT, rho_thresh=RHO_THRESH, positive_label="AD",
              use_feature_selection=True, standardize=True,
              selection_mode="full", selection_seed=2026):
    """
    X: (n_subjects, n_features) raw feature matrix (may contain NaN)
    y: (n_subjects,) array of "AD"/"HC" strings (or any two labels)
    subject_ids: (n_subjects,) list/array of subject ids, same order as X
    feature_names: list of length n_features
    make_classifier: callable returning a fresh, unfitted sklearn-style classifier
                      (must implement fit(X, y) and predict_proba(X) or decision_function(X)).
                      May take zero arguments (e.g. the four Step 5 linear models), or one
                      argument -- the fold's TRAINING labels (y_train, 0/1 ints) -- for
                      classifiers that need a fold-specific class-imbalance setting, such as
                      XGBoost's scale_pos_weight (which has no "class_weight='balanced'" option).
                      Detected automatically from the callable's signature.
    k: number of features to retain per fold (ignored if use_feature_selection=False)
    use_feature_selection: if False, skip the rank+Spearman-prune step entirely and hand the
        classifier every feature that survives imputation and zero-variance filtering. This is
        option (B) from the roadmap for tree models with embedded feature handling (RF/XGBoost):
        impute and drop zero-variance features (still fold-wise, still leakage-safe), but let the
        model itself do the feature selection instead of the linear-model-oriented univariate
        ranking + redundancy pruning. Everything else about the fold loop (impute/scale fit on
        training data only, held-out subject only ever transformed) is unchanged.
    standardize: tree-based models don't need standardized inputs, so this can be set to False
        for them; the median-impute + zero-variance steps still always run.
    selection_mode: which selection pipeline to run when use_feature_selection=True --
        "full" (the paper: AUC ranking + Spearman pruning, the default), "no_pruning"
        (AUC ranking only), or "no_ranking" (random order + Spearman pruning, no labels
        used). See select_features. Every mode is fit on the training fold only.
        "No selection at all" is use_feature_selection=False.
    selection_seed: only used by "no_ranking". Fold i draws its random feature order from
        np.random.default_rng([selection_seed, i]), so a given seed is fully reproducible and
        different seeds give independent random orders.

    Returns: list of FoldResult, one per subject, in the same order as X.
    """
    n = X.shape[0]
    y_bin = (np.asarray(y) == positive_label).astype(int)
    results = []

    for i in range(n):
        train_idx = np.array([j for j in range(n) if j != i])
        test_idx = np.array([i])

        X_train_raw, X_test_raw = X[train_idx], X[test_idx]
        y_train = y_bin[train_idx]

        # 1-2. median impute + drop zero-variance, fit on training fold only
        col_median = np.nanmedian(X_train_raw, axis=0)
        X_train = np.where(np.isnan(X_train_raw), col_median, X_train_raw)
        X_test = np.where(np.isnan(X_test_raw), col_median, X_test_raw)

        var = X_train.var(axis=0)
        keep_var = var > 0
        X_train, X_test = X_train[:, keep_var], X_test[:, keep_var]
        kept_names = np.array(feature_names)[keep_var]

        # 3-5. rank + preselect + Spearman prune, fit on training fold only (option A)
        #      -- or skipped entirely, keeping every surviving feature (option B)
        if use_feature_selection:
            fold_rng = np.random.default_rng([selection_seed, i]) if selection_mode == "no_ranking" else None
            sel_idx = select_features(X_train, y_train, k=k, n_preselect=n_preselect,
                                      rho_thresh=rho_thresh, mode=selection_mode, rng=fold_rng)
            X_train_sel, X_test_sel = X_train[:, sel_idx], X_test[:, sel_idx]
            sel_names = kept_names[sel_idx]
        else:
            X_train_sel, X_test_sel = X_train, X_test
            sel_names = kept_names

        # 6. standardize, fit on training fold only (skippable for tree models)
        if standardize:
            mu, sd = X_train_sel.mean(axis=0), X_train_sel.std(axis=0)
            sd_safe = np.where(sd == 0, 1.0, sd)
            X_train_std = (X_train_sel - mu) / sd_safe
            X_test_std = (X_test_sel - mu) / sd_safe
        else:
            X_train_std, X_test_std = X_train_sel, X_test_sel

        # 7-8. fit + predict
        # make_classifier may want the training fold's own labels (e.g. to compute
        # XGBoost's scale_pos_weight) -- detected once via signature, not per fold,
        # but called fresh each fold so the weight reflects THIS fold's training labels.
        clf = make_classifier(y_train) if _wants_y_train(make_classifier) else make_classifier()
        clf.fit(X_train_std, y_train)
        # Score used for AUC/ranking: predict_proba when available (Logistic, LDA),
        # otherwise the raw decision_function (Linear SVM has no predict_proba by default).
        if hasattr(clf, "predict_proba"):
            proba = float(clf.predict_proba(X_test_std)[0, 1])
        else:
            proba = float(clf.decision_function(X_test_std)[0])
        # Hard label: always clf.predict(), which uses the classifier's own decision
        # boundary (threshold 0.5 on proba, or 0 on decision_function) rather than a
        # threshold we impose -- this matters once class_weight="balanced" shifts things.
        pred = int(clf.predict(X_test_std)[0])

        coefs = {}
        if hasattr(clf, "coef_"):
            coefs = dict(zip(sel_names, clf.coef_.ravel()))
        elif hasattr(clf, "feature_importances_"):   # tree models: RF, XGBoost
            coefs = dict(zip(sel_names, clf.feature_importances_.ravel()))

        results.append(FoldResult(
            subject_id=subject_ids[i], y_true=int(y_bin[i]), y_pred=pred, proba=float(proba),
            selected_features=list(sel_names), coefficients=coefs,
        ))

    return results


def summarize(results):
    """AUC, accuracy, sensitivity (recall on positive=AD), specificity, precision, F1."""
    y_true = np.array([r.y_true for r in results])
    y_pred = np.array([r.y_pred for r in results])
    proba = np.array([r.proba for r in results])

    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())

    sens = tp / (tp + fn) if (tp + fn) else float("nan")
    spec = tn / (tn + fp) if (tn + fp) else float("nan")
    prec = tp / (tp + fp) if (tp + fp) else float("nan")
    f1 = 2 * prec * sens / (prec + sens) if (prec + sens) else float("nan")
    acc = (tp + tn) / len(results)
    auc = roc_auc_score(y_true, proba)

    return {"AUC": auc, "ACC": acc, "SEN": sens, "SPE": spec, "PRE": prec, "F1": f1,
            "TP": tp, "TN": tn, "FP": fp, "FN": fn}


# ---------------------------------------------------------------------------
# Self-test: verifies the leakage-safety and correctness of select_features
# and run_loocv on synthetic data with a KNOWN informative feature.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from sklearn.linear_model import LogisticRegression

    rng = np.random.default_rng(0)
    n_subj, n_feat = 40, 50
    y = np.array(["AD"] * 20 + ["HC"] * 20)
    X = rng.standard_normal((n_subj, n_feat))
    X[:20, 0] += 3.0                      # feature 0: strongly informative
    X[:20, 1] += 3.0
    X[:, 2] = X[:, 0] + rng.normal(0, 0.01, n_subj)   # feature 2: near-duplicate of feature 0 -> should be pruned
    X[0, 5] = np.nan                                  # a missing value to impute
    names = [f"f{j}" for j in range(n_feat)]
    ids = [f"sub-{i:03d}" for i in range(n_subj)]

    results = run_loocv(X, y, ids, names, lambda: LogisticRegression(max_iter=1000), k=6)
    m = summarize(results)
    print("Self-test LOOCV metrics (expect AUC close to 1.0, informative features selected most often):")
    print(m)

    from collections import Counter
    freq = Counter(f for r in results for f in r.selected_features)
    print("\nTop selected features (expect f0/f1 near top, f2 rare due to Spearman pruning against f0):")
    print(freq.most_common(8))

    # Leakage sanity check: refit feature selection on ALL data vs. fold 0's training data
    # and confirm fold 0 does NOT use information from the held-out subject's label.
    sel_fold0 = set(results[0].selected_features)
    X_train0 = X[1:]  # everyone except subject 0
    y_train0 = (y[1:] == "AD").astype(int)
    sel_direct = set(np.array(names)[select_features(
        np.where(np.isnan(X_train0), np.nanmedian(X_train0, axis=0), X_train0), y_train0, k=6)])
    assert sel_fold0 == sel_direct, "leakage check failed: fold selection does not match a direct refit on the training fold"
    print("\nLeakage check passed: fold 0's selected features exactly reproduce a fit on the training fold alone.")

    # ---- Ablation modes (Step 6 of the robustness study) ----
    Xi = np.where(np.isnan(X_train0), np.nanmedian(X_train0, axis=0), X_train0)

    # no_pruning == plain top-k by univariate |AUC - 0.5|, and keeps the near-duplicate f2
    sel_np = select_features(Xi, y_train0, k=6, mode="no_pruning")
    assert list(sel_np) == list(np.argsort(-univariate_auc_scores(Xi, y_train0), kind="stable")[:6])
    print(f"no_pruning check passed: top-6 by AUC = {[names[j] for j in sel_np]} "
          f"(near-duplicate f2 kept: {'f2' in [names[j] for j in sel_np]})")

    # no_ranking never looks at the labels: same rng -> same features whatever the labels are,
    # and the near-duplicate pair (f0, f2) can never both survive the Spearman pruning
    sel_a = select_features(Xi, y_train0, k=6, mode="no_ranking", rng=np.random.default_rng(1))
    sel_b = select_features(Xi, 1 - y_train0, k=6, mode="no_ranking", rng=np.random.default_rng(1))
    sel_c = select_features(Xi, y_train0, k=6, mode="no_ranking", rng=np.random.default_rng(2))
    assert list(sel_a) == list(sel_b), "no_ranking must not depend on the labels"
    assert not ({0, 2} <= set(sel_a)) and not ({0, 2} <= set(sel_c)), "Spearman pruning must drop the duplicate"
    print(f"no_ranking check passed: label-independent, reproducible, duplicate pruned "
          f"(seed1 -> {[names[j] for j in sel_a]}, seed2 -> {[names[j] for j in sel_c]})")