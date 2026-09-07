from __future__ import annotations

import numpy as np
from sklearn.datasets import load_iris
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import StratifiedKFold

from tabicl import TabICLClassifier

SEED = 1200
TARGET_AUROC = 0.9700000000000000
TARGET_AUPRC = 0.6758748826930644


def archived_iris_class0():
    d = load_iris()
    X0 = np.asarray(d.data, dtype=np.float32)
    yy = np.asarray(d.target)
    normal = np.flatnonzero(yy != 0)
    anomaly = np.flatnonzero(yy == 0)
    m = round(len(normal) * 0.1 / 0.9)
    rng = np.random.default_rng(SEED)
    sel = rng.choice(anomaly, size=m, replace=False)
    idx = np.r_[normal, sel]
    idx = idx[rng.permutation(len(idx))]
    return X0[idx], (yy[idx] == 0).astype(int)


def make_syn(N0, n_syn, rng, base_mode):
    n, d = N0.shape
    if base_mode == "all":
        assert n_syn == n
        base = np.arange(n)
    elif base_mode == "sample_no" and n_syn <= n:
        base = rng.choice(n, size=n_syn, replace=False)
    else:
        base = rng.integers(0, n, size=n_syn)
    donor = rng.integers(0, n - 1, size=n_syn)
    donor += donor >= base
    feat = rng.integers(0, d, size=n_syn)
    S = N0[base].copy()
    S[np.arange(n_syn), feat] = N0[donor, feat]
    return S


def evaluate(clf, X, y, nfold, count_rule):
    out = np.empty(len(y), dtype=float)
    skf = StratifiedKFold(n_splits=nfold, shuffle=True, random_state=SEED)
    fold_meta = []
    for fold, (tr, te) in enumerate(skf.split(X, y)):
        # Recovered archived convention: IForest seed is the dataset seed.
        pseudo = IsolationForest(
            n_estimators=300, contamination="auto", random_state=SEED
        ).fit(X[tr]).predict(X[tr])
        N0 = X[tr][pseudo == 1]
        A0 = X[tr][pseudo == -1]
        if count_rule == "a0":
            n_syn, base_mode = len(A0), "sample_no"
        elif count_rule == "balance":
            n_syn, base_mode = max(0, len(N0) - len(A0)), "sample_no"
        elif count_rule == "n0":
            n_syn, base_mode = len(N0), "all"
        else:
            raise ValueError(count_rule)
        rng = np.random.default_rng(SEED + fold)
        S = make_syn(N0, n_syn, rng, base_mode)
        C = np.concatenate([N0, A0, S], axis=0)
        yc = np.r_[np.zeros(len(N0), dtype=int), np.ones(len(A0) + len(S), dtype=int)]
        clf.fit(C, yc)
        p = clf.predict_proba(X[te])
        c1 = int(np.flatnonzero(clf.classes_ == 1)[0])
        out[te] = p[:, c1]
        fold_meta.append((len(N0), len(A0), n_syn))
    auc = float(roc_auc_score(y, out))
    ap = float(average_precision_score(y, out))
    return auc, ap, fold_meta


def main():
    X, y = archived_iris_class0()
    assert X.shape == (111, 4) and int(y.sum()) == 11
    print("target", TARGET_AUROC, TARGET_AUPRC)
    for wrapper_seed in [42, SEED]:
        clf = TabICLClassifier(
            n_estimators=8,
            random_state=wrapper_seed,
            device="cpu",
            use_amp=False,
            use_fa3=False,
            offload_mode="cpu",
            n_jobs=-1,
        )
        for nfold in [3, 5]:
            for count_rule in ["a0", "balance", "n0"]:
                auc, ap, meta = evaluate(clf, X, y, nfold, count_rule)
                err = abs(auc - TARGET_AUROC) + abs(ap - TARGET_AUPRC)
                print(
                    f"wrapper_seed={wrapper_seed} folds={nfold} count={count_rule} "
                    f"AUROC={auc:.16f} AUPRC={ap:.16f} aggregate_abs_err={err:.16f} meta={meta}",
                    flush=True,
                )


if __name__ == "__main__":
    main()
