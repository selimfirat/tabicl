from __future__ import annotations

import hashlib
import io
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score, roc_auc_score

BASE = "https://raw.githubusercontent.com/Minqi824/ADBench/main/adbench/datasets/Classical"
OUT = Path("experiments/tcse_heldout_if_results.csv")

# Frozen before heldout scoring.
CUTOFF = 1.5
IF_KW = dict(n_estimators=300, contamination="auto", random_state=42)

SPECS = [
    # Regression control: must reproduce the archived development result exactly.
    dict(name="Pima", file="29_Pima.npz", n=768, d=8, anomalies=268, holdout=False),
    # Predeclared sealed block, order preserved.
    dict(name="thyroid", file="38_thyroid.npz", n=3772, d=6, anomalies=93, holdout=True),
    dict(name="cardio", file="6_cardio.npz", n=1831, d=21, anomalies=176, holdout=True),
    dict(name="glass", file="14_glass.npz", n=214, d=7, anomalies=9, holdout=True),
    dict(name="mammography", file="23_mammography.npz", n=11183, d=6, anomalies=260, holdout=True),
    dict(name="annthyroid", file="2_annthyroid.npz", n=7200, d=6, anomalies=534, holdout=True),
]


def fetch_npz(filename: str):
    url = f"{BASE}/{filename}"
    with urllib.request.urlopen(url, timeout=60) as r:
        raw = r.read()
    sha = hashlib.sha256(raw).hexdigest()
    z = np.load(io.BytesIO(raw), allow_pickle=True)
    X = np.asarray(z["X"], dtype=float)
    y = np.asarray(z["y"]).reshape(-1).astype(int)
    return X, y, sha, url


def evaluate(spec):
    X, y, sha, url = fetch_npz(spec["file"])
    assert X.shape == (spec["n"], spec["d"]), (spec["name"], X.shape)
    assert len(y) == spec["n"], (spec["name"], len(y))
    assert int(y.sum()) == spec["anomalies"], (spec["name"], int(y.sum()))
    assert set(np.unique(y)).issubset({0, 1}), (spec["name"], np.unique(y))
    assert np.isfinite(X).all(), f"non-finite X in {spec['name']}"

    model = IsolationForest(**IF_KW).fit(X)
    score = -model.score_samples(X)
    q25, q75, q90, q99 = np.quantile(score, [0.25, 0.75, 0.90, 0.99])
    iqr = q75 - q25
    tail_sharp = float((q99 - q90) / iqr)
    auroc = float(roc_auc_score(y, score))
    auprc = float(average_precision_score(y, score))
    choice = "SE-1" if tail_sharp <= CUTOFF else "IForest"
    return dict(
        dataset=spec["name"], holdout=spec["holdout"], n=len(y), d=X.shape[1],
        anomalies=int(y.sum()), contamination=float(y.mean()), sha256=sha, url=url,
        q25=float(q25), q75=float(q75), q90=float(q90), q99=float(q99),
        tail_sharp=tail_sharp, cutoff=CUTOFF, tcse_choice=choice,
        if_auroc=auroc, if_auprc=auprc,
    )


def main():
    rows = [evaluate(s) for s in SPECS]
    df = pd.DataFrame(rows)

    # Exact archived regression target. Tiny tolerance catches protocol/version drift.
    p = df[df.dataset == "Pima"].iloc[0]
    assert abs(p.if_auroc - 0.6723880597014925) < 1e-12, p.to_dict()
    assert abs(p.if_auprc - 0.5031704502475960) < 1e-12, p.to_dict()
    assert abs(p.tail_sharp - 1.2224432848655550) < 1e-10, p.to_dict()
    assert p.tcse_choice == "SE-1", p.to_dict()

    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print(df[["dataset", "tail_sharp", "tcse_choice", "if_auroc", "if_auprc", "sha256"]].to_string(index=False))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
