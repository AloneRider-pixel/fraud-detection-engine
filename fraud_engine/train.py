"""Train, calibrate a threshold, export ONNX, write metrics.json.

Time-based split (train on earlier steps, validate then test on later ones).
Imbalance is handled with XGBoost scale_pos_weight, evaluated with PR-AUC.
The threshold is picked on the validation split only, then reported once on test.
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve
from xgboost import XGBClassifier

from .data import generate
from .features import FEATURES, build_matrix


def pick_threshold(y, p, min_precision):
    """Highest recall subject to precision >= min_precision (validation only)."""
    prec, rec, thr = precision_recall_curve(y, p)
    ok = np.where(prec[:-1] >= min_precision)[0]
    if len(ok) == 0:
        return float(thr[np.argmax(prec[:-1] * rec[:-1])])
    best = ok[np.argmax(rec[ok])]
    return float(thr[best])


def export_onnx(model, path):
    from onnxmltools import convert_xgboost
    from onnxmltools.convert.common.data_types import FloatTensorType
    onx = convert_xgboost(model, initial_types=[("input", FloatTensorType([None, len(FEATURES)]))],
                          target_opset=15)
    with open(path, "wb") as f:
        f.write(onx.SerializeToString())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", help="real PaySim CSV (optional)")
    ap.add_argument("--rows", type=int, default=60000)
    ap.add_argument("--out", default="artifacts")
    ap.add_argument("--min-precision", type=float, default=0.90)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    df = pd.read_csv(a.csv) if a.csv else generate(a.rows, seed=a.seed)
    df = df.sort_values("step", kind="stable").reset_index(drop=True)
    X = build_matrix(df)
    y = df["isFraud"].to_numpy()
    n = len(df)
    i1, i2 = int(n * 0.6), int(n * 0.8)
    Xtr, ytr, Xva, yva, Xte, yte = X[:i1], y[:i1], X[i1:i2], y[i1:i2], X[i2:], y[i2:]

    spw = float((ytr == 0).sum() / max(1, (ytr == 1).sum()))
    model = XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.1,
                          scale_pos_weight=spw, eval_metric="aucpr", random_state=a.seed,
                          n_jobs=2)
    model.fit(Xtr, ytr)
    pva = model.predict_proba(Xva)[:, 1]
    pte = model.predict_proba(Xte)[:, 1]
    thr = pick_threshold(yva, pva, a.min_precision)
    pred = pte >= thr
    tp = int((pred & (yte == 1)).sum())
    metrics = {
        "data": "csv" if a.csv else "synthetic",
        "rows": n, "fraud_rate": float(y.mean()),
        "scale_pos_weight": spw,
        "val_pr_auc": float(average_precision_score(yva, pva)),
        "test_pr_auc": float(average_precision_score(yte, pte)),
        "threshold_chosen_on_validation": thr,
        "min_precision_target_on_validation": a.min_precision,
        "test_precision_at_threshold": float(tp / max(1, pred.sum())),
        "test_recall_at_threshold": float(tp / max(1, (yte == 1).sum())),
        "test_fraud_count": int(yte.sum()),
    }
    export_onnx(model, os.path.join(a.out, "model.onnx"))
    with open(os.path.join(a.out, "threshold.json"), "w") as f:
        json.dump({"threshold": thr, "features": FEATURES}, f)
    with open(os.path.join(a.out, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
