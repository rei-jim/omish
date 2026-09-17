"""Recall-first operating point (step 1 revision: a missed Responder costs more than a false alarm).
at_recall(y, score, 0.9) -> (threshold, specificity, precision) at the loosest threshold that still catches >=90% of Responders."""
import numpy as np
from sklearn.metrics import roc_curve

def at_recall(y, score, target=0.9):
    fpr, tpr, thr = roc_curve(y, score)
    i = np.argmax(tpr >= target)                     # first (highest) threshold reaching the target recall
    y = np.asarray(y); pred = np.asarray(score) >= thr[i]
    prec = y[pred].mean() if pred.any() else np.nan
    return thr[i], 1 - fpr[i], prec

if __name__ == "__main__":
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1]); s = np.arange(10) / 10
    t, spec, prec = at_recall(y, s, 0.9)
    assert spec == 1.0 and prec == 1.0, (t, spec, prec)   # perfect ranking: 90% recall reachable with no false alarms
    assert at_recall(y, 1 - s, 0.9)[1] == 0.0               # reversed ranking: recall 90% forces every negative through
    print("ok")
