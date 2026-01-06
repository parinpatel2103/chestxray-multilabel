import numpy as np
from sklearn.metrics import roc_auc_score, f1_score

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def tune_thresholds(probs, labels, t_min=0.05, t_max=0.40, steps=8):
    thresholds = np.linspace(t_min, t_max, steps)
    num_classes = labels.shape[1]

    best_thr = np.zeros(num_classes, dtype=np.float32)
    best_f1 = np.zeros(num_classes, dtype=np.float32)

    for c in range(num_classes):
        y_true = labels[:, c]
        y_prob = probs[:, c]

        best = 0.0
        best_t = 0.5
        for t in thresholds:
            y_pred = (y_prob >= t).astype(int)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            if f1 > best:
                best = f1
                best_t = t

        best_thr[c] = best_t
        best_f1[c] = best

    return best_thr, best_f1

def compute_auc_f1(probs, labels, thresholds, class_names):
    per_auc = []
    per_f1 = []

    for i, name in enumerate(class_names):
        y_true = labels[:, i]
        y_prob = probs[:, i]

        # AUC can fail if only one class present in labels
        try:
            auc = roc_auc_score(y_true, y_prob)
        except ValueError:
            auc = np.nan

        y_pred = (y_prob >= thresholds[i]).astype(int)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        per_auc.append(auc)
        per_f1.append(f1)

    macro_auc = float(np.nanmean(per_auc))
    macro_f1 = float(np.mean(per_f1))

    rows = []
    for name, auc, f1, thr in zip(class_names, per_auc, per_f1, thresholds):
        rows.append((name, auc, f1, float(thr)))

    return rows, macro_auc, macro_f1
