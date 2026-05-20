# bootstrap_ci.py
# Computes 95% bootstrap confidence intervals for per-class AUC and accuracy.
# Run AFTER evaluate.py has saved test_labels.npy, test_preds.npy, test_probs.npy
#
# Usage: python bootstrap_ci.py

import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score

OUTPUT_DIR   = "outputs"   # match your config.OUTPUT_DIR
N_BOOTSTRAP  = 10000
RANDOM_SEED  = 42
CLASS_NAMES  = ["DR0", "DR1", "DR2", "DR3", "DR4"]

rng = np.random.default_rng(RANDOM_SEED)

y_true = np.load(f"{OUTPUT_DIR}/test_labels.npy")
y_pred = np.load(f"{OUTPUT_DIR}/test_preds.npy")
y_prob = np.load(f"{OUTPUT_DIR}/test_probs.npy")  # [N, 5]

n = len(y_true)

auc_boots  = {c: [] for c in range(5)}
acc_boots  = {c: [] for c in range(5)}

for _ in range(N_BOOTSTRAP):
    idx = rng.integers(0, n, size=n)          # resample with replacement
    yt  = y_true[idx]
    yp  = y_pred[idx]
    ypr = y_prob[idx]

    for c in range(5):
        mask = (yt == c)
        if mask.sum() < 2:                    # need at least 2 samples for AUC
            continue
        # one-vs-rest AUC for this class
        try:
            auc = roc_auc_score((yt == c).astype(int), ypr[:, c])
            auc_boots[c].append(auc)
        except ValueError:
            pass
        # per-class accuracy
        acc = (yp[mask] == c).mean() * 100
        acc_boots[c].append(acc)

print("\n95% Bootstrap Confidence Intervals (10,000 resamples)")
print("=" * 55)
for c in range(5):
    name = CLASS_NAMES[c]
    n_c  = (y_true == c).sum()

    if auc_boots[c]:
        auc_arr = np.array(auc_boots[c])
        auc_lo, auc_hi = np.percentile(auc_arr, [2.5, 97.5])
        auc_pt = roc_auc_score((y_true == c).astype(int), y_prob[:, c])
        print(f"  {name} AUC  (n={n_c:3d}): {auc_pt:.3f}  "
              f"[{auc_lo:.3f}, {auc_hi:.3f}]")

    if acc_boots[c]:
        acc_arr = np.array(acc_boots[c])
        acc_lo, acc_hi = np.percentile(acc_arr, [2.5, 97.5])
        acc_pt = (y_pred[y_true == c] == c).mean() * 100
        print(f"  {name} ACC  (n={n_c:3d}): {acc_pt:.1f}%  "
              f"[{acc_lo:.1f}%, {acc_hi:.1f}%]")