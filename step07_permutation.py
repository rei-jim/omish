"""Step 7: permutation test of the whole supervised pipeline, training set only.
Shuffle the labels, rerun cross-validated selection + model, record the AUC. Repeat N_PERM times.
Two pipelines: ridge on all features; ANOVA top-5 filter + ridge (selection inside the folds).
Outputs: results/step07_permutation.txt, step07_permutation.png"""
import numpy as np, pandas as pd, matplotlib, warnings
matplotlib.use("Agg"); warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score
from split import ids

N_PERM, REPEATS, FOLDS, SEED = 500, 5, 5, 0
df = pd.read_csv("Ornish.csv", index_col=0).loc[ids("train")]
y = (df["WeightLoss"] == "Responders").astype(int).values
E = pd.read_csv("results/step03_eigengenes.csv", index_col=0).loc[df.index]
X = pd.concat([E, df[["Age", "Sex", "COPD", "Diabetes"]]], axis=1).values
pipes = {"ridge, all 25 features": make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=5000)),
         "ANOVA top-5 + ridge": make_pipeline(StandardScaler(), SelectKBest(f_classif, k=5), LogisticRegression(C=1.0, max_iter=5000))}

def cv_auc(pipe, labels):
    """AUC of out-of-fold predictions, averaged over REPEATS fold assignments."""
    return np.mean([roc_auc_score(labels, cross_val_predict(pipe, X, labels, cv=StratifiedKFold(FOLDS, shuffle=True, random_state=SEED + r), method="predict_proba")[:, 1]) for r in range(REPEATS)])

rng = np.random.default_rng(SEED)
perms = [rng.permutation(y) for _ in range(N_PERM)]
out = [f"TRAIN SET ONLY: {len(y)} samples. {N_PERM} label permutations; each scored by {REPEATS}x{FOLDS}-fold CV (selection inside folds)."]
fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
for ax, (name, pipe) in zip(axes, pipes.items()):
    real = cv_auc(pipe, y)
    null = np.array([cv_auc(pipe, p) for p in perms])
    pval = (1 + (null >= real).sum()) / (1 + N_PERM)
    out.append(f"\n{name}: real AUC {real:.3f}; null mean {null.mean():.3f}, 95th percentile {np.percentile(null, 95):.3f}, max {null.max():.3f}; permutation p = {pval:.3f}")
    ax.hist(null, bins=30, color="grey", alpha=0.7, label="shuffled labels"); ax.axvline(real, c="C3", lw=2, label=f"real labels (p={pval:.3f})")
    ax.set(title=name, xlabel="cross-validated ROC-AUC", ylabel="permutations"); ax.legend()
fig.tight_layout(); fig.savefig("results/step07_permutation.png", dpi=120)
open("results/step07_permutation.txt", "w").write("\n".join(out)); print("\n".join(out))
