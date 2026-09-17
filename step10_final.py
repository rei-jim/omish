"""Step 10: the one and only look at the 18 held-out samples. Models are fitted on the 71 training samples with
hyperparameters fixed in advance (from steps 4-5), thresholds fixed from training out-of-fold predictions, then scored once.
Outputs: results/step10_test.txt, step10_test.png"""
import numpy as np, pandas as pd, matplotlib, warnings
matplotlib.use("Agg"); warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve
from split import ids
from metrics import at_recall

SEED = 0
df = pd.read_csv("Ornish.csv", index_col=0); y = (df["WeightLoss"] == "Responders").astype(int)
E = pd.read_csv("results/step03_eigengenes.csv", index_col=0).loc[df.index]
X = pd.concat([E, df[["Age", "Sex", "COPD", "Diabetes"]]], axis=1)
tr, te = ids("train"), ids("test")
models = {"ANOVA top-5 + ridge": make_pipeline(StandardScaler(), SelectKBest(f_classif, k=5), LogisticRegression(C=1.0, max_iter=5000)),
          "ridge, all 25 features": make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=5000)),
          "ME9 eigengene alone": None}
rng = np.random.default_rng(SEED)
out = [f"TEST SET, scored once: {len(te)} samples ({int(y[te].sum())} Responders, {int((1-y[te]).sum())} Non_responders). Models fitted on {len(tr)} training samples."]
fig, ax = plt.subplots(figsize=(5, 4.5))
for name, m in models.items():
    if m is None:
        s_tr, s_te = X.loc[tr, "ME9"], X.loc[te, "ME9"]
    else:
        s_tr = cross_val_predict(m, X.loc[tr], y[tr], cv=StratifiedKFold(5, shuffle=True, random_state=SEED), method="predict_proba")[:, 1]
        s_te = m.fit(X.loc[tr], y[tr]).predict_proba(X.loc[te])[:, 1]
    thr = at_recall(y[tr], s_tr, 0.9)[0]                       # threshold fixed on training OOF, never on test
    yt = y[te].values; pred = np.asarray(s_te) >= thr
    auc = roc_auc_score(yt, s_te); pr = average_precision_score(yt, s_te)
    boot = [roc_auc_score(yt[i], np.asarray(s_te)[i]) for i in (rng.integers(0, len(yt), len(yt)) for _ in range(2000)) if len(set(yt[i])) == 2]
    sens = pred[yt == 1].mean(); spec = 1 - pred[yt == 0].mean()
    out.append(f"\n{name}: test ROC-AUC {auc:.2f} (bootstrap 95% {np.percentile(boot, 2.5):.2f}-{np.percentile(boot, 97.5):.2f}), PR-AUC {pr:.2f} (chance {yt.mean():.2f});"
               f" at the training threshold for 90% recall: sensitivity {sens:.2f} ({int(pred[yt==1].sum())}/{int((yt==1).sum())}), specificity {spec:.2f} ({int((~pred[yt==0]).sum())}/{int((yt==0).sum())})")
    fpr, tpr, _ = roc_curve(yt, s_te); ax.plot(fpr, tpr, label=f"{name} (AUC {auc:.2f})")
ax.plot([0, 1], [0, 1], "--", c="grey"); ax.set(xlabel="1 - specificity", ylabel="sensitivity", title="Held-out test set (n=18), scored once"); ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig("results/step10_test.png", dpi=120)
open("results/step10_test.txt", "w").write("\n".join(out)); print("\n".join(out))
