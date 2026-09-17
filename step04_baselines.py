"""Step 4: pipelines + baselines on WGCNA eigengenes (+ clinical). Nested, repeated, stratified CV.
Outputs: results/step04_baselines.txt, results/step04_cv_auc.png, results/step04_oof.csv"""
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import roc_auc_score, average_precision_score
from metrics import at_recall

REPEATS, FOLDS, SEED = 20, 5, 0
from split import ids
df = pd.read_csv("Ornish.csv", index_col=0).loc[ids("train")]   # test set is never touched here
y = (df["WeightLoss"] == "Responders").astype(int)
clin = df[["Age", "Sex", "COPD", "Diabetes"]]
E = pd.read_csv("results/step03_eigengenes.csv", index_col=0).loc[df.index]
feature_sets = {"clinical only": clin, "eigengenes only": E, "eigengenes + clinical": pd.concat([E, clin], axis=1)}

def lr(): return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
grid = {"logisticregression__C": np.logspace(-3, 2, 11)}

def nested_oof(model, X, seed, tune):
    """Out-of-fold probabilities for one repeat. Inner CV picks C; outer folds score it."""
    oof = pd.Series(index=X.index, dtype=float)
    for tr, te in StratifiedKFold(FOLDS, shuffle=True, random_state=seed).split(X, y):
        m = GridSearchCV(model, grid, cv=StratifiedKFold(FOLDS, shuffle=True, random_state=seed), scoring="roc_auc") if tune else model
        m.fit(X.iloc[tr], y.iloc[tr])
        oof.iloc[te] = m.predict_proba(X.iloc[te])[:, 1]
    return oof

rows, oofs = [], {}
for name, X, tune in [("chance", clin, False)] + [(n, X, True) for n, X in feature_sets.items()]:
    model = DummyClassifier(strategy="prior") if name == "chance" else lr()
    per_rep = []
    for r in range(REPEATS):
        oof = nested_oof(model, X, SEED + r, tune)
        _, spec, prec = at_recall(y, oof, 0.9)
        per_rep.append((roc_auc_score(y, oof), average_precision_score(y, oof), spec, prec))
        oofs[f"{name}|{r}"] = oof
    per_rep = np.array(per_rep)
    rows.append([name, *np.nanmean(per_rep, 0), *np.percentile(per_rep[:, 0], [2.5, 97.5])])
res = pd.DataFrame(rows, columns=["model", "ROC-AUC", "PR-AUC", "Spec@90%Rec", "Prec@90%Rec", "AUC 2.5%", "AUC 97.5%"]).set_index("model")
pd.DataFrame(oofs).to_csv("results/step04_oof.csv")

# bootstrap CI on the pooled out-of-fold predictions of the main model (samples resampled with replacement)
rng = np.random.default_rng(SEED)
main = pd.DataFrame({r: oofs[f"eigengenes + clinical|{r}"] for r in range(REPEATS)}).mean(1)
boot = [roc_auc_score(y.iloc[i], main.iloc[i]) for i in (rng.integers(0, len(y), len(y)) for _ in range(2000)) if y.iloc[i].nunique() == 2]
lines = [f"TRAIN SET ONLY: {len(y)} samples. ", f"{REPEATS} repeats x {FOLDS}-fold stratified CV, C tuned by inner {FOLDS}-fold CV (nested). Prevalence {y.mean():.3f}.",
         "AUC 2.5%/97.5% = spread across repeats (how much the score depends on the fold split).",
         "Spec@90%Rec / Prec@90%Rec = specificity and precision at the loosest threshold that still catches 90% of Responders (recall-first operating point).", "",
         res.round(3).to_string(), "",
         f"bootstrap 95% CI of ROC-AUC, eigengenes + clinical, averaged OOF predictions: {np.percentile(boot, 2.5):.3f} - {np.percentile(boot, 97.5):.3f}"]
open("results/step04_baselines.txt", "w").write("\n".join(lines)); print("\n".join(lines))

fig, ax = plt.subplots(figsize=(7, 3.5))
data = [[roc_auc_score(y, oofs[f"{n}|{r}"]) for r in range(REPEATS)] for n in res.index]
ax.boxplot(data, tick_labels=res.index); ax.axhline(0.5, ls="--", c="grey"); ax.set(ylabel="ROC-AUC per repeat", title="Cross-validated AUC across 20 fold splits")
plt.setp(ax.get_xticklabels(), rotation=15); fig.tight_layout(); fig.savefig("results/step04_cv_auc.png", dpi=120)
