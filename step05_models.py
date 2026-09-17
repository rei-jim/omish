"""Step 5: regularisation types and model families on eigengenes + clinical, training set only.
Same outer folds as step 4 so rows are comparable. Outputs: results/step05_models.txt, step05_selection.csv, step05_cv_auc.png"""
import numpy as np, pandas as pd, matplotlib, warnings
matplotlib.use("Agg"); warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import roc_auc_score, average_precision_score
from metrics import at_recall
from xgboost import XGBClassifier
from split import ids

REPEATS, FOLDS, SEED = 10, 5, 0
df = pd.read_csv("Ornish.csv", index_col=0).loc[ids("train")]
y = (df["WeightLoss"] == "Responders").astype(int)
E = pd.read_csv("results/step03_eigengenes.csv", index_col=0).loc[df.index]
X = pd.concat([E, df[["Age", "Sex", "COPD", "Diabetes"]]], axis=1)
Cs = np.logspace(-3, 2, 11)

models = {
    "L2 ridge": (make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000)), {"logisticregression__C": Cs}),
    "L1 lasso": (make_pipeline(StandardScaler(), LogisticRegression(penalty="l1", solver="liblinear", max_iter=5000)), {"logisticregression__C": Cs}),
    "elastic net": (make_pipeline(StandardScaler(), LogisticRegression(penalty="elasticnet", solver="saga", l1_ratio=0.5, max_iter=20000)), {"logisticregression__C": Cs}),
    "ANOVA filter + L2": (make_pipeline(StandardScaler(), SelectKBest(f_classif), LogisticRegression(max_iter=5000)), {"selectkbest__k": [1, 2, 3, 5, 8, 12], "logisticregression__C": [0.1, 1, 10]}),
    "XGBoost": (XGBClassifier(n_estimators=200, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, random_state=SEED, eval_metric="logloss", n_jobs=1),
                {"max_depth": [1, 2, 3], "reg_lambda": [1, 10], "min_child_weight": [1, 5]}),
}

def selected(fitted, name):
    if "filter" in name:
        return set(X.columns[fitted.named_steps["selectkbest"].get_support()])
    if "L1" in name or "elastic" in name:
        return set(X.columns[fitted.named_steps["logisticregression"].coef_[0] != 0])
    return set()

rows, per_rep_auc, sel_counts, n_fits = [], {}, {n: pd.Series(0, index=X.columns) for n in models}, 0
for name, (model, grid) in models.items():
    aucs, prs, specs, precs = [], [], [], []
    for r in range(REPEATS):
        oof = pd.Series(index=X.index, dtype=float)
        for tr, te in StratifiedKFold(FOLDS, shuffle=True, random_state=SEED + r).split(X, y):
            gs = GridSearchCV(model, grid, cv=StratifiedKFold(FOLDS, shuffle=True, random_state=SEED + r), scoring="roc_auc", n_jobs=-1)
            gs.fit(X.iloc[tr], y.iloc[tr])
            oof.iloc[te] = gs.predict_proba(X.iloc[te])[:, 1]
            for f in selected(gs.best_estimator_, name): sel_counts[name][f] += 1
            n_fits += 1
        aucs.append(roc_auc_score(y, oof)); prs.append(average_precision_score(y, oof))
        _, sp, pr = at_recall(y, oof, 0.9); specs.append(sp); precs.append(pr)
    per_rep_auc[name] = aucs
    rows.append([name, np.mean(aucs), np.mean(prs), np.mean(specs), np.nanmean(precs), *np.percentile(aucs, [2.5, 97.5])])
res = pd.DataFrame(rows, columns=["model", "ROC-AUC", "PR-AUC", "Spec@90%Rec", "Prec@90%Rec", "AUC 2.5%", "AUC 97.5%"]).set_index("model")

n_outer = REPEATS * FOLDS
sel = pd.DataFrame({n: sel_counts[n] / n_outer for n in ("L1 lasso", "elastic net", "ANOVA filter + L2")})
sel.to_csv("results/step05_selection.csv")
lines = [f"TRAIN SET ONLY: {len(y)} samples. {REPEATS} repeats x {FOLDS}-fold outer CV, hyperparameters by inner {FOLDS}-fold CV.", "",
         res.round(3).to_string(), "",
         f"Fraction of the {n_outer} outer-fold models in which each feature was kept (sparse models only), top 10 by lasso:",
         sel.sort_values("L1 lasso", ascending=False).head(10).round(2).to_string(), "",
         "mean number of features kept: " + ", ".join(f"{n}={sel_counts[n].sum() / n_outer:.1f}" for n in sel.columns)]
open("results/step05_models.txt", "w").write("\n".join(lines)); print("\n".join(lines))

fig, ax = plt.subplots(figsize=(8, 3.5))
ax.boxplot(list(per_rep_auc.values()), tick_labels=list(per_rep_auc)); ax.axhline(0.5, ls="--", c="grey")
ax.set(ylabel="ROC-AUC per repeat", title=f"Regularisation / model comparison, {REPEATS} fold assignments, training set")
plt.setp(ax.get_xticklabels(), rotation=15); fig.tight_layout(); fig.savefig("results/step05_cv_auc.png", dpi=120)
