"""Step 1 of the Ornish project: baseline + first feature-selection pass.
89 samples, 13519 features -> everything supervised must live inside CV folds."""
import numpy as np, pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

df = pd.read_csv("Ornish.csv", index_col=0).dropna()
y = (df.pop("WeightLoss") == "Responders").astype(int)
X = df

# Unsupervised pre-filter: drop the 50% lowest-variance genes (label not used, so no leakage).
clin = ["Age", "Sex", "COPD", "Diabetes"]
genes = X.drop(columns=clin)
keep = genes.var().sort_values(ascending=False).index[: len(genes.columns) // 2]
X = pd.concat([X[clin], genes[keep]], axis=1)
print("after variance filter:", X.shape)

cv = StratifiedKFold(5, shuffle=True, random_state=0)
def score(name, model, cols=None):
    s = cross_val_score(model, X if cols is None else X[cols], y, cv=cv, scoring="roc_auc")
    print(f"{name:35s} AUC {s.mean():.3f} +/- {s.std():.3f}")

score("chance (DummyClassifier)", DummyClassifier())
score("ridge on all genes (no selection)", make_pipeline(StandardScaler(), LogisticRegression(C=0.01, max_iter=1000)))
score("clinical only", make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)), clin)
for k in (20, 100, 500):
    score(f"ANOVA top-{k} + logistic",
          make_pipeline(StandardScaler(), SelectKBest(f_classif, k=k), LogisticRegression(max_iter=1000)))
score("L1 logistic (embedded)",
      make_pipeline(StandardScaler(), LogisticRegression(l1_ratio=1, solver="liblinear", C=0.1)))
