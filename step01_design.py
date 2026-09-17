"""Step 1: experiment design. Produces the numbers behind the design choices.
No modelling. Output: results/step01_design.txt"""
import numpy as np, pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score

df = pd.read_csv("Ornish.csv", index_col=0)
y = (df["WeightLoss"] == "Responders").astype(int)
clin = df[["Age", "Sex", "COPD", "Diabetes"]]
out = []

out.append(f"samples: {len(y)}   Responders: {y.sum()}   Non_responders: {(1-y).sum()}")
out.append(f"majority-class accuracy (chance baseline): {y.mean():.3f}")
out.append(f"chance AUC: 0.500\n")

# How wide would a single hold-out test set be? Bootstrap AUC width for n_test=18 vs full CV n=89.
rng = np.random.default_rng(0)
def auc_ci_width(n, true_auc=0.75, B=300):
    # simulate scores with the given AUC, bootstrap the AUC, return 95% CI width
    widths = []
    for _ in range(10):
        yy = rng.binomial(1, y.mean(), n)
        s = rng.normal(0, 1, n) + yy * (np.sqrt(2) * stats.norm.ppf(true_auc))
        if yy.min() == yy.max(): continue
        b = [roc_auc_score(yy[i], s[i]) for i in (rng.integers(0, n, n) for _ in range(B)) if len(set(yy[i])) > 1]
        widths.append(np.percentile(b, 97.5) - np.percentile(b, 2.5))
    return np.mean(widths)
out.append("Expected 95% CI width of an AUC estimate (true AUC 0.75):")
out.append(f"  single 20% hold-out set (n=18): {auc_ci_width(18):.2f}")
out.append(f"  all samples via CV (n=89):      {auc_ci_width(89):.2f}\n")

# Clinical covariates vs label: are any of them predictors/confounders on their own?
out.append("Clinical covariates vs label (potential confounders):")
t = stats.ttest_ind(clin.loc[y==1, "Age"], clin.loc[y==0, "Age"])
out.append(f"  Age    mean R={clin.loc[y==1,'Age'].mean():+.2f} NR={clin.loc[y==0,'Age'].mean():+.2f}  t-test p={t.pvalue:.3f}  AUC={roc_auc_score(y, clin['Age']):.3f}")
for c in ["Sex", "COPD", "Diabetes"]:
    ct = pd.crosstab(clin[c], y)
    p = stats.fisher_exact(ct.values)[1]
    out.append(f"  {c:8s} rate R={clin.loc[y==1,c].mean():.2f} NR={clin.loc[y==0,c].mean():.2f}  Fisher p={p:.3f}  AUC={roc_auc_score(y, clin[c]):.3f}")

open("results/step01_design.txt", "w").write("\n".join(out)); print("\n".join(out))
