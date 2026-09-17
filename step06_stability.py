"""Step 6: stability selection (Meinshausen & Buhlmann 2010) on eigengenes + clinical, training set only.
Two selection engines, since step 5 showed hard lasso loses the signal:
  (a) L1 logistic at a mild penalty, (b) ANOVA F-filter keeping the top-k features.
B random half-subsamples each; selection frequency per feature; threshold from the MB bound on expected false positives.
Outputs: results/step06_stability.csv, step06_stability.png, step06_stability.txt"""
import numpy as np, pandas as pd, matplotlib, warnings
matplotlib.use("Agg"); warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import f_classif
from split import ids

B, SEED, EV_TARGETS = 500, 0, (2,)          # subsamples, seed, tolerated expected false selections; budget 2 per DECISIONS step 6 (sensitivity-first: rather a false alarm than a missed real biomarker)
C_L1, K_FILTER = 0.3, 5                     # mild lasso; filter size ~ what step 5's inner CV picked
df = pd.read_csv("Ornish.csv", index_col=0).loc[ids("train")]
y = (df["WeightLoss"] == "Responders").astype(int).values
E = pd.read_csv("results/step03_eigengenes.csv", index_col=0).loc[df.index]
X = pd.concat([E, df[["Age", "Sex", "COPD", "Diabetes"]]], axis=1)
p, n = X.shape[1], len(y)
rng = np.random.default_rng(SEED)

freq = pd.DataFrame(0.0, index=X.columns, columns=["L1 (C=%.1f)" % C_L1, f"ANOVA top-{K_FILTER}"])
q = np.zeros(2)                             # average number of features selected per subsample, per engine
for _ in range(B):
    idx = np.concatenate([rng.choice(np.where(y == c)[0], size=(y == c).sum() // 2, replace=False) for c in (0, 1)])  # stratified half
    Xs, ys = StandardScaler().fit_transform(X.values[idx]), y[idx]
    sel = LogisticRegression(penalty="l1", C=C_L1, solver="liblinear", max_iter=5000).fit(Xs, ys).coef_[0] != 0
    freq.iloc[:, 0] += sel; q[0] += sel.sum()
    F = f_classif(Xs, ys)[0]; top = np.argsort(-F)[:K_FILTER]
    freq.iloc[top, 1] += 1; q[1] += K_FILTER
freq /= B; q /= B
# MB bound: E[false selections] <= q^2 / ((2*pi - 1) * p)  ->  pi_thr = (q^2 / (p * E_V) + 1) / 2
out = [f"TRAIN SET ONLY: {n} samples, {p} features, {B} stratified half-subsamples per engine.",
       "EXPLORATORY, sensitivity-first: lists below tolerate more false positives to avoid missing a real biomarker (cancer-screening logic).", ""]
thr = {ev: np.clip((q ** 2 / (p * ev) + 1) / 2, 0.5, 1.0) for ev in EV_TARGETS}  # per-budget thresholds
for j, col in enumerate(freq.columns):
    out.append(f"{col}: mean #selected per subsample q={q[j]:.1f}")
    for ev in EV_TARGETS:
        keep = freq[col][freq[col] >= thr[ev][j]].sort_values(ascending=False)
        out.append(f"  E[false selections]<={ev:g}: pi>={thr[ev][j]:.2f}  ->  " + (", ".join(f"{f} ({v:.2f})" for f, v in keep.items()) if len(keep) else "none"))
out += ["", "selection frequency, all features:", freq.sort_values(freq.columns[0], ascending=False).round(2).to_string()]
freq.to_csv("results/step06_stability.csv")
open("results/step06_stability.txt", "w").write("\n".join(out)); print("\n".join(out))

order = freq.mean(1).sort_values(ascending=False).index
fig, ax = plt.subplots(figsize=(10, 4))
w = 0.4; xpos = np.arange(p)
ev_plot = max(EV_TARGETS)
for j, col in enumerate(freq.columns):
    ax.bar(xpos + (j - 0.5) * w, freq.loc[order, col], w, label=col)
    ax.axhline(thr[ev_plot][j], ls="--", c=f"C{j}", lw=1)
ax.set(xticks=xpos, ylabel="selection frequency", title=f"Stability selection (dashed = threshold for <={ev_plot} expected false selections, exploratory)")
ax.set_xticklabels(order, rotation=90); ax.legend(); fig.tight_layout(); fig.savefig("results/step06_stability.png", dpi=120)
