"""Step 2: EDA. Per-sample expression distributions, PCA coloured by response and by
each clinical covariate, sex-gene check. Unsupervised only: nothing here uses the label
to change the data. Outputs: results/step02_*.png, results/step02_eda.txt"""
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy import stats

df = pd.read_csv("Ornish.csv", index_col=0)
y = (df.pop("WeightLoss") == "Responders").astype(int)
clin = df[["Age", "Sex", "COPD", "Diabetes"]]
genes = df.drop(columns=clin.columns)
out = []

# 1. per-sample distribution: are arrays normalised to the same scale?
q = genes.quantile([0.05, 0.5, 0.95], axis=1).T
out.append("per-sample median expression: min %.2f  max %.2f  (spread %.2f)" % (q[0.5].min(), q[0.5].max(), q[0.5].max() - q[0.5].min()))
fig, ax = plt.subplots(figsize=(10, 4))
for i in range(len(genes)):
    ax.hist(genes.iloc[i], bins=100, histtype="step", alpha=0.3, color="C1" if y.iloc[i] else "C0")
ax.set(xlabel="log expression", ylabel="genes", title="Per-sample expression distributions (orange=Responders)")
fig.tight_layout(); fig.savefig("results/step02_distributions.png", dpi=120); plt.close(fig)

# 2. sex-gene check: XIST female, Y-linked genes male
sexg = genes[["XIST", "RPS4Y1", "DDX3Y", "KDM5D"]]
pred_male = (sexg["RPS4Y1"] > sexg["XIST"]).astype(int)
for lab in (0, 1):
    out.append(f"Sex=={lab}: n={int((clin.Sex==lab).sum())}  XIST mean={sexg.XIST[clin.Sex==lab].mean():.2f}  RPS4Y1 mean={sexg.RPS4Y1[clin.Sex==lab].mean():.2f}")
agree = (pred_male == clin.Sex).mean()
out.append(f"fraction agreeing with Sex column if 1=male: {agree:.2f}  (if 1=female: {1-agree:.2f})")
mism = clin.index[(pred_male != clin.Sex) if agree > 0.5 else (pred_male == clin.Sex)]
out.append(f"samples whose sex genes disagree with Sex column: {list(mism)}")
fig, ax = plt.subplots(figsize=(5, 4))
ax.scatter(sexg.XIST, sexg.RPS4Y1, c=clin.Sex, cmap="coolwarm"); ax.set(xlabel="XIST", ylabel="RPS4Y1", title="Sex-gene check (colour = Sex column)")
fig.tight_layout(); fig.savefig("results/step02_sexcheck.png", dpi=120); plt.close(fig)

# 3. PCA on standardised genes (unsupervised; label only used for colouring)
Z = StandardScaler().fit_transform(genes)
pca = PCA(n_components=10, random_state=0).fit(Z)
pcs = pca.transform(Z)
evr = pca.explained_variance_ratio_ * 100
out.append("variance explained by PC1..PC5: " + ", ".join(f"{v:.1f}%" for v in evr[:5]))
# does any PC separate the label or a covariate? AUC of PC vs binary variable, |r| vs Age
out.append("\nPC vs variable (AUC for binary, |Spearman r| for Age); values near 0.5 / 0 mean no relation:")
from sklearn.metrics import roc_auc_score
hdr = f"{'':5s}" + "".join(f"{c:>10s}" for c in ["Response", "Sex", "COPD", "Diabetes", "Age"])
out.append(hdr)
for k in range(5):
    row = [max(roc_auc_score(v, pcs[:, k]), 1 - roc_auc_score(v, pcs[:, k])) for v in (y, clin.Sex, clin.COPD, clin.Diabetes)]
    row.append(abs(stats.spearmanr(clin.Age, pcs[:, k])[0]))
    out.append(f"PC{k+1:<3d}" + "".join(f"{r:10.2f}" for r in row))

# 4. batch: sample IDs carry two GEO series prefixes (GSM1123xxx / GSM1616xxx)
batch = (genes.index.str[:7] == "GSM1616").astype(int)
out.append(f"\nGEO series prefix as batch: GSM1123={int((batch==0).sum())}  GSM1616={int((batch==1).sum())}")
out.append("batch x response:\n" + pd.crosstab(batch, y, rownames=["GSM1616"], colnames=["Responder"]).to_string())
out.append("PC1..PC5 vs batch AUC: " + ", ".join(f"{max(roc_auc_score(batch, pcs[:, k]), 1-roc_auc_score(batch, pcs[:, k])):.2f}" for k in range(5)))

fig2, ax2 = plt.subplots(figsize=(5, 3)); ax2.bar(range(1, 11), evr); ax2.set(title="Variance explained", xlabel="PC", ylabel="%")
fig2.tight_layout(); fig2.savefig("results/step02_pca_variance.png", dpi=120); plt.close(fig2)
fig, axes = plt.subplots(2, 3, figsize=(15, 9))
colours = {"Response": y, "Sex": clin.Sex, "COPD": clin.COPD, "Diabetes": clin.Diabetes, "Age": clin.Age, "Batch (GEO series)": batch}
for ax, (name, c) in zip(axes.flat, colours.items()):
    sc = ax.scatter(pcs[:, 0], pcs[:, 1], c=c, cmap="viridis" if name == "Age" else "coolwarm", s=40)
    ax.set(title=f"PC1 vs PC2 coloured by {name}", xlabel=f"PC1 ({evr[0]:.1f}%)", ylabel=f"PC2 ({evr[1]:.1f}%)")
    if name == "Age": fig.colorbar(sc, ax=ax)
fig.tight_layout(); fig.savefig("results/step02_pca.png", dpi=120); plt.close(fig)


# 4b. batch structure: spread along PCs, sex-gene cluster membership, and sex agreement per batch
p = pd.DataFrame(pcs[:, :2], index=genes.index, columns=["PC1", "PC2"])
out.append("PC spread (SD) by batch:\n" + p.groupby(batch).std().round(1).rename(index={0: "GSM1123", 1: "GSM1616"}).to_string())
mid = genes.XIST.between(2.5, 4.2) & genes.RPS4Y1.between(4.5, 7)
out.append("intermediate sex-gene cluster by batch:\n" + pd.crosstab(mid, batch, rownames=["intermediate"], colnames=["GSM1616"]).to_string())
for b in (0, 1):
    m = batch == b
    out.append(f"sex-gene agreement with Sex column, batch {'GSM1616' if b else 'GSM1123'}: {(pred_male[m] == clin.Sex[m]).mean():.2f}")

# 5. outlier samples: distance from centre in PC space
d = np.sqrt((pcs[:, :5] ** 2).sum(1)); z = (d - d.mean()) / d.std()
out.append(f"\nsamples > 2.5 SD from centre in PC1-5: {list(genes.index[z > 2.5])}")


# 6. PCs 1-10: consecutive pairs, coloured by response (top row) and batch (bottom row)
fig, axes = plt.subplots(2, 5, figsize=(20, 8))
for c, (a, b) in enumerate([(0, 1), (2, 3), (4, 5), (6, 7), (8, 9)]):
    for r, (name, col) in enumerate([("Response", y), ("Batch", batch)]):
        ax = axes[r, c]
        ax.scatter(pcs[:, a], pcs[:, b], c=col, cmap="coolwarm", s=30)
        ax.set(xlabel=f"PC{a+1} ({evr[a]:.1f}%)", ylabel=f"PC{b+1} ({evr[b]:.1f}%)", title=f"PC{a+1} vs PC{b+1} by {name}")
fig.tight_layout(); fig.savefig("results/step02_pca_1to10.png", dpi=110); plt.close(fig)
out.append("\nPC1..PC10 AUC vs response / batch:")
for k in range(10):
    ar, ab = roc_auc_score(y, pcs[:, k]), roc_auc_score(batch, pcs[:, k])
    out.append(f"  PC{k+1:<3d} var {evr[k]:4.1f}%  response {max(ar,1-ar):.2f}  batch {max(ab,1-ab):.2f}")

open("results/step02_eda.txt", "w").write("\n".join(out)); print("\n".join(out))
