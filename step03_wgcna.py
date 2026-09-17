"""Step 3: batch correction + WGCNA. Unsupervised: the label is never read.
Outputs: results/step03_corrected.parquet (batch-corrected genes, all 13,515),
         results/step03_modules.csv (gene -> module, kME), results/step03_eigengenes.csv,
         results/step03_*.png, results/step03_wgcna.txt"""
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.stats import linregress

N_GENES, MIN_MODULE, MAX_MODULE, MERGE_CUT, BETA, SEED = 5000, 30, 500, 0.25, 12, 0
rng = np.random.default_rng(SEED)
df = pd.read_csv("Ornish.csv", index_col=0).drop(columns="WeightLoss")
clin = df[["Age", "Sex", "COPD", "Diabetes"]]
genes = df.drop(columns=clin.columns)
batch = pd.Series(np.where(genes.index.str.startswith("GSM1616"), "GSM1616", "GSM1123"), index=genes.index)
out = []

# 1. batch correction with ComBat (combat.py; parametric empirical Bayes, no covariates, label not used).
# The table arrived with per-batch gene means already equal, so ComBat here mainly equalises the per-gene
# spread: the later batch (GSM1616) is ~1.8x noisier than the earlier one.
from combat import combat
corr = combat(genes, batch)
corr.to_parquet("results/step03_corrected.parquet")
sd_ratio = (corr[batch == "GSM1616"].std() / corr[batch == "GSM1123"].std()).median()
out.append(f"ComBat-corrected {corr.shape[1]} genes across {corr.shape[0]} samples; median SD ratio later/earlier batch: raw {(genes[batch == 'GSM1616'].std() / genes[batch == 'GSM1123'].std()).median():.2f} -> corrected {sd_ratio:.2f}")

# 2. unsupervised gene filter: top-N by variance of the corrected data
X = corr[corr.var().sort_values(ascending=False).index[:N_GENES]]
out.append(f"WGCNA on top {N_GENES} variance genes")

# 3. soft-threshold power: smallest beta where the network is scale-free (R^2 >= 0.8), signed network
C = np.corrcoef(X.values.T).astype(np.float32)
S = (1 + C) / 2
fits = []
for beta in range(1, 21):
    k = (S ** beta).sum(1) - 1
    hist, edges = np.histogram(k, bins=10)
    keep = hist > 0
    r = linregress(np.log10((edges[:-1] + edges[1:])[keep] / 2), np.log10(hist[keep] / hist.sum()))
    fits.append((beta, -np.sign(r.slope) * r.rvalue ** 2, k.mean()))
fits = pd.DataFrame(fits, columns=["beta", "signed_R2", "mean_k"])
# WGCNA FAQ default for a signed network with n > 40 samples is beta=12; we report the scale-free fit at that value
beta = BETA
out.append(f"soft-threshold power beta={beta} (WGCNA default for signed network, n>40); scale-free R^2={fits.loc[fits.beta==beta,'signed_R2'].item():.2f}, mean connectivity {fits.loc[fits.beta==beta,'mean_k'].item():.1f}")
out.append("smallest beta with R^2>=0.8: " + (str(int(fits[fits.signed_R2 >= 0.8].beta.iloc[0])) if (fits.signed_R2 >= 0.8).any() else "none"))
fig, ax = plt.subplots(1, 2, figsize=(9, 3.5))
ax[0].plot(fits.beta, fits.signed_R2, "o-"); ax[0].axhline(0.8, ls="--", c="grey"); ax[0].set(xlabel="beta", ylabel="scale-free fit R^2")
ax[1].plot(fits.beta, fits.mean_k, "o-"); ax[1].set(xlabel="beta", ylabel="mean connectivity", yscale="log")
fig.tight_layout(); fig.savefig("results/step03_softthreshold.png", dpi=120); plt.close(fig)

# 4. adjacency -> topological overlap matrix -> dissimilarity -> average-linkage tree
A = S ** beta; np.fill_diagonal(A, 0)
k = A.sum(1)
L = A @ A
TOM = (L + A) / (np.minimum.outer(k, k) + 1 - A); np.fill_diagonal(TOM, 1)
D = 1 - TOM
Z = linkage(D[np.triu_indices_from(D, 1)], "average")
del L, S, C

# 5. recursive tree cut (the idea behind dynamicTreeCut): cut high, then re-cut any module larger than
# MAX_MODULE on its own subtree until pieces are between MIN_MODULE and MAX_MODULE. Leftovers -> module 0.
def split(idx):
    if len(idx) <= MAX_MODULE: return [idx]
    Zs = linkage(D[np.ix_(idx, idx)][np.triu_indices(len(idx), 1)], "average")
    for q in (0.99, 0.97, 0.95, 0.9, 0.8, 0.7, 0.6, 0.5):
        l = fcluster(Zs, np.quantile(Zs[:, 2], q), "distance")
        parts = [idx[l == c] for c in np.unique(l) if (l == c).sum() >= MIN_MODULE]
        if len(parts) >= 2: return [p for part in parts for p in split(part)]
    return [idx]
lab = np.zeros(len(X.columns), int)
for m, idx in enumerate(split(np.arange(len(X.columns))), 1): lab[idx] = m

def eigengenes(Xs, lab):
    E = {}
    for m in sorted(set(lab) - {0}):
        sub = Xs[:, lab == m]
        u, s, vt = np.linalg.svd(sub - sub.mean(0), full_matrices=False)
        e = u[:, 0] * s[0]
        if np.corrcoef(e, sub.mean(1))[0, 1] < 0: e = -e
        E[m] = e / e.std()
    return pd.DataFrame(E, index=X.index)

# 6. merge modules whose eigengenes are highly correlated (1 - cor < MERGE_CUT), repeat until stable
Xs = X.values
while True:
    E = eigengenes(Xs, lab)
    if E.shape[1] < 2: break
    ce = 1 - np.corrcoef(E.values.T); np.fill_diagonal(ce, np.inf)
    i, j = np.unravel_index(ce.argmin(), ce.shape)
    if ce[i, j] > MERGE_CUT: break
    lab = np.where(lab == E.columns[j], E.columns[i], lab)
mods = {m: i + 1 for i, m in enumerate(sorted(set(lab) - {0}))}
lab = np.array([mods.get(m, 0) for m in lab])
E = eigengenes(Xs, lab); E.columns = [f"ME{m}" for m in E.columns]
kME = pd.DataFrame({g: [np.corrcoef(Xs[:, gi], E[f"ME{lab[gi]}"])[0, 1] if lab[gi] else np.nan] for gi, g in enumerate(X.columns)}).T[0]
modules = pd.DataFrame({"module": lab, "kME": kME.values}, index=X.columns).sort_values(["module", "kME"], ascending=[True, False])
modules.to_csv("results/step03_modules.csv"); E.to_csv("results/step03_eigengenes.csv")
sizes = modules.module.value_counts().sort_index()
out.append(f"modules: {int((sizes.index != 0).sum())}   unassigned genes (module 0): {int(sizes.get(0, 0))}")
out.append("module sizes: " + ", ".join(f"ME{m}={n}" for m, n in sizes.items() if m))
out.append("hub gene per module (highest kME): " + ", ".join(f"ME{m}:{modules[modules.module==m].index[0]}" for m in sizes.index if m))

# 7. diagnostics: eigengene vs batch (should be ~0 after correction), dendrogram with module colours
E_b = E.assign(batch=(batch == "GSM1616").astype(int).values)
out.append("max |cor(eigengene, batch)| after correction: %.2f" % E_b.corr().loc["batch"].drop("batch").abs().max())
fig, ax = plt.subplots(figsize=(11, 3.5))
ax.bar([f"ME{m}" if m else "none" for m in sizes.index], sizes.values); ax.set(ylabel="genes", title="Module sizes (none = unassigned)")
plt.setp(ax.get_xticklabels(), rotation=90)
fig.tight_layout(); fig.savefig("results/step03_module_sizes.png", dpi=120); plt.close(fig)
fig, ax = plt.subplots(figsize=(6, 5)); im = ax.imshow(E.corr(), cmap="coolwarm", vmin=-1, vmax=1)
ax.set(xticks=range(len(E.columns)), yticks=range(len(E.columns)), xticklabels=E.columns, yticklabels=E.columns, title="Eigengene correlations")
plt.setp(ax.get_xticklabels(), rotation=90); fig.colorbar(im); fig.tight_layout(); fig.savefig("results/step03_eigengene_cor.png", dpi=120); plt.close(fig)

open("results/step03_wgcna.txt", "w").write("\n".join(out)); print("\n".join(out))
