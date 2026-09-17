"""Step 8: inside the lead modules (ME9 from the screening list, ME6 runner-up). Training set only, ComBat-corrected data.
Per gene: module membership (kME), log fold change Responders - Non_responders, limma-style moderated t-test, BH FDR.
Outputs: results/step08_genes_ME*.csv, step08_modules.txt, step08_eigengenes.png, step08_volcano.png"""
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats, optimize, special
from split import ids

MODULES = [9, 6]
tr = ids("train")
df = pd.read_csv("Ornish.csv", index_col=0).loc[tr]
y = (df["WeightLoss"] == "Responders").astype(int)
G = pd.read_parquet("results/step03_corrected.parquet").loc[tr]
mods = pd.read_csv("results/step03_modules.csv", index_col=0)
E = pd.read_csv("results/step03_eigengenes.csv", index_col=0).loc[tr]

def moderated_t(X, y):
    """limma-style: per-gene pooled variance s2 shrunk toward a prior s0^2 with d0 prior df (Smyth 2004, method of moments)."""
    a, b = X[y == 1], X[y == 0]; n1, n2 = len(a), len(b); d = n1 + n2 - 2
    diff = a.mean() - b.mean()
    s2 = ((a.var(ddof=1) * (n1 - 1)) + (b.var(ddof=1) * (n2 - 1))) / d
    z = np.log(s2); e = z.mean(); v = z.var(ddof=1)
    # solve trigamma(d0/2) = v - trigamma(d/2) for d0; if v too small, d0 -> inf (all genes share one variance)
    target = v - special.polygamma(1, d / 2)
    d0 = optimize.brentq(lambda x: special.polygamma(1, x / 2) - target, 1e-3, 1e6) if target > 1e-6 else 1e6
    s0 = np.exp(e - special.digamma(d / 2) + special.digamma(d0 / 2) - np.log(d0 / d))  # prior variance
    s2_mod = (d0 * s0 + d * s2) / (d0 + d)
    t = diff / np.sqrt(s2_mod * (1 / n1 + 1 / n2))
    p = 2 * stats.t.sf(np.abs(t), d0 + d)
    return diff, t, p, d0

def bh(p):
    p = np.asarray(p); o = np.argsort(p); q = np.empty_like(p); r = np.arange(1, len(p) + 1)
    q[o] = np.minimum.accumulate((p[o] * len(p) / r)[::-1])[::-1]; return np.minimum(q, 1)

# whole-array statistics first so that FDR is over all 13,515 genes, not just the module
lfc, t, p, d0 = moderated_t(G, y.values)
allg = pd.DataFrame({"logFC": lfc, "mod_t": t, "p": p, "fdr_all": bh(p)}, index=G.columns)
out = [f"TRAIN SET ONLY: {len(y)} samples. Moderated t-test prior df d0={d0:.1f} (larger = more shrinkage toward a common variance).",
       f"genes with FDR<0.05 across all {G.shape[1]} genes: {(allg.fdr_all < 0.05).sum()}; FDR<0.25: {(allg.fdr_all < 0.25).sum()}", ""]
fig, axes = plt.subplots(1, len(MODULES), figsize=(4 * len(MODULES), 3.8))
for ax, m in zip(axes, MODULES):
    genes = mods[mods.module == m].index
    tab = allg.loc[genes].assign(kME=mods.loc[genes, "kME"], fdr_in_module=bh(allg.loc[genes, "p"])).sort_values("kME", ascending=False)
    tab.to_csv(f"results/step08_genes_ME{m}.csv")
    e = E[f"ME{m}"]; auc = stats.mannwhitneyu(e[y == 1], e[y == 0]).statistic / ((y == 1).sum() * (y == 0).sum())
    out += [f"ME{m}: {len(genes)} genes. Eigengene Responders - Non_responders = {e[y==1].mean() - e[y==0].mean():+.2f} SD units, AUC {auc:.2f}, Mann-Whitney p={stats.mannwhitneyu(e[y==1], e[y==0]).pvalue:.3f}",
            f"  genes with p<0.05 (unadjusted): {(tab.p < 0.05).sum()} of {len(tab)} (expected by chance ~{0.05*len(tab):.0f}); FDR<0.25 within module: {(tab.fdr_in_module < 0.25).sum()}",
            f"  direction: {(tab.logFC > 0).mean():.0%} of genes higher in Responders",
            "  top hub genes (kME): " + ", ".join(f"{g} (kME {r.kME:.2f}, logFC {r.logFC:+.2f}, p {r.p:.3f})" for g, r in tab.head(8).iterrows()),
            "  most differential genes (|t|): " + ", ".join(f"{g} (logFC {r.logFC:+.2f}, p {r.p:.4f})" for g, r in tab.reindex(tab.mod_t.abs().sort_values(ascending=False).index).head(8).iterrows()), ""]
    ax.boxplot([e[y == 0], e[y == 1]], tick_labels=["Non_resp", "Resp"], widths=0.5)
    ax.scatter(np.random.default_rng(0).normal(1, 0.05, (y == 0).sum()), e[y == 0], s=12, alpha=0.6)
    ax.scatter(np.random.default_rng(1).normal(2, 0.05, (y == 1).sum()), e[y == 1], s=12, alpha=0.6)
    ax.set(title=f"ME{m} eigengene (AUC {auc:.2f})", ylabel="eigengene (SD units)")
fig.tight_layout(); fig.savefig("results/step08_eigengenes.png", dpi=120); plt.close(fig)

fig, ax = plt.subplots(figsize=(6, 4.5))
ax.scatter(allg.logFC, -np.log10(allg.p), s=4, c="lightgrey", label="all genes")
for m, c in zip(MODULES, ("C3", "C0")):
    g = mods[mods.module == m].index; ax.scatter(allg.loc[g, "logFC"], -np.log10(allg.loc[g, "p"]), s=8, c=c, label=f"ME{m}")
ax.axhline(-np.log10(0.05), ls="--", c="grey", lw=1); ax.set(xlabel="log fold change (Responders - Non_responders)", ylabel="-log10 p (moderated t)", title="Per-gene differential expression, training set"); ax.legend()
fig.tight_layout(); fig.savefig("results/step08_volcano.png", dpi=120)
open("results/step08_modules.txt", "w").write("\n".join(out)); print("\n".join(out))
