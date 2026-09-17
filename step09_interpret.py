"""Step 9: interpretation. (a) Bootstrap ridge coefficients of the eigengene model (replaces SHAP: for a linear model on
standardised features SHAP = coefficient x deviation, so the coefficients are the explanation). (b) Gene-set enrichment
of ME9 and ME6 via Enrichr, background = the 5,000 genes WGCNA saw. Training set only.
Outputs: results/step09_coefficients.csv/.png, step09_enrichment_ME*.csv, step09_interpret.txt"""
import numpy as np, pandas as pd, matplotlib, warnings
matplotlib.use("Agg"); warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from split import ids

B, SEED, C = 1000, 0, 0.1        # C=0.1: strong ridge, in the range step 4's inner CV chose
tr = ids("train")
df = pd.read_csv("Ornish.csv", index_col=0).loc[tr]
y = (df["WeightLoss"] == "Responders").astype(int).values
E = pd.read_csv("results/step03_eigengenes.csv", index_col=0).loc[tr]
X = pd.concat([E, df[["Age", "Sex", "COPD", "Diabetes"]]], axis=1)
mods = pd.read_csv("results/step03_modules.csv", index_col=0)
out = [f"TRAIN SET ONLY: {len(y)} samples."]

# (a) coefficients on standardised features, bootstrap over samples
rng = np.random.default_rng(SEED)
def fit(idx):
    Xs = StandardScaler().fit_transform(X.values[idx]); return LogisticRegression(C=C, max_iter=5000).fit(Xs, y[idx]).coef_[0]
full = fit(np.arange(len(y)))
boot = np.array([fit(rng.integers(0, len(y), len(y))) for _ in range(B)])
coef = pd.DataFrame({"coef": full, "lo": np.percentile(boot, 2.5, 0), "hi": np.percentile(boot, 97.5, 0),
                     "P(coef>0)": (boot > 0).mean(0)}, index=X.columns).sort_values("coef", key=abs, ascending=False)
coef.to_csv("results/step09_coefficients.csv")
out += ["", f"Ridge (C={C}) coefficients per SD of feature, log-odds of Responder; 95% bootstrap interval; P(coef>0):", coef.round(2).head(10).to_string(),
        "features whose interval excludes 0: " + (", ".join(coef.index[(coef.lo > 0) | (coef.hi < 0)]) or "none")]
fig, ax = plt.subplots(figsize=(6, 6)); c = coef.iloc[::-1]
ax.errorbar(c.coef, range(len(c)), xerr=[c.coef - c.lo, c.hi - c.coef], fmt="o", ms=4, capsize=2)
ax.axvline(0, c="grey", lw=1); ax.set(yticks=range(len(c)), yticklabels=c.index, xlabel="log-odds per SD (95% bootstrap)", title="Ridge coefficients, eigengenes + clinical")
fig.tight_layout(); fig.savefig("results/step09_coefficients.png", dpi=120)

# (b) enrichment: Enrichr over-representation with the WGCNA input genes as background
import gseapy as gp
background = list(mods.index)
libs = ["MSigDB_Hallmark_2020", "GO_Biological_Process_2023", "KEGG_2021_Human", "Reactome_2022"]
for m in (9, 6):
    genes = list(mods[mods.module == m].index.str.replace(r"\.\d+$", "", regex=True).unique())
    try:
        res = gp.enrich(gene_list=genes, gene_sets=libs, background=background, outdir=None).results
        res = res.sort_values("Adjusted P-value"); res.to_csv(f"results/step09_enrichment_ME{m}.csv", index=False)
        sig = res[res["Adjusted P-value"] < 0.1]
        out += ["", f"ME{m} ({len(genes)} genes) enrichment vs {len(background)}-gene background; terms with adjusted p<0.1: {len(sig)}",
                sig.head(12)[["Gene_set", "Term", "Genes", "Adjusted P-value"]].to_string(index=False) if len(sig) else "  none; top 5 by raw p: " + "; ".join(f"{r.Term} [{r.Genes}] adj p {r['Adjusted P-value']:.2f}" for _, r in res.head(5).iterrows())]
    except Exception as e:
        out += ["", f"ME{m}: enrichment failed ({e}); gene list saved for manual run"]
    pd.Series(genes).to_csv(f"results/step09_genes_ME{m}.txt", index=False, header=False)
open("results/step09_interpret.txt", "w").write("\n".join(out)); print("\n".join(out))

# (c) comparison with the MEvA-X paper (Bioinformatics 2023): their best Ornish model used 9 genes, hubs AP2B1, RAC3, TMEM33,
# plus LINC00588, BIK (others not listed in the abstract-level text); AUC > 0.75 under 10-fold CV without a hold-out set.
meva = ["AP2B1", "RAC3", "TMEM33", "LINC00588", "BIK"]
allg = pd.read_parquet("results/step03_corrected.parquet").columns
rows = []
for g in meva:
    hits = [c for c in allg if c == g or c.startswith(g + ".")]
    rows.append(f"  {g}: " + (", ".join(f"{h} -> module {'ME%d' % mods.loc[h, 'module'] if h in mods.index and mods.loc[h, 'module'] else ('unassigned' if h in mods.index else 'not in top-5000 variance')}" for h in hits) if hits else "not in table"))
out += ["", "MEvA-X paper genes (named in the paper) located in our module map:"] + rows
open("results/step09_interpret.txt", "w").write("\n".join(out)); print("\n".join(out[-7:]))
