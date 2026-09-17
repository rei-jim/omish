# OMISH — gene biomarkers for weight-loss response (Ornish dataset)

## What this repo does
Find gene-expression biomarkers that separate **Responders** (weight loss > 10%) from
**Non_responders** in the Ornish cohort, following the data-science course in
`C:\Users\ReiJiménez\data-science\lessons` step by step, so the domain experts
(biology / omics) can judge every decision and result before we move on.

Source problem: https://github.com/sib-swiss/feature-selection-training/blob/main/notebooks/project_Ornish.ipynb
Data origin: MEvA-X paper, https://doi.org/10.1093/bioinformatics/btad384

## The data (`Ornish.csv`)
| item | value |
|---|---|
| samples | 89 (54 Responders, 35 Non_responders) |
| features | 13,515 genes (microarray, log-scale, range 0.16–9.99) + Age (z-scored), Sex, COPD, Diabetes (0/1) |
| missing values | none |
| target | `WeightLoss` (binary) |
| sample metadata | GEO GSE66175 (`GSE66175_samples.txt`, joined in `results/sample_metadata.csv`) |

From the GEO metadata: all 89 samples are **baseline** arrays, **one per participant**, all from the **intervention
arm** (no controls). The two GEO sub-series (GSM1123xxx, 63 samples; GSM1616xxx, 26 samples) are two array batches of
the same study. So the question is well posed: does baseline expression predict who will respond to the Ornish
programme. No repeated measures, so plain stratified CV over samples is valid. The full GEO series also holds 3-month
and 1-year arrays and a control arm that are not in our table; they could serve later for validation or for
"change over time" analyses.

The defining constraint is **p >> n**: 13,519 features for 89 samples. Any procedure
that looks at the label (feature ranking, selection, tuning) must run inside
cross-validation folds or the results are optimistic and non-reproducible.

## How we work
One step at a time. For each step we produce:
1. **What we do** — the procedure, in plain terms.
2. **Why it applies here** — how the generic lesson maps to a small-n, high-p omics cohort.
3. **Consequences** — what the outcome does and does not allow us to conclude, and what it forces downstream.
4. **Checkpoint** — results are reviewed by the expert group; the group decides how to continue. Nothing proceeds automatically.

Each step lives in its own script `stepNN_<name>.py`, runnable on its own, writing
outputs to `results/`. Decisions taken at each checkpoint are logged in `DECISIONS.md`.

## Step plan (lesson → this project)
| step | lesson | what we settle here |
|---|---|---|
| 1 | 01 Experiment design | Positive class, cost of each error type, primary metric, baseline, honest evaluation scheme for n=89 |
| 2 | 02 EDA | Class balance, expression distributions per sample (normalisation check), clinical covariates vs label, PCA, batch/confounder hunt, leakage hunt. Sanity check: sex-chromosome genes (XIST, RPS4Y1) must agree with the Sex column |
| 3 | **ComBat + WGCNA gene modules** | ComBat batch correction (`combat.py`, parametric empirical Bayes, no covariates), then Weighted Correlation Network Analysis on the genes (unsupervised, label never used): soft-threshold power, topological overlap, hierarchical clustering, dynamic tree cut. Each module is summarised by its representative (module eigengene, the module's first principal component). Genes go from 13,515 to a few tens of module representatives |
| 4 | 09 Pipelines | Everything supervised inside a `Pipeline` over module representatives + clinical covariates; repeated stratified CV with hyperparameters tuned in an inner loop (nested CV) |
| 5 | 01/03/05 Baseline + metrics | Chance model and clinical-only model; ROC-AUC / PR-AUC / F1 with bootstrap confidence intervals |
| 6 | 04/06 Models + feature selection (scikit-learn) | Univariate filter + logistic, L1/L2 logistic, XGBoost on the module representatives; compared on identical folds |
| 7 | **Stability selection** (Meinshausen & Bühlmann 2010) | Subsample 50% of samples a few hundred times, fit the L1 model on module representatives, record selection frequency per module. Keep modules above a threshold chosen to bound expected false positives |
| 8 | **Permutation test** | Rerun the whole pipeline on shuffled labels ~200 times. The real AUC must sit outside the null distribution |
| 9 | **Inside the selected modules** | For each selected module: member genes, hub genes (highest module membership), per-gene fold change and moderated t-test between classes, so biologists can read the module without the model |
| 10 | 07 SHAP | Direction and size of each module's contribution to the prediction |
| 11 | **Biological plausibility** | Gene-set enrichment per selected module; comparison with genes reported in the MEvA-X paper |
| 12 | 08 Write-up | Findings, limits (n=89, no external cohort, batch), what is needed to validate |
| 13 | 10 Structure | Refactor surviving scripts into a small package if the group wants to reuse it |
Steps may be merged, repeated, or dropped at any checkpoint.

### Why WGCNA before feature selection
Co-expressed genes are interchangeable to a sparse model, so selecting single genes out of 13,515 is unstable and
biologically misleading. WGCNA groups genes into modules of co-expression first, without looking at the label, and
feature selection then runs on module representatives. Fewer, less correlated features, and each selected feature
is a biological programme rather than an arbitrary member of one. WGCNA is unsupervised, so fitting it on all samples
is not label leakage. Batch (step 2) must be handled before WGCNA: correlations driven by batch would form their own modules.

## Reproducible notebook
`OMISH.ipynb` is the record of the whole analysis: one section per step (what / why / consequences / decision),
each running its `stepNN_*.py` with `%run`. Rerun top to bottom to reproduce everything. Seeds are fixed in every script.
