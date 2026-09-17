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

## Train / test split
`split.py` sets aside 18 of the 89 samples (20%, stratified on response and batch, fixed seed) as a test set in
`results/split.csv`. Every supervised step (4 onward) uses only the 71 training samples, with cross-validation inside
them. The test set is scored once, in the write-up step. ComBat and WGCNA are unsupervised and stay fitted on all 89.

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
| 1 | 01 Experiment design | Positive class, cost of each error type, primary metric, baseline, honest evaluation scheme for n=89. Revised 2026-09-17: recall first (a missed Responder or module costs more than a false alarm); classifiers reported at 90% recall; stability budget 2 expected false selections |
| 2 | 02 EDA | Class balance, expression distributions per sample (normalisation check), clinical covariates vs label, PCA, batch/confounder hunt, leakage hunt. Sanity check: sex-chromosome genes (XIST, RPS4Y1) must agree with the Sex column |
| 3 | **ComBat + WGCNA gene modules** | ComBat batch correction (`combat.py`, parametric empirical Bayes, no covariates), then Weighted Correlation Network Analysis on the genes (unsupervised, label never used): soft-threshold power, topological overlap, hierarchical clustering, dynamic tree cut. Each module is summarised by its representative (module eigengene, the module's first principal component). Genes go from 13,515 to a few tens of module representatives |
| 4 | 09 + 01/03/05 Pipelines, nested CV, baselines | Hold-out split; then on the training set: `Pipeline` over eigengenes + clinical, repeated stratified CV with C tuned in an inner loop; chance and clinical-only baselines; ROC-AUC / PR-AUC with spread across repeats and bootstrap CI |
| 5 | 04/06 Models + feature selection (scikit-learn) | Univariate filter + logistic, L1/L2 logistic, XGBoost on the module representatives; compared on identical folds |
| 6 | **Stability selection** (Meinshausen & Bühlmann 2010) | Subsample 50% of samples a few hundred times, fit the L1 model on module representatives, record selection frequency per module. Keep modules above a threshold chosen to bound expected false positives |
| 7 | **Permutation test** | Rerun the whole pipeline on shuffled labels ~200 times. The real AUC must sit outside the null distribution |
| 8 | **Inside the selected modules** | For each selected module: member genes, hub genes (highest module membership), per-gene fold change and moderated t-test between classes, so biologists can read the module without the model |
| 9 | Interpretation | Bootstrap ridge coefficients (SHAP dropped: linear model, coefficients are the explanation); gene-set enrichment per lead module vs the WGCNA background; comparison with genes reported in the MEvA-X paper |
| 10 | 08 Write-up | Single scoring of the 18-sample test set; conclusions in REPORT.md |
| 11 | 10 Structure | Refactor surviving scripts into a small package if the group wants to reuse it |
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
