# Baseline blood gene expression and response to the Ornish programme: findings

Cohort: 89 participants of the Ornish lifestyle intervention (GEO GSE66175), one baseline microarray each,
13,515 genes plus Age, Sex, COPD, Diabetes. Outcome: Responder (weight loss > 10%, 54) vs Non-responder (35).
Question: does baseline expression predict who will respond. Full record: `OMISH.ipynb`; decisions: `DECISIONS.md`.

## Conclusions

1. **No confirmed biomarker.** No gene, and no gene module, predicts response at a level distinguishable from chance
   in this cohort. Held-out test set (18 participants, scored once): ROC-AUC 0.51 to 0.52 for the models, 0.43 for the
   lead module alone; all bootstrap intervals span 0.2 to 0.8. Permutation test on the training set: p about 0.08.
   Across all 13,515 genes, none passes FDR 0.05.

2. **A weak whole-profile signal exists in the training data but did not transfer.** Cross-validated AUC 0.61 on the
   71 training participants, above chance on every fold assignment, yet inside the range that shuffled labels reach
   (95th percentile 0.64). At the operating point the group requires, 90% recall, the best model's specificity is 0.19
   in cross-validation and 0.00 on the test set: it flags essentially everyone.

3. **One lead, ME9, for a larger cohort.** A 270-gene co-expression module, higher at baseline in future Responders
   (eigengene shift +0.67 SD, 98% of member genes in the same direction, 133 of 270 nominally significant against 14
   expected). Enriched for T-cell signalling (CD4, CD3E, ZAP70, IL32) with metabolic members (DHCR24, GPI, MICU1). It
   cleared stability selection only under the loosened false-selection budget of 2 (frequency 0.77) and did not
   generalise to the 18 test participants. Treat as a hypothesis, not a result.

4. **ME6 is a likely artefact.** 38 genes, neural cadherins (CDH9, CDH12) in a blood profile, small module. Not
   worth pursuing.

5. **The published MEvA-X 9-gene set is not reproduced.** Its named genes (AP2B1, RAC3, TMEM33, LINC00588, BIK) fall
   in a different module, in no module, or below the variance filter. Its reported AUC above 0.75 (10-fold CV, no
   hold-out) is within what our permutation null produces on shuffled labels (maximum 0.82). Two methods on the same
   89 people agree on almost nothing, which is what a cohort this size produces.

## Data-quality findings that matter for any reanalysis

- The table mixes two array batches (63 and 26 samples) that differ in per-gene spread by a factor of 1.8 and were
  already mean-aligned by the data providers. ComBat was applied. Batch is not associated with response.
- In the 26-sample batch the sex genes XIST and RPS4Y1 carry no sex information (agreement with the Sex column 65%
  vs 98% in the other batch). Those arrays measure at least some genes differently. Cause unknown.
- One sample (GSM1123370, participant cv000777) is labelled female with a male expression profile.

## Why the answer is "not established" rather than "no"

At n = 71 training samples with 25 features, random labels reach AUC 0.63 in one run out of twenty. A true effect of
AUC 0.6 to 0.65, which is what the training data suggest, cannot be separated from that noise here. Detecting it
with confidence would need on the order of 300 to 400 participants, or an independent cohort of similar size to the
present one to test the ME9 hypothesis directly. The GEO series holds 3-month and 1-year arrays for these
participants and a control arm; those could test whether ME9 changes with weight loss, which is a different and
cheaper question.

## Method in one paragraph

Hold-out split (18 of 89, stratified on response and batch) before any supervised step. ComBat batch correction and
WGCNA (signed network, beta 12, 5,000 most variable genes, 21 modules) fitted without the label. Supervised work on
21 module eigengenes plus 4 clinical variables, 71 training samples: repeated stratified nested cross-validation;
ridge, lasso, elastic net, ANOVA filter + ridge, XGBoost compared on identical folds; stability selection over 500
half-subsamples with the Meinshausen-Buhlmann false-selection bound; 500-permutation test of the full pipeline;
moderated t-tests and Enrichr over-representation within lead modules against the WGCNA background; final single
scoring of the test set with thresholds fixed on training data. Error rule revised mid-project to recall-first
(missed Responder costs more than a false alarm); all lists labelled screening, not confirmatory.
