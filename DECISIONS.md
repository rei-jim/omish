# Decisions log

| date | step | decision | reason |
|---|---|---|---|
| 2026-09-17 | 1 | Positive class = Responders; primary metric ROC-AUC (+PR-AUC, confusion matrix); baselines = chance and clinical-only; evaluation = repeated stratified CV with nested supervised steps, no single hold-out; selection stability outranks raw score | p>>n; 18-sample hold-out gives AUC CI ~0.47 wide vs ~0.21 with CV; false-positive genes are the costly error (pending expert confirmation) |
| 2026-09-17 | plan | Feature selection runs on WGCNA module representatives, not raw genes; selection with scikit-learn | co-expressed genes are interchangeable to sparse models; modules are the biologically meaningful unit |
| 2026-09-17 | 3 | WGCNA implemented in numpy/scipy (no new dependency) on top-5000 variance genes | transparent, reproducible; 5000 genes is standard WGCNA practice and keeps the 13.5k x 13.5k matrices out of memory |
| 2026-09-17 | 2 | GEO GSE66175 metadata joined: all 89 samples are baseline, one per participant, intervention arm only; the two GSM prefixes are array batches of one study. Batch correction from step 2 stands; no grouped CV needed | metadata provided by the expert group |
| 2026-09-17 | 2/3 | Keep all 89 samples; batch-correct with ComBat (parametric EB, no covariates), implemented in combat.py because inmoose has no Windows/py3.13 wheel. Data arrived with per-batch means already equal, so the effective correction is equalising per-gene variance (later batch ~1.8x noisier). GSM1123370 kept, flagged as probable sex swap | expert group chose ComBat over plain scaling or no correction; EB shrinkage stabilises variance estimates from the 26-sample batch |
| 2026-09-17 | 3 | Accept the 21 ComBat+WGCNA modules; downstream features = 21 eigengenes + Age, Sex, COPD, Diabetes | modules map to recognisable blood cell-type programmes; batch correlation 0.00 |
