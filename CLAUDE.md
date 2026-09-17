# Working protocol for this repo
Read README.md first. It defines the goal, the data, and the step plan.

Rules:
- Work one step at a time. Never run ahead to a later step.
- For every step, deliver: what we do, why it applies to this dataset (n=89, p=13,519, omics), consequences, then stop for the expert group's checkpoint.
- Audience is biologists / omics experts, not ML engineers: explain statistical choices in terms of what they mean for biomarker validity, avoid ML jargon without a one-line definition.
- Any label-dependent operation (feature ranking, selection, scaling fitted on data, tuning) goes inside a sklearn Pipeline evaluated by cross-validation. Flag any violation explicitly.
- Record every checkpoint decision in DECISIONS.md (date, decision, reason).
- One script per step: stepNN_<name>.py, outputs under results/.
- After each step, add its section to OMISH.ipynb (markdown: what/why/consequences/decision, then `%run stepNN_*.py`). Scripts hold the code; the notebook never duplicates it.
- Every script sets fixed random seeds.
- Feature selection runs on WGCNA module representatives (eigengenes), not on raw genes. Selection itself uses scikit-learn.
