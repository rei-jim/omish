"""Parametric ComBat (Johnson, Li & Rabinovic 2007), same algorithm as sva::ComBat with par.prior=TRUE.
Location/scale batch adjustment with empirical-Bayes shrinkage of per-gene batch effects toward the
across-gene average. No biological covariates are passed, so the label is never used.
combat(X, batch) -> corrected DataFrame, X is samples x genes."""
import numpy as np, pandas as pd

def combat(X: pd.DataFrame, batch: pd.Series, n_iter=100, tol=1e-4) -> pd.DataFrame:
    Y = X.values.T.astype(float)                      # genes x samples
    b = pd.Categorical(batch.loc[X.index]); levels = list(b.categories)
    n_b = np.array([(b == l).sum() for l in levels]); n = Y.shape[1]
    D = np.stack([(b == l).astype(float) for l in levels], 1)          # samples x batches design
    # 1. standardise: grand mean and pooled variance per gene
    B = np.linalg.lstsq(D, Y.T, rcond=None)[0].T                        # genes x batches: per-batch means
    grand = B @ (n_b / n)
    var_pooled = ((Y - (D @ B.T).T) ** 2).mean(1)
    Z = (Y - grand[:, None]) / np.sqrt(var_pooled)[:, None]
    # 2. per-gene batch effects on standardised data
    gamma_hat = np.linalg.lstsq(D, Z.T, rcond=None)[0].T                # genes x batches
    delta_hat = np.stack([Z[:, b == l].var(1, ddof=1) for l in levels], 1)
    # 3. parametric priors: gamma ~ N(g_bar, t2), delta2 ~ InvGamma(a_prior, b_prior)
    g_bar, t2 = gamma_hat.mean(0), gamma_hat.var(0, ddof=1)
    m, s2 = delta_hat.mean(0), delta_hat.var(0, ddof=1)
    a_prior, b_prior = (2 * s2 + m ** 2) / s2, (m * s2 + m ** 3) / s2
    # 4. EB estimates by iterating posterior means per batch
    gamma_star, delta_star = np.empty_like(gamma_hat), np.empty_like(delta_hat)
    for j, l in enumerate(levels):
        Zj = Z[:, b == l]; nj = n_b[j]
        g_new, d_new = gamma_hat[:, j].copy(), delta_hat[:, j].copy()
        for _ in range(n_iter):
            g_old, d_old = g_new, d_new
            g_new = (t2[j] * nj * gamma_hat[:, j] + d_old * g_bar[j]) / (t2[j] * nj + d_old)
            ss = ((Zj - g_new[:, None]) ** 2).sum(1)
            d_new = (0.5 * ss + b_prior[j]) / (nj / 2 + a_prior[j] - 1)
            if max(np.abs(g_new - g_old).max(), np.abs(d_new - d_old).max()) < tol: break
        gamma_star[:, j], delta_star[:, j] = g_new, d_new
    # 5. adjust: remove EB batch shift, rescale by EB batch spread, restore grand mean and pooled variance
    adj = (Z - (D @ gamma_star.T).T) / np.sqrt((D @ delta_star.T).T)
    out = adj * np.sqrt(var_pooled)[:, None] + grand[:, None]
    return pd.DataFrame(out.T, index=X.index, columns=X.columns)

if __name__ == "__main__":  # self-check: a planted location+scale batch effect is removed
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(5, 1, (90, 300)))
    batch = pd.Series(["A"] * 60 + ["B"] * 30, index=X.index)
    X.loc[batch == "B"] = X.loc[batch == "B"] * 2 + 1.5
    C = combat(X, batch)
    ratio = (C[batch == "B"].std() / C[batch == "A"].std()).median()
    shift = (C[batch == "B"].mean() - C[batch == "A"].mean()).abs().median()
    assert 0.85 < ratio < 1.15 and shift < 0.15, (ratio, shift)
    print(f"ok: post-ComBat SD ratio {ratio:.2f}, mean shift {shift:.3f}")
