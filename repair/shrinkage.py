"""
Empirical Bayes Shrinkage for Classwise Offsets
================================================
Shrinks MLE classwise offsets a_y toward the grand mean using
an empirical Bayes (James-Stein-type) estimator.

This stabilizes rare-class offsets that are estimated from very few
calibration examples, reducing variance without large bias.

Estimator:
    a_y^shrunk = a_bar + B_y * (a_y - a_bar)
    B_y = max(0, 1 - sigma_y^2 / tau^2)

where:
    a_bar   = grand mean of MLE offsets
    sigma_y = estimated standard error of a_y (from Hessian or bootstrap)
    tau^2   = between-class variance of MLE offsets

Reference: Theorem 4.1 and Proposition 4.5 in the paper.
"""

import numpy as np


def shrink_offsets(a_mle, se_a=None, n_per_class=None):
    """Apply empirical Bayes shrinkage to classwise offsets.

    Parameters
    ----------
    a_mle : array of shape (K,)
        MLE classwise offsets from ``repair.core.fit``.
    se_a : array of shape (K,) or None
        Standard errors of each a_y. If None, estimated from n_per_class.
    n_per_class : array of shape (K,) or None
        Number of calibration examples per class (used to estimate SE
        if se_a is not provided).

    Returns
    -------
    a_shrunk : array of shape (K,)
        Shrunk classwise offsets.
    B : array of shape (K,)
        Shrinkage factors (0 = full shrinkage, 1 = no shrinkage).
    """
    K = len(a_mle)
    a_bar = np.mean(a_mle)

    # Estimate standard errors if not provided
    if se_a is None:
        if n_per_class is not None:
            # Rough SE estimate: inversely proportional to sqrt(n)
            se_a = 1.0 / np.sqrt(np.maximum(n_per_class, 1))
        else:
            # Fallback: constant SE from overall variance
            se_a = np.full(K, np.std(a_mle) / np.sqrt(K))

    # Between-class variance (tau^2)
    tau_sq = np.var(a_mle)

    # Shrinkage factor per class
    B = np.maximum(0.0, 1.0 - se_a ** 2 / (tau_sq + 1e-10))

    # Shrunk offsets
    a_shrunk = a_bar + B * (a_mle - a_bar)

    return a_shrunk, B


def shrink_offsets_bootstrap(a_samples):
    """Shrink offsets using bootstrap samples.

    Parameters
    ----------
    a_samples : array of shape (n_boot, K)
        Bootstrap samples of classwise offsets.

    Returns
    -------
    a_shrunk : array of shape (K,)
        Shrunk classwise offsets (mean of shrunk bootstrap samples).
    B : array of shape (K,)
        Shrinkage factors.
    """
    a_mle = a_samples.mean(axis=0)
    se_a = a_samples.std(axis=0)
    return shrink_offsets(a_mle, se_a=se_a)
