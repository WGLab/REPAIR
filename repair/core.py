"""
REPAIR Core Algorithm
=====================
Post-hoc reranking for long-tailed classification.

Score decomposition:  r_y(x) = g_y(x) + a_y + ell_y(x)
  g_y   = base logit (from pretrained model)
  a_y   = classwise offset (empirical Bayes shrinkage)
  ell_y = pairwise correction = (1/(k-1)) * sum_{j != y} theta^T phi(x,y,j)

Fitting is done via L-BFGS on conditional log-likelihood over calibration data.

Functions
---------
split_rare       : Partition classes into rare (bottom 80%) and frequent (top 20%)
get_shortlist    : Extract top-k shortlist indices from logits
compute_phi      : Compute pairwise feature tensor phi(x,y,j) for all (y,j) in S
fit              : Fit classwise offsets a and/or pairwise weights theta via L-BFGS
apply_scores     : Apply fitted a and theta to produce reranked scores
evaluate         : Compute Hit@1, Hit@3, MRR, Rare Hit@1, Freq Hit@1, HFR
"""

import numpy as np
from scipy.special import logsumexp
from scipy.optimize import minimize


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def split_rare(train_counts, K):
    """Partition classes into rare (bottom 80%) and frequent (top 20%) by count.

    Parameters
    ----------
    train_counts : array of shape (K,)
        Training set frequency for each class.
    K : int
        Number of classes.

    Returns
    -------
    rare_set : set of int
        Class indices in the rare stratum (bottom 80%).
    freq_set : set of int
        Class indices in the frequent stratum (top 20%).
    cutoff : int
        Training count at the boundary.
    """
    sorted_classes = sorted(range(K), key=lambda c: train_counts[c])
    n_rare = int(np.floor(0.8 * K))
    rare_set = set(sorted_classes[:n_rare])
    freq_set = set(sorted_classes[n_rare:])
    cutoff = train_counts[sorted_classes[n_rare]] if n_rare < K else 0
    return rare_set, freq_set, cutoff


def get_shortlist(logits, k):
    """Return top-k shortlist indices per example (descending by logit).

    Parameters
    ----------
    logits : array of shape (N, K)
        Full logit matrix.
    k : int
        Shortlist size.

    Returns
    -------
    shortlists : array of shape (N, k)
        Top-k class indices per example.
    """
    return np.argsort(logits, axis=1)[:, -k:][:, ::-1]


# ---------------------------------------------------------------------------
# Pairwise features
# ---------------------------------------------------------------------------

def compute_phi(g_S, shortlists, k, sim_matrix, train_counts):
    """Compute pairwise feature tensor for all shortlisted pairs.

    Features (5-dim if sim_matrix is provided, 4-dim otherwise):
      [0] score_gap       = g_y(x) - g_j(x)
      [1] rank_gap        = rank(j) - rank(y)
      [2] log_prob_ratio  = log(p_y / p_j)
      [3] domain_sim      = sim(y, j)             (only if sim_matrix given)
      [3 or 4] log_freq_ratio = log((n_y+1)/(n_j+1))

    Parameters
    ----------
    g_S : array of shape (N, k)
        Base scores restricted to shortlist.
    shortlists : array of shape (N, k)
        Shortlist class indices.
    k : int
        Shortlist size.
    sim_matrix : array of shape (K, K) or None
        Domain similarity matrix (e.g. WordNet, taxonomic, HPO Jaccard).
    train_counts : array of shape (K,)
        Training set class frequencies.

    Returns
    -------
    phi : array of shape (N, k, k, D_phi)
        Pairwise feature tensor. D_phi = 5 if sim_matrix else 4.
    """
    N = g_S.shape[0]
    has_sim = sim_matrix is not None
    D_phi = 5 if has_sim else 4
    phi = np.zeros((N, k, k, D_phi))

    # Ranks within shortlist
    ranks = np.zeros_like(g_S)
    for i in range(N):
        order = np.argsort(-g_S[i])
        ranks[i, order] = np.arange(k)

    # Softmax probabilities within shortlist
    probs = np.exp(g_S - logsumexp(g_S, axis=1, keepdims=True))

    for y in range(k):
        for j in range(k):
            if y == j:
                continue
            phi[:, y, j, 0] = g_S[:, y] - g_S[:, j]
            phi[:, y, j, 1] = ranks[:, j] - ranks[:, y]
            phi[:, y, j, 2] = np.log(
                probs[:, y] / (probs[:, j] + 1e-10) + 1e-10
            )
            sy = shortlists[:, y]
            sj = shortlists[:, j]
            if has_sim:
                phi[:, y, j, 3] = sim_matrix[sy, sj]
                phi[:, y, j, 4] = np.log(
                    (train_counts[sy] + 1) / (train_counts[sj] + 1)
                )
            else:
                phi[:, y, j, 3] = np.log(
                    (train_counts[sy] + 1) / (train_counts[sj] + 1)
                )
    return phi


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------

def fit(g_S, phi, shortlists, labels, K, k,
        lam_a=0.01, lam_t=0.001,
        fit_cw=True, fit_pw=True,
        alpha=0.0, train_counts=None):
    """Fit classwise offsets a and/or pairwise weights theta via L-BFGS.

    Maximizes (weighted) conditional log-likelihood:
        max_{a, theta}  sum_i w_i log p(y_i | S_i, x_i; a, theta)
                        - lam_a ||a||^2  - lam_t ||theta||^2

    Parameters
    ----------
    g_S : array of shape (N_cal, k)
        Base scores on calibration shortlists.
    phi : array of shape (N_cal, k, k, D_phi)
        Pairwise features on calibration shortlists.
    shortlists : array of shape (N_cal, k)
        Calibration shortlist indices.
    labels : array of shape (N_cal,)
        True class labels for calibration examples.
    K : int
        Total number of classes.
    k : int
        Shortlist size.
    lam_a : float
        L2 regularization for classwise offsets.
    lam_t : float
        L2 regularization for pairwise weights.
    fit_cw : bool
        Whether to fit classwise offsets a.
    fit_pw : bool
        Whether to fit pairwise weights theta.
    alpha : float in [0, 1]
        Interpolation weight for rare-class upweighting.
        0 = uniform, 1 = fully inverse-sqrt-freq weighted.
    train_counts : array of shape (K,) or None
        Required if alpha > 0.

    Returns
    -------
    a : array of shape (K,)
        Classwise offsets (zeros if fit_cw=False).
    theta : array of shape (D_phi,)
        Pairwise weights (zeros if fit_pw=False).
    """
    D_phi = phi.shape[-1]

    # Identify covered examples (true label in shortlist)
    covered, yidx = [], []
    for i in range(g_S.shape[0]):
        sl = list(shortlists[i])
        if labels[i] in sl:
            covered.append(i)
            yidx.append(sl.index(labels[i]))
    ci = np.array(covered)
    nc = len(ci)
    if nc < 5:
        return np.zeros(K), np.zeros(D_phi)

    gc = g_S[ci]
    pc = phi[ci]
    Sc = shortlists[ci]
    yc = np.array(yidx)

    # Compute example weights (rare-class upweighting)
    if train_counts is not None and alpha > 0:
        freq = train_counts[Sc[np.arange(nc), yc]]
        wu = np.ones(nc) / nc
        ws = 1.0 / np.sqrt(freq + 1)
        ws /= ws.sum()
        w = (1 - alpha) * wu + alpha * ws
    else:
        w = np.ones(nc) / nc

    n_params = (K if fit_cw else 0) + (D_phi if fit_pw else 0)
    x0 = np.zeros(n_params)

    def loss_and_grad(params):
        a = params[:K] if fit_cw else np.zeros(K)
        off = K if fit_cw else 0
        t = params[off:off + D_phi] if fit_pw else np.zeros(D_phi)

        # Scores
        aS = a[Sc]
        pw = np.einsum("nyjd,d->nyj", pc, t)
        np.einsum("nii->ni", pw)[:] = 0  # zero self-pairs
        ell = pw.sum(axis=2) / max(k - 1, 1)
        sc = gc + aS + ell

        # Log-likelihood
        lZ = logsumexp(sc, axis=1)
        lp = sc - lZ[:, None]
        nll = -(w * lp[np.arange(nc), yc]).sum()

        # Gradient
        ps = np.exp(lp)
        oh = np.zeros_like(ps)
        oh[np.arange(nc), yc] = 1.0
        ds = w[:, None] * (ps - oh)

        g = np.zeros(n_params)
        o = 0
        if fit_cw:
            ga = np.zeros(K)
            np.add.at(ga, Sc.ravel(), ds.ravel())
            g[:K] = ga + 2 * lam_a * a
            o = K
        if fit_pw:
            pps = pc.sum(axis=2) / max(k - 1, 1)
            g[o:o + D_phi] = (
                np.einsum("ny,nyd->d", ds, pps) + 2 * lam_t * t
            )

        reg = 0
        if fit_cw:
            reg += lam_a * (a ** 2).sum()
        if fit_pw:
            reg += lam_t * (t ** 2).sum()

        return nll + reg, g

    res = minimize(
        loss_and_grad, x0, jac=True, method="L-BFGS-B",
        options={"maxiter": 300, "ftol": 1e-11}
    )

    a = res.x[:K] if fit_cw else np.zeros(K)
    off = K if fit_cw else 0
    theta = res.x[off:off + D_phi] if fit_pw else np.zeros(D_phi)
    return a, theta


# ---------------------------------------------------------------------------
# Score application
# ---------------------------------------------------------------------------

def apply_scores(g_S, phi, shortlists, a, theta, k):
    """Apply classwise + pairwise corrections to produce reranked scores.

    Parameters
    ----------
    g_S : array of shape (N, k)
        Base scores on shortlist.
    phi : array of shape (N, k, k, D_phi)
        Pairwise features.
    shortlists : array of shape (N, k)
        Shortlist class indices.
    a : array of shape (K,)
        Classwise offsets.
    theta : array of shape (D_phi,)
        Pairwise weights.
    k : int
        Shortlist size.

    Returns
    -------
    scores : array of shape (N, k)
        Reranked scores.
    """
    scores = g_S + a[shortlists]
    if np.linalg.norm(theta) > 1e-10:
        pw = np.einsum("nyjd,d->nyj", phi, theta)
        np.einsum("nii->ni", pw)[:] = 0
        scores += pw.sum(axis=2) / max(k - 1, 1)
    return scores


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(scores, base_scores, shortlists, labels, k, rare_set):
    """Evaluate reranking quality conditioned on coverage (Y in S).

    Metrics:
      Hit@1   : fraction with rank-1 correct (overall)
      Hit@3   : fraction with rank <= 3 correct
      MRR     : mean reciprocal rank
      Rare H@1: Hit@1 restricted to rare classes
      Freq H@1: Hit@1 restricted to frequent classes
      HFR     : head-flip rate (fraction of initially-wrong examples corrected)
      Recall  : fraction of examples with true label in shortlist

    Parameters
    ----------
    scores : array of shape (N, k)
    base_scores : array of shape (N, k)
    shortlists : array of shape (N, k)
    labels : array of shape (N,)
    k : int
    rare_set : set of int

    Returns
    -------
    metrics : dict
    per_example_correct : list of (bool or None)
        Per-example correctness (None if not covered).
    """
    N = len(labels)
    h1, h3, mrr, nc = 0, 0, 0.0, 0
    h1r, nr, h1f, nf = 0, 0, 0, 0
    hfr_n, hfr_d = 0, 0
    per_example_correct = []

    for i in range(N):
        y = labels[i]
        sl = list(shortlists[i])
        if y not in sl:
            per_example_correct.append(None)
            continue
        nc += 1
        yi = sl.index(y)
        ry = scores[i, yi]
        rank = 1 + sum(1 for j in range(k) if j != yi and scores[i, j] > ry)
        correct = (rank == 1)
        per_example_correct.append(correct)
        h1 += int(correct)
        h3 += int(rank <= 3)
        mrr += 1.0 / rank

        if y in rare_set:
            h1r += int(correct); nr += 1
        else:
            h1f += int(correct); nf += 1

        # Head-flip rate
        bp = sl[base_scores[i].argmax()]
        if bp != y:
            hfr_d += 1
            rivals = [(base_scores[i, j2], j2) for j2 in range(k) if j2 != yi]
            if rivals:
                _, hi = max(rivals)
                if ry > scores[i, hi]:
                    hfr_n += 1

    metrics = {
        "hit1": h1 / max(nc, 1) * 100,
        "hit3": h3 / max(nc, 1) * 100,
        "mrr": mrr / max(nc, 1),
        "rare_hit1": h1r / max(nr, 1) * 100,
        "freq_hit1": h1f / max(nf, 1) * 100,
        "hfr": hfr_n / max(hfr_d, 1),
        "recall": nc / N * 100,
        "n_cov": nc,
        "n_total": N,
        "n_rare": nr,
        "n_freq": nf,
    }
    return metrics, per_example_correct
