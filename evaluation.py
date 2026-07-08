"""
evaluation.py
=============
Clustering evaluation metrics exactly as defined in the paper (Eqs. 16-17) plus ARI,
and the statistical tests used in Section IV-C/D (Friedman test + Nemenyi post-hoc,
Wilcoxon signed-rank as an additional pairwise check requested for this reproduction).
"""

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from scipy.stats import friedmanchisquare, wilcoxon


def nmi_score(y_true, y_pred):
    """Eq. (16). sklearn's normalized_mutual_info_score with 'arithmetic' averaging
    matches the standard NMI definition used in the clustering literature."""
    return normalized_mutual_info_score(y_true, y_pred, average_method="arithmetic")


def ari_score(y_true, y_pred):
    return adjusted_rand_score(y_true, y_pred)


def clustering_accuracy(y_true, y_pred):
    """Eq. (17): ACC via optimal label permutation (Hungarian / Kuhn-Munkres algorithm),
    exactly as the paper specifies ('map(.) ... realized by the Hungarian algorithm')."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    labels_true = np.unique(y_true)
    labels_pred = np.unique(y_pred)
    n_true = labels_true.size
    n_pred = labels_pred.size
    size = max(n_true, n_pred)

    cost = np.zeros((size, size), dtype=np.int64)
    true_index = {l: i for i, l in enumerate(labels_true)}
    pred_index = {l: i for i, l in enumerate(labels_pred)}
    for t, p in zip(y_true, y_pred):
        cost[pred_index[p], true_index[t]] += 1

    row_ind, col_ind = linear_sum_assignment(-cost)  # maximize matches
    matched = cost[row_ind, col_ind].sum()
    return matched / y_true.size


def evaluate_clustering(y_true, y_pred):
    return {
        "NMI": nmi_score(y_true, y_pred),
        "ACC": clustering_accuracy(y_true, y_pred),
        "ARI": ari_score(y_true, y_pred),
    }


# ---------------------------------------------------------------------------
# Statistical tests (Section IV-C: Friedman test + Nemenyi post-hoc; Table IX)
# ---------------------------------------------------------------------------

def friedman_test(rank_matrix):
    """rank_matrix: (N_datasets, K_algorithms) of raw performance scores (higher=better).
    Returns (chi2_stat, p_value, mean_ranks) using the paper's chi-square formula (Eq. 18),
    which is the classical Friedman statistic (average-rank based, higher-is-better)."""
    N, K = rank_matrix.shape
    # rank each row (dataset) in descending order of performance -> rank 1 = best
    ranks = np.zeros_like(rank_matrix, dtype=float)
    for i in range(N):
        order = np.argsort(-rank_matrix[i])  # descending
        r = np.empty(K)
        r[order] = np.arange(1, K + 1)
        ranks[i] = r
    mean_ranks = ranks.mean(axis=0)
    chi2 = (12.0 * N / (K * (K + 1))) * (np.sum(mean_ranks ** 2) - K * (K + 1) ** 2 / 4.0)
    # p-value via chi-square survival function with K-1 dof
    from scipy.stats import chi2 as chi2_dist
    p = chi2_dist.sf(chi2, df=K - 1)
    return chi2, p, mean_ranks


def nemenyi_cd(K, N, q_alpha=3.164):
    """Eq. (19): Critical Difference for the Nemenyi post-hoc test.
    q_alpha=3.164 is the paper's stated value (alpha=0.05, K=10 algorithms)."""
    return q_alpha * np.sqrt(K * (K + 1) / (6.0 * N))


def wilcoxon_test(scores_a, scores_b):
    """Paired Wilcoxon signed-rank test between two algorithms' per-dataset (or per-run)
    scores. Returns (statistic, p_value). Falls back gracefully if all differences
    are zero (scipy raises in that degenerate case)."""
    diffs = np.asarray(scores_a) - np.asarray(scores_b)
    if np.allclose(diffs, 0):
        return 0.0, 1.0
    stat, p = wilcoxon(scores_a, scores_b)
    return stat, p
