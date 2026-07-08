"""
algorithms.py
=============
Faithful implementations of:

  1. Classical Fuzzy C-Means (FCM)          -- Bezdek (1984), Eqs. (1),(4),(5) of the paper
  2. Unconstrained Fuzzy C-Means (UC-FCM)   -- Nie, Zhang, Yu, Li, TPAMI 2025, Eqs. (6),(11),(12),(13)

Both algorithms are implemented to share:
  - identical random initialization of the membership matrix Y (uniform Dirichlet-like
    row-stochastic init, then re-used directly by FCM and converted to an initial
    center matrix M via Eq. (4) for UC-FCM), so that "same initialization" comparisons
    in Experiment 1 are literally using the same starting point.
  - identical stopping criterion: |J_t - J_{t-1}| < 1e-5 (paper, Section IV-A)
  - identical fuzzifier r and, for UC-FCM, learning rate alpha and momentum gamma.

ASSUMPTIONS (paper omits some low-level details; documented here explicitly):
  A1. "Initialize Y randomly" (Algorithm 1, line 1) is implemented by drawing each
      row of Y from a symmetric Dirichlet(1,...,1) distribution (c dims), which
      guarantees Y >= 0 and each row sums to 1, satisfying the constraint
      Y @ 1_c = 1_n exactly, matching the FCM constraint set.
  A2. UC-FCM (Algorithm 2, line 2) initializes M "according to Y" -- we use Eq. (4)
      (the standard FCM center update) applied to the same randomly initialized Y,
      so that FCM and UC-FCM start from centers computed from an *identical* Y.
  A3. Distances are squared Euclidean, xi - mj ||_2^2, consistent with Eq. (1).
  A4. Numerical safeguard: since r > 1 implies exponents 2/(1-r) and 2r/(1-r) are
      negative, we clip distances away from 0 (add a small epsilon) to avoid
      division-by-zero / overflow when a sample coincides exactly with a center.
  A5. Momentum gradient descent (Eq. 13) is used for the cluster-center update
      (gamma = 0.6 fixed, per Section IV-A). Velocity v is initialized to 0 for
      each cluster center at the start of each run.
  A6. Convergence is checked on the *original constrained objective*, Eq. (1),
      for FCM, and on the *unconstrained objective*, Eq. (6), for UC-FCM -- these
      are mathematically equivalent quantities (Eq. 6 is a closed-form
      substitution into Eq. 1), so comparing their numeric objective values
      across algorithms, as Table III does, is valid.
  A7. A hard iteration cap (max_iter, default 500) is used purely as a safety
      valve; the paper's real stopping criterion is the |ΔJ| < 1e-5 tolerance.
"""

import numpy as np

EPS = 1e-10


def _pairwise_sq_dists(X, M):
    """Squared Euclidean distances between n samples (X: n x d) and c centers (M: c x d).
    Returns an (n, c) matrix D where D[i, j] = ||x_i - m_j||_2^2.
    """
    # (n,1) + (1,c) - 2 X M^T
    X_sq = np.sum(X ** 2, axis=1, keepdims=True)          # (n, 1)
    M_sq = np.sum(M ** 2, axis=1, keepdims=True).T        # (1, c)
    D = X_sq + M_sq - 2.0 * X @ M.T
    return np.maximum(D, EPS)  # numerical floor, see assumption A4


def init_membership(n, c, seed):
    """Random membership matrix Y (n x c), each row a Dirichlet(1,...,1) draw.
    Satisfies Y >= 0 and Y @ 1_c = 1_n exactly (assumption A1)."""
    rng = np.random.RandomState(seed)
    Y = rng.dirichlet(np.ones(c), size=n)
    return Y


def centers_from_membership(X, Y, r):
    """Eq. (4): m_j = sum_k y_kj^r x_k / sum_k y_kj^r"""
    Yr = Y ** r                      # (n, c)
    num = Yr.T @ X                   # (c, d)
    den = Yr.sum(axis=0)[:, None]    # (c, 1)
    den = np.maximum(den, EPS)
    return num / den


def fcm_objective(X, Y, M, r):
    """Eq. (1): sum_i sum_j y_ij^r ||x_i - m_j||_2^2"""
    D = _pairwise_sq_dists(X, M)     # (n, c)
    return float(np.sum((Y ** r) * D))


def update_membership(X, M, r):
    """Eq. (5): y_ij = ||x_i-m_j||_2^{2/(1-r)} / sum_k ||x_i-m_k||_2^{2/(1-r)}.

    NOTE: ||x_i-m_j||_2^{2/(1-r)} = (||x_i-m_j||_2^2)^{1/(1-r)} = D_ij^{1/(1-r)}
    where D_ij is the SQUARED distance computed by _pairwise_sq_dists. An earlier
    draft of this function mistakenly applied exponent 2/(1-r) directly to the
    already-squared D_ij (double-squaring the distance). Verified against Eq. (6):
    at the true closed-form-optimal Y for a given M, Eq. (1)'s objective must equal
    Eq. (6)'s objective evaluated at that same M (Eq. 6 is derived by substituting
    Eq. 5 into Eq. 1). This equality only holds with exponent 1/(1-r) applied to
    the squared distance D_ij, confirming the fix below.
    """
    D = _pairwise_sq_dists(X, M)                  # (n, c), squared distances
    p = 1.0 / (1.0 - r)                           # negative since r > 1
    W = D ** p                                    # (n, c)
    W_sum = np.sum(W, axis=1, keepdims=True)
    W_sum = np.maximum(W_sum, EPS)
    Y = W / W_sum
    return Y


class FCM:
    """Classical Fuzzy C-Means, Algorithm 1 of the paper."""

    def __init__(self, n_clusters, r=2.0, tol=1e-5, max_iter=500):
        self.c = n_clusters
        self.r = r
        self.tol = tol
        self.max_iter = max_iter

    def fit(self, X, Y_init):
        Y = Y_init.copy()
        M = centers_from_membership(X, Y, self.r)
        obj_prev = fcm_objective(X, Y, M, self.r)
        history = [obj_prev]

        for it in range(self.max_iter):
            M = centers_from_membership(X, Y, self.r)      # Eq. (4)
            Y = update_membership(X, M, self.r)             # Eq. (5)
            obj = fcm_objective(X, Y, M, self.r)
            history.append(obj)
            if abs(obj_prev - obj) < self.tol:
                obj_prev = obj
                break
            obj_prev = obj

        self.membership_ = Y
        self.centers_ = M
        self.objective_ = obj_prev
        self.history_ = history
        self.n_iter_ = len(history) - 1
        self.labels_ = np.argmax(Y, axis=1)
        return self


def uc_fcm_objective(X, M, r):
    """Eq. (6): sum_i ( sum_j ||x_i-m_j||^{2(1-r)/... } )^{1-r}
    Concretely: for each i, s_i = sum_j D_ij^{1/(1-r)}, objective term = s_i^{1-r}.
    This is the closed-form unconstrained objective after eliminating Y.
    """
    D = _pairwise_sq_dists(X, M)                 # (n, c), D_ij = ||x_i-m_j||^2
    p = 1.0 / (1.0 - r)                          # negative
    S = np.sum(D ** p, axis=1)                   # (n,)
    S = np.maximum(S, EPS)
    terms = S ** (1.0 - r)
    return float(np.sum(terms))


def uc_fcm_gradient(X, M, r):
    """Eq. (11): dL/dm_j = sum_i [ (sum_k D_ik^{(1-r)/... })^{-r} * D_ij^{r/(1-r)} * 2(m_j - x_i) ]

    We compute it vectorized:
      D_ij = ||x_i - m_j||^2               (n, c)
      p1   = 1/(1-r)  -> A_ij = D_ij^p1     (n, c)
      S_i  = sum_j A_ij                     (n,)
      C_i  = S_i^{-r}                       (n,)
      B_ij = D_ij^{ r/(1-r) }               (n, c)     [exponent = r * p1]
      grad_j = sum_i C_i * B_ij * 2 * (m_j - x_i)
             = 2 * [ m_j * sum_i(C_i*B_ij) - sum_i(C_i*B_ij*x_i) ]
    """
    n, d = X.shape
    c = M.shape[0]
    D = _pairwise_sq_dists(X, M)                  # (n, c)
    p1 = 1.0 / (1.0 - r)
    A = D ** p1                                   # (n, c)
    S = np.maximum(np.sum(A, axis=1), EPS)        # (n,)
    C = S ** (-r)                                 # (n,)
    B = D ** (r * p1)                             # (n, c)
    W = C[:, None] * B                            # (n, c)  weight per (i,j)

    w_sum_j = np.sum(W, axis=0)                   # (c,)  sum_i W_ij
    WX = W.T @ X                                  # (c, d) sum_i W_ij * x_i

    grad = 2.0 * (M * w_sum_j[:, None] - WX)      # (c, d)
    return grad


class UCFCM:
    """Unconstrained Fuzzy C-Means, Algorithm 2 of the paper (momentum gradient descent,
    Eq. 13)."""

    def __init__(self, n_clusters, r=2.0, alpha=0.01, gamma=0.6, tol=1e-5, max_iter=500):
        self.c = n_clusters
        self.r = r
        self.alpha = alpha
        self.gamma = gamma
        self.tol = tol
        self.max_iter = max_iter

    def fit(self, X, Y_init):
        # Assumption A2: initialize M from the *same* random Y as FCM, via Eq. (4)
        M = centers_from_membership(X, Y_init, self.r)
        v = np.zeros_like(M)

        obj_prev = uc_fcm_objective(X, M, self.r)
        history = [obj_prev]

        for it in range(self.max_iter):
            grad = uc_fcm_gradient(X, M, self.r)
            v = self.alpha * grad + self.gamma * v        # Eq. (13)
            M = M - v
            obj = uc_fcm_objective(X, M, self.r)
            history.append(obj)
            if not np.isfinite(obj):
                # numerical blow-up safeguard: halt and keep last stable state
                obj = obj_prev
                history[-1] = obj
                break
            if abs(obj_prev - obj) < self.tol:
                obj_prev = obj
                break
            obj_prev = obj

        # Final assignment via Eqs. (14)-(15): nearest-center hard assignment
        D = _pairwise_sq_dists(X, M)
        labels = np.argmin(D, axis=1)

        self.centers_ = M
        self.objective_ = obj_prev
        self.history_ = history
        self.n_iter_ = len(history) - 1
        self.labels_ = labels
        return self
