"""
Neural state analysis and dynamical model fitting for CeDNe.

This module provides statistical and analytical methods for analyzing
neural population activity. Unlike ``optimizer.py`` (which searches
parameter spaces for simulation models using Optuna/JAX), this module
focuses on closed-form or iterative analytical fits:

- **State space analysis**: SVD, PCA, NMF for dimensionality reduction
  of neural population activity.
- **Dynamical model fitting**: VAR and LDS models fit to neural
  time-series data with optional regularization and connectome
  constraints.
- **Connectome comparison**: Threshold fitted dynamics matrices and
  compare with structural connectome adjacency.
"""

__author__ = "Sahil Moza"
__date__ = "2025-03-27"
__license__ = "MIT"

import numpy as np
from numpy.typing import NDArray
from typing import Optional, Dict, Any, List, Tuple
from scipy import linalg
from scipy.sparse.linalg import svds


def _truncated_svd(
    matrix: NDArray, n_components: int
) -> Tuple[NDArray, NDArray, NDArray]:
    """Top-K SVD.

    Uses ``scipy.sparse.linalg.svds`` for ``k < min(M, N)``, which computes only
    the top singular triplets and scales linearly in matrix size — avoids the
    O(min(M^2 N, N^2 M)) cost of a full dense SVD when we only want a handful
    of components. Falls back to the full dense SVD for small matrices where
    ``svds`` is not applicable (``k`` must be strictly less than ``min(M, N)``).

    Returns ``(U, S, Vt)`` in *descending* singular-value order, matching
    ``scipy.linalg.svd``'s convention. ``svds`` itself returns ascending order;
    we reverse here so callers don't need to know which path ran.
    """
    M, T = matrix.shape
    min_dim = min(M, T)
    if n_components >= min_dim:
        U, S, Vt = linalg.svd(matrix, full_matrices=False)
        return U[:, :n_components], S[:n_components], Vt[:n_components, :]
    U, S, Vt = svds(matrix, k=n_components)
    order = np.argsort(S)[::-1]
    return U[:, order], S[order], Vt[order, :]


# ═══════════════════════════════════════════════════════════════════
# State Space Analysis
# ═══════════════════════════════════════════════════════════════════


def compute_state_space(
    activity_matrix: NDArray,
    method: str = "svd",
    n_components: int = 3,
    neuron_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Compute low-dimensional neural state representation.

    Projects high-dimensional neural population activity into a
    low-dimensional state space using SVD, PCA, or NMF.

    Args:
        activity_matrix: (N_neurons × T_timepoints) array of neural
            activity. Rows are neurons, columns are time points.
        method: Decomposition method.
            - ``"svd"``: Truncated SVD (no mean-centering).
            - ``"pca"``: PCA (mean-centered SVD).
            - ``"nmf"``: Non-negative matrix factorization.
        n_components: Number of components to extract.
        neuron_names: Optional ordered list of neuron names matching
            rows of *activity_matrix*.

    Returns:
        Dictionary with:
            - ``projections``: (T × n_components) state trajectories.
            - ``loadings``: (N_neurons × n_components) neuron weights.
            - ``explained_variance``: Per-component variance fraction.
            - ``singular_values``: Raw singular values (SVD/PCA only).
            - ``neuron_names``: Ordered list matching rows.
            - ``method``: Method used.

    Raises:
        ValueError: If method is unknown or matrix dimensions are invalid.
    """
    if activity_matrix.ndim != 2:
        raise ValueError(
            f"activity_matrix must be 2-D (N×T), got {activity_matrix.ndim}-D"
        )

    N, T = activity_matrix.shape
    if n_components > min(N, T):
        n_components = min(N, T)

    names = neuron_names or [f"neuron_{i}" for i in range(N)]

    if method == "svd":
        # Top-K SVD; total variance comes from the matrix Frobenius norm so
        # the explained-variance fractions remain comparable to the full-SVD
        # path (||X||_F^2 == sum of all sigma^2).
        U, S, Vt = _truncated_svd(activity_matrix, n_components)
        loadings = U  # N × K
        projections = (np.diag(S) @ Vt).T  # T × K
        total_var = float((activity_matrix**2).sum())
        explained = (S**2) / total_var if total_var > 0 else np.zeros(n_components)
        singular_values = S

    elif method == "pca":
        mean = activity_matrix.mean(axis=1, keepdims=True)
        centered = activity_matrix - mean
        U, S, Vt = _truncated_svd(centered, n_components)
        loadings = U
        projections = (np.diag(S) @ Vt).T
        total_var = float((centered**2).sum())
        explained = (S**2) / total_var if total_var > 0 else np.zeros(n_components)
        singular_values = S

    elif method == "nmf":
        from sklearn.decomposition import NMF

        # NMF requires non-negative input
        X = activity_matrix.T  # (T × N) — sklearn convention
        X_nn = np.clip(X, 0, None)
        model = NMF(n_components=n_components, max_iter=500, random_state=42)
        projections = model.fit_transform(X_nn)  # T × K
        loadings = model.components_.T  # N × K
        # Approximate explained variance via reconstruction error
        reconstruction = projections @ model.components_
        total_var = np.sum(X_nn**2)
        residual_var = np.sum((X_nn - reconstruction) ** 2)
        total_explained = 1 - residual_var / total_var if total_var > 0 else 0.0
        # Split proportionally by component norm
        comp_norms = np.sum(model.components_**2, axis=1)
        comp_fracs = (
            comp_norms / comp_norms.sum()
            if comp_norms.sum() > 0
            else np.ones(n_components) / n_components
        )
        explained = comp_fracs * total_explained
        singular_values = np.sqrt(comp_norms)

    else:
        raise ValueError(f"Unknown method '{method}'. Use 'svd', 'pca', or 'nmf'.")

    # Component-label prefix for axes/legends: PCA produces principal components
    # ("PC"), NMF produces non-negative factors ("Factor"), and a plain
    # uncentered SVD produces neither — call them generic components ("Comp").
    # Centralising this here is deliberate: the frontend should not switch on
    # method itself, it just renders whatever label the analysis returned.
    component_label = {"pca": "PC", "nmf": "Factor"}.get(method, "Comp")

    return {
        "projections": projections,
        "loadings": loadings,
        "explained_variance": explained.tolist(),
        "singular_values": singular_values.tolist(),
        "neuron_names": names,
        "method": method,
        "n_components": n_components,
        "component_label": component_label,
    }


# ═══════════════════════════════════════════════════════════════════
# Dynamical Model Fitting
# ═══════════════════════════════════════════════════════════════════


def fit_var(
    activity_matrix: NDArray,
    lag: int = 1,
    regularizer: float = 0.0,
    connectome_mask: Optional[NDArray] = None,
    neuron_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Fit a Vector Autoregressive (VAR) model to neural activity.

    Model: x(t+1) = A @ x(t) + noise  (for lag=1)

    For higher lags, constructs the standard VAR(p) regression:
    x(t) = A₁ x(t-1) + A₂ x(t-2) + ... + Aₚ x(t-p) + noise

    Uses Ridge regression (L2 regularization) for stability.

    Args:
        activity_matrix: (N × T) neural activity time series.
        lag: VAR lag order (number of past time steps).
        regularizer: L2 (Ridge) regularization strength (λ).
            Higher values encourage smaller coefficients.
        connectome_mask: Optional (N × N) binary mask. If provided,
            entries of A where mask==0 are forced to zero after fitting
            (connectome-constrained). For lag > 1, the mask is applied
            to each Aᵢ block.
        neuron_names: Optional names matching rows.

    Returns:
        Dictionary with:
            - ``A``: (N × N*lag) coefficient matrix (or (N × N) for lag=1).
            - ``A_blocks``: List of (N × N) matrices [A₁, A₂, ..., Aₚ].
            - ``residuals``: (N × T-lag) residual matrix.
            - ``r_squared``: Per-neuron R² values.
            - ``r_squared_mean``: Mean R² across neurons.
            - ``aic``: Akaike information criterion.
            - ``neuron_names``: Ordered list.
    """
    N, T = activity_matrix.shape
    names = neuron_names or [f"neuron_{i}" for i in range(N)]

    if lag < 1:
        raise ValueError("lag must be >= 1")
    if T <= lag:
        raise ValueError(f"Not enough time points ({T}) for lag={lag}")

    # Build regression matrices
    # Y = A @ X  where Y[:, t] = x(t+lag) and X[:, t] = [x(t+lag-1); ...; x(t)]
    Y = activity_matrix[:, lag:]  # N × (T - lag)
    X_blocks = []
    for p in range(1, lag + 1):
        X_blocks.append(activity_matrix[:, lag - p : T - p])
    X = np.vstack(X_blocks)  # (N*lag) × (T - lag)

    # Ridge regression: A = Y @ X.T @ (X @ X.T + λI)^{-1}.
    # Use linalg.solve instead of forming the inverse explicitly — same result,
    # faster (one factorization vs. inversion), and more numerically stable
    # near a singular XXT. (XXT + λI) is symmetric, so we pass assume_a='sym'
    # to take the symmetric solver path.
    XXT = X @ X.T
    reg_matrix = regularizer * np.eye(N * lag)
    # Solve (XXT + λI) @ A.T = X @ Y.T  →  A = solve(...).T
    A = linalg.solve(XXT + reg_matrix, X @ Y.T, assume_a="sym").T  # N × (N*lag)

    # Apply connectome mask if provided
    if connectome_mask is not None:
        mask = np.asarray(connectome_mask, dtype=float)
        if mask.shape != (N, N):
            raise ValueError(
                f"connectome_mask shape {mask.shape} doesn't match ({N}, {N})"
            )
        # Apply mask to each lag block
        for p in range(lag):
            A[:, p * N : (p + 1) * N] *= mask

    # Compute residuals and R²
    Y_hat = A @ X
    residuals = Y - Y_hat
    ss_res = np.sum(residuals**2, axis=1)
    ss_tot = np.sum((Y - Y.mean(axis=1, keepdims=True)) ** 2, axis=1)
    r_squared = 1 - ss_res / np.maximum(ss_tot, 1e-12)

    # AIC: N_obs * ln(det(Σ)) + 2 * n_params
    n_obs = T - lag
    sigma = residuals @ residuals.T / n_obs
    sign, logdet = np.linalg.slogdet(sigma)
    n_params = N * N * lag
    aic = n_obs * logdet + 2 * n_params

    # Split A into per-lag blocks
    A_blocks = [A[:, p * N : (p + 1) * N] for p in range(lag)]

    return {
        "A": A,
        "A_blocks": A_blocks,
        "residuals": residuals,
        "r_squared": r_squared.tolist(),
        "r_squared_mean": float(np.mean(r_squared)),
        "aic": float(aic),
        "neuron_names": names,
        "lag": lag,
        "regularizer": regularizer,
    }


def fit_lds(
    activity_matrix: NDArray,
    n_latent: Optional[int] = None,
    regularizer: float = 0.0,
    connectome_mask: Optional[NDArray] = None,
    max_iter: int = 100,
    tol: float = 1e-4,
    neuron_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Fit a Linear Dynamical System (LDS) via Expectation-Maximization.

    Model:
        z(t+1) = A @ z(t) + w,   w ~ N(0, Q)
        x(t)   = C @ z(t) + v,   v ~ N(0, R)

    where z is the latent state and x is observed neural activity.

    When n_latent == N (observed dimensionality), this reduces to a
    state-space model in the observed space.

    Args:
        activity_matrix: (N × T) observed neural activity.
        n_latent: Latent state dimensionality. Defaults to N.
        regularizer: L2 regularization on A matrix.
        connectome_mask: Optional (K × K) mask on A (where K = n_latent).
        max_iter: Maximum EM iterations.
        tol: Convergence tolerance on log-likelihood.
        neuron_names: Optional names matching rows.

    Returns:
        Dictionary with:
            - ``A``: (K × K) dynamics matrix.
            - ``C``: (N × K) observation matrix.
            - ``Q``: (K × K) process noise covariance.
            - ``R``: (N × N) observation noise covariance.
            - ``latent_states``: (K × T) inferred latent states.
            - ``log_likelihood``: Final log-likelihood.
            - ``log_likelihood_history``: Per-iteration LL.
            - ``converged``: Whether EM converged.
            - ``neuron_names``: Ordered list.
    """
    N, T = activity_matrix.shape
    K = n_latent if n_latent is not None else N
    names = neuron_names or [f"neuron_{i}" for i in range(N)]

    # Initialize parameters
    # Use PCA for initial C and latent states
    mean = activity_matrix.mean(axis=1, keepdims=True)
    centered = activity_matrix - mean
    U, S, Vt = linalg.svd(centered, full_matrices=False)

    C = U[:, :K]  # N × K
    z = np.diag(S[:K]) @ Vt[:K, :]  # K × T  (initial latent)
    A = np.eye(K) * 0.9  # K × K  (stable init)
    Q = np.eye(K) * 0.1  # K × K
    R = np.eye(N) * 0.1  # N × N
    z0 = z[:, 0]  # K,     initial state mean
    P0 = np.eye(K)  # K × K  initial state cov

    ll_history = []
    converged = False

    for iteration in range(max_iter):
        # ── E-step: Kalman smoother ──
        # Forward pass (filter)
        z_filt = np.zeros((K, T))
        P_filt = np.zeros((K, K, T))
        z_pred = np.zeros((K, T))
        P_pred = np.zeros((K, K, T))

        z_pred[:, 0] = z0
        P_pred[:, :, 0] = P0

        log_lik = 0.0

        for t in range(T):
            # Innovation
            y_pred = C @ z_pred[:, t]
            innov = activity_matrix[:, t] - y_pred
            S_innov = C @ P_pred[:, :, t] @ C.T + R
            try:
                S_inv = linalg.inv(S_innov)
            except linalg.LinAlgError:
                S_inv = linalg.pinv(S_innov)

            # Kalman gain
            K_gain = P_pred[:, :, t] @ C.T @ S_inv

            # Update
            z_filt[:, t] = z_pred[:, t] + K_gain @ innov
            P_filt[:, :, t] = (np.eye(K) - K_gain @ C) @ P_pred[:, :, t]

            # Log-likelihood contribution
            sign, logdet = np.linalg.slogdet(S_innov)
            if sign > 0:
                log_lik += -0.5 * (
                    N * np.log(2 * np.pi) + logdet + innov @ S_inv @ innov
                )

            # Predict next step
            if t < T - 1:
                z_pred[:, t + 1] = A @ z_filt[:, t]
                P_pred[:, :, t + 1] = A @ P_filt[:, :, t] @ A.T + Q

        ll_history.append(float(log_lik))

        # Check convergence
        if iteration > 0 and abs(ll_history[-1] - ll_history[-2]) < tol:
            converged = True
            break

        # Backward pass (smoother)
        z_smooth = np.zeros((K, T))
        P_smooth = np.zeros((K, K, T))
        Plag_smooth = np.zeros((K, K, T))  # E[z_t z_{t-1}^T]

        z_smooth[:, T - 1] = z_filt[:, T - 1]
        P_smooth[:, :, T - 1] = P_filt[:, :, T - 1]

        for t in range(T - 2, -1, -1):
            try:
                J = P_filt[:, :, t] @ A.T @ linalg.inv(P_pred[:, :, t + 1])
            except linalg.LinAlgError:
                J = P_filt[:, :, t] @ A.T @ linalg.pinv(P_pred[:, :, t + 1])
            z_smooth[:, t] = z_filt[:, t] + J @ (z_smooth[:, t + 1] - A @ z_filt[:, t])
            P_smooth[:, :, t] = (
                P_filt[:, :, t]
                + J @ (P_smooth[:, :, t + 1] - P_pred[:, :, t + 1]) @ J.T
            )
            Plag_smooth[:, :, t + 1] = P_smooth[:, :, t + 1] @ J.T

        # ── M-step ──
        # Sufficient statistics
        Ez = z_smooth  # K × T
        Ezz = np.zeros((K, K))  # sum E[z_t z_t^T]
        Ezz_prev = np.zeros((K, K))  # sum E[z_{t-1} z_{t-1}^T]
        Ezz_cross = np.zeros((K, K))  # sum E[z_t z_{t-1}^T]

        for t in range(T):
            Ezz += P_smooth[:, :, t] + np.outer(z_smooth[:, t], z_smooth[:, t])
        for t in range(1, T):
            Ezz_prev += P_smooth[:, :, t - 1] + np.outer(
                z_smooth[:, t - 1], z_smooth[:, t - 1]
            )
            Ezz_cross += Plag_smooth[:, :, t] + np.outer(
                z_smooth[:, t], z_smooth[:, t - 1]
            )

        # Update A with regularization
        reg = regularizer * np.eye(K)
        try:
            A = Ezz_cross @ linalg.inv(Ezz_prev + reg)
        except linalg.LinAlgError:
            A = Ezz_cross @ linalg.pinv(Ezz_prev + reg)

        # Apply connectome mask
        if connectome_mask is not None:
            mask = np.asarray(connectome_mask, dtype=float)
            if mask.shape == (K, K):
                A *= mask

        # Update Q
        Q = (Ezz - Ezz_cross @ A.T - A @ Ezz_cross.T + A @ Ezz_prev @ A.T) / (T - 1)
        Q = (Q + Q.T) / 2 + 1e-6 * np.eye(K)  # Symmetrize + regularize

        # Update C
        Yz = activity_matrix @ Ez.T  # N × K
        try:
            C = Yz @ linalg.inv(Ezz)
        except linalg.LinAlgError:
            C = Yz @ linalg.pinv(Ezz)

        # Update R
        R_new = (activity_matrix @ activity_matrix.T - C @ Yz.T) / T
        R = (R_new + R_new.T) / 2 + 1e-6 * np.eye(N)

        # Update initial state
        z0 = z_smooth[:, 0]
        P0 = P_smooth[:, :, 0]

    return {
        "A": A,
        "C": C,
        "Q": Q,
        "R": R,
        "latent_states": z_smooth,
        "log_likelihood": float(ll_history[-1]) if ll_history else 0.0,
        "log_likelihood_history": ll_history,
        "converged": converged,
        "n_iterations": iteration + 1,
        "neuron_names": names,
        "n_latent": K,
        "regularizer": regularizer,
    }


# ═══════════════════════════════════════════════════════════════════
# Connectome Comparison
# ═══════════════════════════════════════════════════════════════════


def threshold_and_compare(
    A_fit: NDArray,
    connectome_adj: NDArray,
    threshold: float = 0.1,
    absolute: bool = True,
    neuron_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Threshold a fitted dynamics matrix and compare with connectome.

    Binarizes the fitted A matrix by thresholding and computes overlap
    metrics against the structural connectome adjacency matrix.

    Args:
        A_fit: (N × N) fitted dynamics matrix (e.g., from VAR).
        connectome_adj: (N × N) binary or weighted adjacency matrix
            from the structural connectome. Non-zero entries are
            treated as connections.
        threshold: Threshold value for binarizing A_fit.
        absolute: If True, threshold on |A_ij|; else on raw A_ij.
        neuron_names: Optional neuron names for edge labeling.

    Returns:
        Dictionary with:
            - ``thresholded_A``: (N × N) binary matrix.
            - ``precision``: TP / (TP + FP).
            - ``recall``: TP / (TP + FN).
            - ``f1``: F1 score.
            - ``jaccard``: Jaccard similarity.
            - ``true_positives``: List of (i, j) edge tuples.
            - ``false_positives``: Edges in A but not connectome.
            - ``false_negatives``: Edges in connectome but not A.
            - ``n_inferred``: Number of inferred connections.
            - ``n_connectome``: Number of connectome connections.
    """
    N = A_fit.shape[0]
    if A_fit.shape != (N, N):
        raise ValueError(f"A_fit must be square, got {A_fit.shape}")
    if connectome_adj.shape != (N, N):
        raise ValueError(
            f"connectome_adj shape {connectome_adj.shape} != A_fit shape {A_fit.shape}"
        )

    names = neuron_names or [f"neuron_{i}" for i in range(N)]

    # Binarize
    if absolute:
        inferred = (np.abs(A_fit) > threshold).astype(int)
    else:
        inferred = (A_fit > threshold).astype(int)

    # Remove diagonal (self-connections)
    np.fill_diagonal(inferred, 0)

    connectome_binary = (np.abs(connectome_adj) > 0).astype(int)
    np.fill_diagonal(connectome_binary, 0)

    # Compute metrics
    tp_mask = (inferred == 1) & (connectome_binary == 1)
    fp_mask = (inferred == 1) & (connectome_binary == 0)
    fn_mask = (inferred == 0) & (connectome_binary == 1)

    tp = int(np.sum(tp_mask))
    fp = int(np.sum(fp_mask))
    fn = int(np.sum(fn_mask))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    jaccard = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0

    # Edge lists
    def _edge_list(mask):
        indices = np.argwhere(mask)
        return [(names[i], names[j]) for i, j in indices]

    return {
        "thresholded_A": inferred,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "jaccard": float(jaccard),
        "true_positives": _edge_list(tp_mask),
        "false_positives": _edge_list(fp_mask),
        "false_negatives": _edge_list(fn_mask),
        "n_inferred": int(np.sum(inferred)),
        "n_connectome": int(np.sum(connectome_binary)),
        "threshold": threshold,
    }
