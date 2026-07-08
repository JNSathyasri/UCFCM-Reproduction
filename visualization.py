"""
convergence curves, sensitivity heatmaps,
runtime bar charts, Nemenyi critical-difference diagrams, PCA/t-SNE embeddings,
confusion matrices, boxplots over runs.

"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "axes.grid": True,
    "grid.alpha": 0.3,
})

FIG_DIR = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(FIG_DIR, exist_ok=True)


def _save(fig, name):
    for ext in ("png", "pdf", "svg"):
        fig.savefig(os.path.join(FIG_DIR, f"{name}.{ext}"), bbox_inches="tight")
    plt.close(fig)


def plot_convergence(history_fcm, history_ucfcm, dataset_name):
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(history_fcm, label="FCM", marker="o", markersize=3, linewidth=1.5)
    ax.plot(history_ucfcm, label="UC-FCM", marker="s", markersize=3, linewidth=1.5)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Objective value")
    ax.set_title(f"Convergence: {dataset_name}")
    ax.legend()
    _save(fig, f"convergence_{dataset_name}")


def plot_convergence_grid(all_histories, dataset_names, ncols=5):
    n = len(dataset_names)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 2.8 * nrows))
    axes = np.array(axes).reshape(-1)
    for i, name in enumerate(dataset_names):
        ax = axes[i]
        h_fcm, h_uc = all_histories[name]
        ax.plot(h_fcm, label="FCM", linewidth=1.2)
        ax.plot(h_uc, label="UC-FCM", linewidth=1.2)
        ax.set_title(name, fontsize=9)
        ax.tick_params(labelsize=7)
        if i == 0:
            ax.legend(fontsize=7)
    for j in range(n, len(axes)):
        axes[j].axis("off")
    fig.suptitle("Convergent curves of FCM and UC-FCM (cf. Fig. 5 of the paper)")
    fig.tight_layout()
    _save(fig, "convergence_grid_all_datasets")


def plot_sensitivity_heatmap(acc_grid, r_values, alpha_values, dataset_name):
    """acc_grid: (len(r_values), len(alpha_values)) mean ACC values."""
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(acc_grid, aspect="auto", cmap="viridis", origin="lower")
    ax.set_xticks(range(len(alpha_values)))
    ax.set_xticklabels(alpha_values)
    ax.set_yticks(range(len(r_values)))
    ax.set_yticklabels(r_values)
    ax.set_xlabel(r"$\alpha$ (learning rate)")
    ax.set_ylabel(r"$r$ (fuzzy exponent)")
    ax.set_title(f"UC-FCM sensitivity: {dataset_name}")
    fig.colorbar(im, ax=ax, label="ACC")
    _save(fig, f"sensitivity_{dataset_name}")


def plot_objective_vs_r(obj_by_r, r_values, dataset_name):
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(r_values, obj_by_r, marker="o")
    ax.set_xlabel("r")
    ax.set_ylabel("Mean objective value")
    ax.set_title(f"Objective vs r: {dataset_name}")
    _save(fig, f"objective_vs_r_{dataset_name}")


def plot_runtime_bars(runtime_dict, dataset_name):
    """runtime_dict: {algo_name: seconds}"""
    fig, ax = plt.subplots(figsize=(6, 4))
    names = list(runtime_dict.keys())
    vals = [runtime_dict[n] for n in names]
    ax.bar(names, vals, color=plt.cm.tab10.colors[: len(names)])
    ax.set_ylabel("Runtime (s)")
    ax.set_title(f"Runtime comparison: {dataset_name}")
    plt.xticks(rotation=45, ha="right")
    _save(fig, f"runtime_{dataset_name}")


def plot_cd_diagram(mean_ranks, algo_names, cd, metric_name):
    order = np.argsort(mean_ranks)
    ranks_sorted = np.array(mean_ranks)[order]
    names_sorted = np.array(algo_names)[order]

    fig, ax = plt.subplots(figsize=(6, 0.4 * len(algo_names) + 1))
    y = np.arange(len(names_sorted))
    ax.errorbar(ranks_sorted, y, xerr=cd / 2, fmt="o", capsize=4, color="steelblue")
    ax.set_yticks(y)
    ax.set_yticklabels(names_sorted)
    ax.axvline(ranks_sorted[0] + cd, linestyle="--", color="gray", linewidth=1)
    ax.set_xlabel("Average rank")
    ax.set_title(f"Post-Hoc (Nemenyi) CD diagram — {metric_name}")
    _save(fig, f"cd_diagram_{metric_name}")


def plot_boxplot_over_runs(values_dict, dataset_name, metric_name):
    """values_dict: {algo_name: array of per-run values}"""
    fig, ax = plt.subplots(figsize=(5, 4))
    data = list(values_dict.values())
    names = list(values_dict.keys())
    # matplotlib >=3.9 renamed the `labels` kwarg to `tick_labels`; support both.
    try:
        ax.boxplot(data, tick_labels=names)
    except TypeError:
        ax.boxplot(data, labels=names)
    ax.set_ylabel(metric_name)
    ax.set_title(f"{metric_name} over 20 runs: {dataset_name}")
    _save(fig, f"boxplot_{metric_name}_{dataset_name}")


def plot_pca(X, y_true, y_fcm, y_uc, dataset_name):
    from sklearn.decomposition import PCA
    Z = PCA(n_components=2, random_state=0).fit_transform(X)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, labels, title in zip(
        axes, [y_true, y_fcm, y_uc], ["Ground truth", "FCM", "UC-FCM"]
    ):
        sc = ax.scatter(Z[:, 0], Z[:, 1], c=labels, cmap="tab20", s=12)
        ax.set_title(f"{title}")
    fig.suptitle(f"PCA visualization: {dataset_name}")
    _save(fig, f"pca_{dataset_name}")


def plot_tsne(X, y_true, y_fcm, y_uc, dataset_name):
    from sklearn.manifold import TSNE
    n = X.shape[0]
    perplexity = min(30, max(5, n // 4))
    Z = TSNE(n_components=2, random_state=0, perplexity=perplexity, init="pca").fit_transform(X)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, labels, title in zip(
        axes, [y_true, y_fcm, y_uc], ["Ground truth", "FCM", "UC-FCM"]
    ):
        ax.scatter(Z[:, 0], Z[:, 1], c=labels, cmap="tab20", s=12)
        ax.set_title(f"{title}")
    fig.suptitle(f"t-SNE visualization: {dataset_name}")
    _save(fig, f"tsne_{dataset_name}")


def plot_confusion_matrix(y_true, y_pred, dataset_name, algo_name):
    from sklearn.metrics import confusion_matrix
    from scipy.optimize import linear_sum_assignment

    labels_true = np.unique(y_true)
    labels_pred = np.unique(y_pred)
    size = max(len(labels_true), len(labels_pred))
    cost = np.zeros((size, size), dtype=np.int64)
    ti = {l: i for i, l in enumerate(labels_true)}
    pi = {l: i for i, l in enumerate(labels_pred)}
    for t, p in zip(y_true, y_pred):
        cost[pi[p], ti[t]] += 1
    row, col = linear_sum_assignment(-cost)
    mapping = {labels_pred[r]: labels_true[c] for r, c in zip(row, col) if r < len(labels_pred) and c < len(labels_true)}
    y_pred_mapped = np.array([mapping.get(p, p) for p in y_pred])

    cm = confusion_matrix(y_true, y_pred_mapped)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xlabel("Predicted (matched)")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion matrix: {algo_name} on {dataset_name}")
    fig.colorbar(im, ax=ax)
    _save(fig, f"confusion_{algo_name}_{dataset_name}")


def plot_gradient_norms(grad_norms, dataset_name):
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(grad_norms)
    ax.set_xlabel("Iteration")
    ax.set_ylabel(r"$\|\nabla_M L\|_F$")
    ax.set_title(f"Gradient norm: UC-FCM on {dataset_name}")
    ax.set_yscale("log")
    _save(fig, f"gradient_norm_{dataset_name}")


def plot_center_movement(center_traj, dataset_name):
    """center_traj: list of (c, d) center matrices across iterations."""
    dists = [np.linalg.norm(center_traj[i] - center_traj[i - 1]) for i in range(1, len(center_traj))]
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(dists)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("||M_t - M_{t-1}||_F")
    ax.set_title(f"Center movement: UC-FCM on {dataset_name}")
    _save(fig, f"center_movement_{dataset_name}")
