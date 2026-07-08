"""
Reproduces:
  Experiment 1: FCM vs UC-FCM, 20 runs, same initialization per run
                (Mean/Min/Max objective, wins/ties/losses, NMI/ACC/ARI mean+std)
  Experiment 2: Sensitivity analysis over r in {1.1,...,2.0}, alpha in
                {0.001,0.005,0.01,0.05,0.1}
  Experiment 3: Convergence curves (objective vs iteration) for FCM vs UC-FCM
  Experiment 4: Runtime comparison

"""

import os
import json
import time
import argparse
import numpy as np
import pandas as pd

from algorithms import FCM, UCFCM, init_membership
from evaluation import evaluate_clustering, friedman_test, nemenyi_cd, wilcoxon_test
from datasets import load_dataset, DATASET_LOADERS
import visualization as viz


RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
TABLES_DIR = os.path.join(os.path.dirname(__file__), "tables")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(TABLES_DIR, exist_ok=True)

R_GRID = [round(1.1 + 0.1 * i, 1) for i in range(10)]     # {1.1,...,2.0}
ALPHA_GRID = [0.001, 0.005, 0.01, 0.05, 0.1]
GAMMA = 0.6
N_RUNS = 20
TOL = 1e-5


def grid_search_best_params(X, y, n_clusters, seeds):
    """Grid search r in R_GRID, alpha in ALPHA_GRID for UC-FCM, selecting the
    (r, alpha) pair with the best mean ACC across `seeds` random inits (paper,
    Section IV-A: 'optimal values for r and alpha are determined using grid search')."""
    best = None
    grid_acc = np.zeros((len(R_GRID), len(ALPHA_GRID)))
    grid_obj = np.zeros((len(R_GRID),))
    obj_counts = np.zeros((len(R_GRID),))

    for ri, r in enumerate(R_GRID):
        objs_this_r = []
        for ai, alpha in enumerate(ALPHA_GRID):
            accs = []
            for seed in seeds:
                Y0 = init_membership(X.shape[0], n_clusters, seed)
                model = UCFCM(n_clusters, r=r, alpha=alpha, gamma=GAMMA, tol=TOL)
                model.fit(X, Y0)
                metrics = evaluate_clustering(y, model.labels_)
                accs.append(metrics["ACC"])
                objs_this_r.append(model.objective_)
            mean_acc = float(np.mean(accs))
            grid_acc[ri, ai] = mean_acc
            if best is None or mean_acc > best[0]:
                best = (mean_acc, r, alpha)
        grid_obj[ri] = float(np.mean(objs_this_r))

    return best, grid_acc, grid_obj


def experiment1(name, X, y, n_clusters, r, alpha, gamma=GAMMA, n_runs=N_RUNS):
    """FCM vs UC-FCM, same initialization each run."""
    fcm_objs, uc_objs = [], []
    fcm_metrics, uc_metrics = [], []
    wins = ties = losses = 0
    fcm_time_total = uc_time_total = 0.0
    example_hist = None

    for run in range(n_runs):
        seed = 1000 + run
        Y0 = init_membership(X.shape[0], n_clusters, seed)

        t0 = time.perf_counter()
        fcm = FCM(n_clusters, r=r, tol=TOL).fit(X, Y0)
        fcm_time_total += time.perf_counter() - t0

        t0 = time.perf_counter()
        ucfcm = UCFCM(n_clusters, r=r, alpha=alpha, gamma=gamma, tol=TOL).fit(X, Y0)
        uc_time_total += time.perf_counter() - t0

        fcm_objs.append(fcm.objective_)
        uc_objs.append(ucfcm.objective_)

        if ucfcm.objective_ < fcm.objective_ - 1e-8:
            wins += 1
        elif abs(ucfcm.objective_ - fcm.objective_) <= 1e-8:
            ties += 1
        else:
            losses += 1

        fcm_metrics.append(evaluate_clustering(y, fcm.labels_))
        uc_metrics.append(evaluate_clustering(y, ucfcm.labels_))

        if run == 0:
            example_hist = (fcm.history_, ucfcm.history_)

    def agg(metrics_list, key):
        arr = np.array([m[key] for m in metrics_list])
        return arr.mean(), arr.std()

    row = {
        "Dataset": name,
        "Mean_obj_FCM": np.mean(fcm_objs), "Mean_obj_UCFCM": np.mean(uc_objs),
        "Min_obj_FCM": np.min(fcm_objs), "Min_obj_UCFCM": np.min(uc_objs),
        "Max_obj_FCM": np.max(fcm_objs), "Max_obj_UCFCM": np.max(uc_objs),
        "Runtime_FCM_s": fcm_time_total / n_runs, "Runtime_UCFCM_s": uc_time_total / n_runs,
    }
    for key in ["NMI", "ACC", "ARI"]:
        m_f, s_f = agg(fcm_metrics, key)
        m_u, s_u = agg(uc_metrics, key)
        row[f"{key}_FCM_mean"], row[f"{key}_FCM_std"] = m_f, s_f
        row[f"{key}_UCFCM_mean"], row[f"{key}_UCFCM_std"] = m_u, s_u

    # Wilcoxon signed-rank test on paired objective values across the 20 runs
    try:
        stat, p = wilcoxon_test(fcm_objs, uc_objs)
        row["Wilcoxon_stat"], row["Wilcoxon_p"] = stat, p
    except Exception as e:
        row["Wilcoxon_stat"], row["Wilcoxon_p"] = np.nan, np.nan

    return row, example_hist, (fcm_objs, uc_objs), (fcm_metrics, uc_metrics)


def experiment2(name, X, y, n_clusters, seeds):
    """Sensitivity analysis over r and alpha (Fig. 3, Fig. 4 style)."""
    grid_acc = np.zeros((len(R_GRID), len(ALPHA_GRID)))
    obj_by_r = []
    for ri, r in enumerate(R_GRID):
        objs_r = []
        for ai, alpha in enumerate(ALPHA_GRID):
            accs = []
            for seed in seeds:
                Y0 = init_membership(X.shape[0], n_clusters, seed)
                model = UCFCM(n_clusters, r=r, alpha=alpha, gamma=GAMMA, tol=TOL).fit(X, Y0)
                accs.append(evaluate_clustering(y, model.labels_)["ACC"])
                objs_r.append(model.objective_)
            grid_acc[ri, ai] = np.mean(accs)
        obj_by_r.append(np.mean(objs_r))
    return grid_acc, obj_by_r


def run_all(dataset_names, n_runs=N_RUNS, sensitivity_seeds=5, do_sensitivity=True):
    summary_rows = []
    all_histories = {}
    per_dataset_details = {}

    for name in dataset_names:
        print(f"\n=== Dataset: {name} ===")
        X, y, label = load_dataset(name)
        n_clusters = len(np.unique(y))
        print(f"  shape={X.shape}, n_clusters={n_clusters}, loader_label={label}")

        # NOTE: paper does not specify how many seeds the grid search itself uses
        # (only that r,alpha are "determined using grid search"); we use a small
        # number of seeds here purely to keep grid-search cost tractable. For
        # datasets with many samples/clusters, fewer seeds keeps runtime reasonable
        # without changing the grid itself (r in {1.1..2.0}, alpha in {.001..0.1}).
        n_search_seeds = 5 if (X.shape[0] * n_clusters) < 5000 else 2
        seeds_for_search = list(range(n_search_seeds))
        best, grid_acc_search, _ = grid_search_best_params(X, y, n_clusters, seeds_for_search)
        best_acc, best_r, best_alpha = best
        print(f"  Grid search selected r={best_r}, alpha={best_alpha} (mean ACC={best_acc:.4f})")

        row, hist, objs, metrics = experiment1(name, X, y, n_clusters, best_r, best_alpha, n_runs=n_runs)
        row["best_r"], row["best_alpha"] = best_r, best_alpha
        summary_rows.append(row)
        all_histories[name] = hist

        if do_sensitivity:
            grid_acc, obj_by_r = experiment2(name, X, y, n_clusters, seeds_for_search[:2])
            viz.plot_sensitivity_heatmap(grid_acc, R_GRID, ALPHA_GRID, name)
            viz.plot_objective_vs_r(obj_by_r, R_GRID, name)

        viz.plot_convergence(hist[0], hist[1], name)
        viz.plot_runtime_bars({"FCM": row["Runtime_FCM_s"], "UC-FCM": row["Runtime_UCFCM_s"]}, name)

        fcm_objs, uc_objs = objs
        viz.plot_boxplot_over_runs({"FCM": fcm_objs, "UC-FCM": uc_objs}, name, "Objective")

        fcm_metrics, uc_metrics = metrics
        for key in ["NMI", "ACC", "ARI"]:
            viz.plot_boxplot_over_runs(
                {"FCM": [m[key] for m in fcm_metrics], "UC-FCM": [m[key] for m in uc_metrics]},
                name, key,
            )

        # PCA / confusion matrix using one representative run
        Y0 = init_membership(X.shape[0], n_clusters, 1000)
        fcm_final = FCM(n_clusters, r=best_r, tol=TOL).fit(X, Y0)
        uc_final = UCFCM(n_clusters, r=best_r, alpha=best_alpha, gamma=GAMMA, tol=TOL).fit(X, Y0)
        try:
            viz.plot_pca(X, y, fcm_final.labels_, uc_final.labels_, name)
        except Exception as e:
            print(f"  [viz] PCA failed for {name}: {e}")
        try:
            viz.plot_confusion_matrix(y, fcm_final.labels_, name, "FCM")
            viz.plot_confusion_matrix(y, uc_final.labels_, name, "UCFCM")
        except Exception as e:
            print(f"  [viz] confusion matrix failed for {name}: {e}")

        per_dataset_details[name] = row

    viz.plot_convergence_grid(all_histories, dataset_names)

    df = pd.DataFrame(summary_rows)
    df.to_csv(os.path.join(RESULTS_DIR, "experiment1_summary.csv"), index=False)

    # Friedman test across datasets (NMI/ACC/ARI), comparing FCM vs UC-FCM only
    # (full 10-algorithm comparison requires implementing the 8 other baselines,
    # which is out of scope for this reproduction focused on FCM vs UC-FCM; see README).
    for metric in ["NMI", "ACC", "ARI"]:
        mat = df[[f"{metric}_FCM_mean", f"{metric}_UCFCM_mean"]].values
        try:
            chi2, p, ranks = friedman_test(mat)
            cd = nemenyi_cd(K=2, N=len(df))
            print(f"[Friedman] {metric}: chi2={chi2:.4f}, p={p:.4f}, mean_ranks(FCM,UCFCM)={ranks}, CD={cd:.3f}")
            with open(os.path.join(RESULTS_DIR, f"friedman_{metric}.json"), "w") as f:
                json.dump({"chi2": chi2, "p": p, "mean_ranks": ranks.tolist(), "CD": cd}, f, indent=2)
        except Exception as e:
            print(f"[Friedman] failed for {metric}: {e}")

    generate_table3(df)
    return df


def generate_table3(df):
    """Reproduces the layout of Table III (Comparison between FCM and UC-FCM)."""
    cols = [
        "Dataset",
        "Mean_obj_FCM", "Mean_obj_UCFCM",
        "Min_obj_FCM", "Min_obj_UCFCM",
        "Max_obj_FCM", "Max_obj_UCFCM",
        "NMI_FCM_mean", "NMI_UCFCM_mean",
        "ACC_FCM_mean", "ACC_UCFCM_mean",
        "ARI_FCM_mean", "ARI_UCFCM_mean",
    ]
    table = df[cols].copy()
    table.to_csv(os.path.join(TABLES_DIR, "table3_fcm_vs_ucfcm.csv"), index=False)
    try:
        with open(os.path.join(TABLES_DIR, "table3_fcm_vs_ucfcm.tex"), "w") as f:
            f.write(table.to_latex(index=False, float_format="%.4f"))
    except Exception as e:
        print(f"[tables] LaTeX export failed: {e}")
    print("\n=== Table III reproduction (FCM vs UC-FCM) ===")
    print(table.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=list(DATASET_LOADERS.keys()))
    parser.add_argument("--n_runs", type=int, default=N_RUNS)
    parser.add_argument("--no_sensitivity", action="store_true")
    args = parser.parse_args()

    run_all(args.datasets, n_runs=args.n_runs, do_sensitivity=not args.no_sensitivity)
