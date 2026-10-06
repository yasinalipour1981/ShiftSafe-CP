"""Quantify distribution shift between laboratory domains."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

DEFAULT_FEATURES = [
    "w_cm",
    "f_cm",
    "cement",
    "fly_ash",
    "slag",
    "water",
    "superplasticizer",
    "coarse_agg",
    "fine_agg",
    "age",
    "strength_mpa",
]


class ShiftAnalyzer:
    """Compute Wasserstein, MMD, and domain-classifier AUC between domains."""

    def __init__(self, device: str = "cuda", seed: int = 42):
        self.device = device
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    def wasserstein_distance(
        self, X: np.ndarray, Y: np.ndarray, n_projections: int = 50
    ) -> float:
        """Sliced Wasserstein-1 distance."""
        distances = []
        for _ in range(n_projections):
            theta = self.rng.standard_normal(X.shape[1])
            theta /= np.linalg.norm(theta) + 1e-12
            X_proj = X @ theta
            Y_proj = Y @ theta
            X_sorted = np.sort(X_proj)
            Y_sorted = np.sort(Y_proj)
            min_size = min(len(X_sorted), len(Y_sorted))
            w_1d = np.mean(np.abs(X_sorted[:min_size] - Y_sorted[:min_size]))
            distances.append(w_1d)
        return float(np.mean(distances))

    def mmd_rbf(
        self, X: torch.Tensor, Y: torch.Tensor, sigma: float = 1.0
    ) -> float:
        """Maximum Mean Discrepancy with RBF kernel (GPU-accelerated)."""
        X_sq = torch.sum(X**2, dim=1, keepdim=True)
        Y_sq = torch.sum(Y**2, dim=1, keepdim=True)

        XX = X_sq + X_sq.T - 2 * torch.mm(X, X.T)
        YY = Y_sq + Y_sq.T - 2 * torch.mm(Y, Y.T)
        XY_dist = X_sq + Y_sq.T - 2 * torch.mm(X, Y.T)

        K_XX = torch.exp(-XX / (2 * sigma**2))
        K_YY = torch.exp(-YY / (2 * sigma**2))
        K_XY = torch.exp(-XY_dist / (2 * sigma**2))

        m, n = X.shape[0], Y.shape[0]
        if m < 2 or n < 2:
            return 0.0

        mmd_sq = (
            (torch.sum(K_XX) - m) / (m * (m - 1))
            + (torch.sum(K_YY) - n) / (n * (n - 1))
            - 2 * torch.sum(K_XY) / (m * n)
        )
        return float(torch.sqrt(mmd_sq.clamp(min=0)))

    def domain_classifier_auc(
        self, X: np.ndarray, domain_labels: np.ndarray
    ) -> float:
        """Train logistic classifier; AUC near 0.5 = weak shift, near 1 = strong."""
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        unique = np.unique(domain_labels)

        if len(unique) < 2:
            return 0.5

        if len(unique) == 2:
            clf = LogisticRegression(max_iter=1000, solver="lbfgs", random_state=self.seed)
            auc_scores = cross_val_score(
                clf, X_scaled, domain_labels, cv=min(5, len(X) // 10 or 2), scoring="roc_auc"
            )
        else:
            clf = LogisticRegression(
                max_iter=1000, solver="lbfgs", multi_class="ovr", random_state=self.seed
            )
            auc_scores = cross_val_score(
                clf,
                X_scaled,
                domain_labels,
                cv=min(5, len(X) // 10 or 2),
                scoring="roc_auc_ovr",
            )
        return float(np.mean(auc_scores))

    def analyze_all_pairs(
        self,
        datasets: dict[str, pd.DataFrame],
        features_to_use: list[str] | None = None,
    ) -> pd.DataFrame:
        """Compute shift metrics for all pairwise domain comparisons."""
        features = features_to_use or DEFAULT_FEATURES
        domains = sorted(datasets.keys())
        results = []
        device = torch.device(
            self.device if self.device == "cpu" or torch.cuda.is_available() else "cpu"
        )

        for i, d_i in enumerate(domains):
            for j, d_j in enumerate(domains):
                if i >= j:
                    continue

                df_i = datasets[d_i]
                df_j = datasets[d_j]
                avail = [f for f in features if f in df_i.columns and f in df_j.columns]
                if not avail:
                    continue

                X_i = df_i[avail].fillna(0).values.astype(np.float64)
                X_j = df_j[avail].fillna(0).values.astype(np.float64)

                scaler = StandardScaler()
                combined = np.vstack([X_i, X_j])
                combined_std = scaler.fit_transform(combined)
                X_i_std = combined_std[: len(X_i)]
                X_j_std = combined_std[len(X_i) :]

                logger.info("Computing shift %s <-> %s...", d_i, d_j)

                w_dist = self.wasserstein_distance(X_i_std, X_j_std)

                X_i_t = torch.tensor(X_i_std, dtype=torch.float32, device=device)
                X_j_t = torch.tensor(X_j_std, dtype=torch.float32, device=device)
                mmd = self.mmd_rbf(X_i_t, X_j_t)

                X_combined = np.vstack([X_i, X_j])
                y_labels = np.hstack(
                    [np.zeros(len(X_i)), np.ones(len(X_j))]
                )
                auc = self.domain_classifier_auc(X_combined, y_labels)

                severity = (
                    "Strong" if auc > 0.75 else "Moderate" if auc > 0.6 else "Weak"
                )
                results.append(
                    {
                        "domain_i": d_i,
                        "domain_j": d_j,
                        "wasserstein_1": w_dist,
                        "mmd_rbf": mmd,
                        "domain_classifier_auc": auc,
                        "shift_severity": severity,
                    }
                )

        df_results = pd.DataFrame(results)
        logger.info("Shift analysis complete: %d pairs", len(df_results))
        return df_results

    def plot_shift_heatmap(
        self,
        df_results: pd.DataFrame,
        metric: str = "wasserstein_1",
        output_path: Path | None = None,
    ):
        """Generate heatmap of pairwise shift metric."""
        domains = sorted(
            set(df_results["domain_i"].unique()) | set(df_results["domain_j"].unique())
        )
        matrix = np.zeros((len(domains), len(domains)))

        for _, row in df_results.iterrows():
            i = domains.index(row["domain_i"])
            j = domains.index(row["domain_j"])
            val = row[metric]
            matrix[i, j] = val
            matrix[j, i] = val

        fig, ax = plt.subplots(figsize=(8, 7))
        sns.heatmap(
            matrix,
            annot=True,
            fmt=".3f",
            cmap="RdYlGn_r",
            xticklabels=domains,
            yticklabels=domains,
            cbar_kws={"label": metric},
            ax=ax,
            vmin=0,
        )
        ax.set_title(
            f"Distribution Shift: {metric}\n(Higher = More Shift)",
            fontsize=12,
            fontweight="bold",
        )
        plt.tight_layout()

        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            logger.info("Saved heatmap to %s", output_path)

        return fig, ax


def run_shift_analysis(
    datasets: dict[str, pd.DataFrame],
    output_dir: Path | str,
    device: str = "cuda",
    seed: int = 42,
) -> pd.DataFrame:
    """Run full shift analysis and save outputs."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    analyzer = ShiftAnalyzer(device=device, seed=seed)
    df_results = analyzer.analyze_all_pairs(datasets)

    df_results.to_csv(output_dir / "domain_classifier_auc.csv", index=False)
    with open(output_dir / "shift_analysis.json", "w", encoding="utf-8") as f:
        json.dump(df_results.to_dict(orient="records"), f, indent=2)

    analyzer.plot_shift_heatmap(
        df_results, metric="wasserstein_1", output_path=output_dir / "wasserstein_heatmap.pdf"
    )

    return df_results
