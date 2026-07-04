#!/usr/bin/env python3
"""Grouped-CV benchmark for emulsion-gel rheology prediction.

This script compares formulation variables, handcrafted CLSM descriptors, and
precomputed foundation-model image embeddings for log10 rheology prediction.
All cross-validation splits are grouped by formulation to avoid leakage among
replicate images from the same formulation.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.base import clone
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVR


FORMULATION_CATEGORICAL = ["protein_type", "heat"]
FORMULATION_NUMERIC = ["protein_concentration", "oil_phase_fraction", "pH", "NaCl_mM", "CaCl2_mM"]
DESCRIPTOR_COLUMNS = [
    "n_droplets",
    "oil_area_fraction",
    "mean_radius",
    "radius_cv",
    "mean_normalized_NND",
    "mean_shell_green",
    "mean_eccentricity",
    "protein_connection_density",
    "protein_fractal_dimension",
]
FEATURE_SETS = [
    "Formulation only",
    "Descriptor only",
    "Foundation only",
    "Formulation + descriptor",
    "Formulation + foundation",
    "All",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", required=True, type=Path, help="Image-level metadata CSV.")
    parser.add_argument("--embedding-dir", required=True, type=Path, help="Directory containing embeddings_*.csv files.")
    parser.add_argument("--output-dir", required=True, type=Path, help="Output directory.")
    parser.add_argument("--target", default="log_Gp", help="Target column, e.g. log_Gp or log_breaking_stress.")
    parser.add_argument("--group-col", default="formulation_group", help="Independent formulation group column.")
    parser.add_argument("--folds", default=10, type=int, help="Maximum grouped CV folds.")
    parser.add_argument("--pca-components", default=32, type=int, help="Maximum foundation PCA components inside each fold.")
    parser.add_argument("--seed", default=63, type=int)
    return parser.parse_args()


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    aliases = {
        "protein type": "protein_type",
        "protein conc_percentage": "protein_concentration",
        "oil vol_percentage": "oil_phase_fraction",
        "G' 1Hz": "Gp_1Hz_Pa",
    }
    return df.rename(columns={k: v for k, v in aliases.items() if k in df.columns})


def log_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = np.maximum(np.abs(y_true), 1e-9)
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100)


def formulation_matrix(df: pd.DataFrame) -> np.ndarray:
    blocks = []
    cat_cols = [c for c in FORMULATION_CATEGORICAL if c in df.columns]
    num_cols = [c for c in FORMULATION_NUMERIC if c in df.columns]
    if cat_cols:
        enc = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        blocks.append(enc.fit_transform(df[cat_cols].astype(str)))
    if num_cols:
        blocks.append(df[num_cols].astype(float).to_numpy())
    return np.column_stack(blocks) if blocks else np.empty((len(df), 0))


def descriptor_matrix(df: pd.DataFrame) -> np.ndarray:
    cols = [c for c in DESCRIPTOR_COLUMNS if c in df.columns]
    return df[cols].astype(float).to_numpy() if cols else np.empty((len(df), 0))


def load_embeddings(metadata: pd.DataFrame, path: Path) -> tuple[pd.DataFrame, list[str]]:
    emb = pd.read_csv(path)
    emb = normalize_columns(emb)
    join_cols = [c for c in ["image_file", "formulation_group"] if c in emb.columns and c in metadata.columns]
    if not join_cols:
        raise ValueError(f"No shared join columns found for {path.name}.")
    data = metadata.merge(emb, on=join_cols, how="inner")
    emb_cols = [c for c in data.columns if c.startswith("foundation_")]
    if not emb_cols:
        raise ValueError(f"No foundation_* embedding columns found in {path.name}.")
    return data, emb_cols


def stratification_labels(df: pd.DataFrame, target: str, group_col: str) -> np.ndarray:
    group_mean = df.groupby(group_col)[target].mean()
    n_bins = min(5, group_mean.nunique())
    bins = pd.qcut(group_mean.rank(method="first"), q=n_bins, labels=False, duplicates="drop")
    return df[group_col].map(bins).astype(int).to_numpy()


def build_fold_features(
    df: pd.DataFrame,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    feature_set: str,
    emb_cols: list[str],
    max_pca: int,
) -> tuple[np.ndarray, np.ndarray]:
    train_blocks, test_blocks = [], []
    if feature_set in {"Formulation only", "Formulation + descriptor", "Formulation + foundation", "All"}:
        X = formulation_matrix(df)
        train_blocks.append(X[train_idx])
        test_blocks.append(X[test_idx])
    if feature_set in {"Descriptor only", "Formulation + descriptor", "All"}:
        X = descriptor_matrix(df)
        train_blocks.append(X[train_idx])
        test_blocks.append(X[test_idx])
    if feature_set in {"Foundation only", "Formulation + foundation", "All"}:
        X_train = df.iloc[train_idx][emb_cols].astype(float).to_numpy()
        X_test = df.iloc[test_idx][emb_cols].astype(float).to_numpy()
        n_components = max(2, min(max_pca, X_train.shape[0] - 1, X_train.shape[1]))
        pca_pipe = Pipeline([("scale", StandardScaler()), ("pca", PCA(n_components=n_components, random_state=63))])
        train_blocks.append(pca_pipe.fit_transform(X_train))
        test_blocks.append(pca_pipe.transform(X_test))
    return np.column_stack(train_blocks), np.column_stack(test_blocks)


def model_library(seed: int) -> dict[str, object]:
    kernel = ConstantKernel(1.0, (1e-2, 1e3)) * RBF(1.0, (1e-2, 1e3)) + WhiteKernel(0.05, (1e-6, 1.0))
    return {
        "RBF-GPR": GaussianProcessRegressor(
            kernel=kernel,
            normalize_y=True,
            alpha=1e-6,
            n_restarts_optimizer=1,
            random_state=seed,
        ),
        "SVR": SVR(kernel="rbf", C=10.0, epsilon=0.05),
        "Random Forest": RandomForestRegressor(n_estimators=300, random_state=seed, min_samples_leaf=2),
        "Ridge": Ridge(alpha=1.0),
    }


def evaluate(data: pd.DataFrame, emb_cols: list[str], args: argparse.Namespace, foundation_name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    y = data[args.target].astype(float).to_numpy()
    groups = data[args.group_col].astype(str).to_numpy()
    labels = stratification_labels(data, args.target, args.group_col)
    n_splits = min(args.folds, pd.Series(groups).nunique())
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=args.seed)
    summary_rows, pred_rows = [], []

    for feature_set in FEATURE_SETS:
        for model_name, base_model in model_library(args.seed).items():
            pred = np.full(len(y), np.nan)
            for fold, (train_idx, test_idx) in enumerate(cv.split(data, labels, groups), start=1):
                X_train, X_test = build_fold_features(data, train_idx, test_idx, feature_set, emb_cols, args.pca_components)
                pipe = Pipeline([("scale", StandardScaler()), ("model", clone(base_model))])
                pipe.fit(X_train, y[train_idx])
                pred[test_idx] = pipe.predict(X_test)
                for i in test_idx:
                    pred_rows.append(
                        {
                            "foundation_model": foundation_name,
                            "feature_set": feature_set,
                            "model": model_name,
                            "fold": fold,
                            "image_file": data.iloc[i].get("image_file", ""),
                            "formulation_group": data.iloc[i][args.group_col],
                            "y_true": y[i],
                            "y_pred": pred[i],
                        }
                    )
            keep = np.isfinite(pred)
            summary_rows.append(
                {
                    "foundation_model": foundation_name,
                    "feature_set": feature_set,
                    "model": model_name,
                    "R2": r2_score(y[keep], pred[keep]),
                    "RMSE": math.sqrt(mean_squared_error(y[keep], pred[keep])),
                    "MAE": mean_absolute_error(y[keep], pred[keep]),
                    "MAPE_log_percent": log_mape(y[keep], pred[keep]),
                    "n_images": int(keep.sum()),
                    "n_formulation_groups": int(pd.Series(groups[keep]).nunique()),
                }
            )
    return pd.DataFrame(summary_rows), pd.DataFrame(pred_rows)


def plot_heatmap(summary: pd.DataFrame, output_dir: Path, metric: str, model: str = "RBF-GPR") -> None:
    sub = summary[summary["model"].eq(model)].copy()
    pivot = sub.pivot_table(index="foundation_model", columns="feature_set", values=metric, aggfunc="mean")
    cols = [c for c in FEATURE_SETS if c in pivot.columns]
    pivot = pivot[cols]
    fig, ax = plt.subplots(figsize=(max(8, len(cols) * 1.4), max(5, len(pivot) * 0.34)))
    cmap = "YlGnBu" if metric == "R2" else "YlOrRd_r"
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap=cmap, linewidths=0.5, linecolor="white", ax=ax)
    ax.set_xlabel("")
    ax.set_ylabel("Foundation model")
    ax.set_title(f"{model} grouped-CV {metric}")
    ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    fig.savefig(output_dir / f"heatmap_{model.replace(' ', '_')}_{metric}.pdf")
    fig.savefig(output_dir / f"heatmap_{model.replace(' ', '_')}_{metric}.png", dpi=600)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metadata = normalize_columns(pd.read_csv(args.metadata))
    all_summary, all_predictions = [], []
    for emb_path in sorted(args.embedding_dir.glob("embeddings_*.csv")):
        foundation_name = emb_path.stem.replace("embeddings_", "")
        data, emb_cols = load_embeddings(metadata, emb_path)
        summary, predictions = evaluate(data, emb_cols, args, foundation_name)
        all_summary.append(summary)
        all_predictions.append(predictions)
    summary_df = pd.concat(all_summary, ignore_index=True)
    pred_df = pd.concat(all_predictions, ignore_index=True)
    summary_df.to_csv(args.output_dir / "foundation_benchmark_summary.csv", index=False)
    pred_df.to_csv(args.output_dir / "foundation_benchmark_predictions.csv", index=False)
    plot_heatmap(summary_df, args.output_dir, "R2")
    plot_heatmap(summary_df, args.output_dir, "RMSE")
    print(f"Saved benchmark outputs to: {args.output_dir}")


if __name__ == "__main__":
    main()
