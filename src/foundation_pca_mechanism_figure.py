#!/usr/bin/env python3
"""PCA mechanism figure for foundation-model CLSM embeddings."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import pearsonr
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


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
LABELS = {
    "n_droplets": "Droplet density",
    "oil_area_fraction": "Oil phase fraction",
    "mean_radius": "Mean droplet radius",
    "radius_cv": "Droplet size polydispersity",
    "mean_normalized_NND": "Size-normalized NND",
    "mean_shell_green": "Interfacial protein intensity",
    "mean_eccentricity": "Droplet anisotropy",
    "protein_connection_density": "Protein network connectivity",
    "protein_fractal_dimension": "Protein network complexity",
    "log_Gp": "G' at 1 Hz",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", required=True, type=Path, help="Metadata/descriptor CSV.")
    parser.add_argument("--embedding", required=True, type=Path, help="Single embeddings_*.csv file.")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--target", default="log_Gp")
    parser.add_argument("--model-label", default="DINOv3-B16")
    return parser.parse_args()


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    aliases = {
        "protein type": "protein_type",
        "protein conc_percentage": "protein_concentration",
        "oil vol_percentage": "oil_phase_fraction",
    }
    return df.rename(columns={k: v for k, v in aliases.items() if k in df.columns})


def load_data(metadata_path: Path, embedding_path: Path) -> tuple[pd.DataFrame, list[str]]:
    metadata = normalize_columns(pd.read_csv(metadata_path))
    emb = normalize_columns(pd.read_csv(embedding_path))
    join_cols = [c for c in ["image_file", "formulation_group"] if c in metadata.columns and c in emb.columns]
    if not join_cols:
        raise ValueError("Metadata and embedding table need image_file and/or formulation_group for joining.")
    data = metadata.merge(emb, on=join_cols, how="inner")
    emb_cols = [c for c in data.columns if c.startswith("foundation_")]
    if not emb_cols:
        raise ValueError("Embedding table must contain foundation_* columns.")
    return data, emb_cols


def compute_pca(data: pd.DataFrame, emb_cols: list[str]) -> tuple[pd.DataFrame, np.ndarray]:
    X = data[emb_cols].astype(float).to_numpy()
    n_components = min(20, X.shape[0] - 1, X.shape[1])
    pipe = Pipeline([("scale", StandardScaler()), ("pca", PCA(n_components=n_components, random_state=63))])
    pcs = pipe.fit_transform(X)
    for i in range(pcs.shape[1]):
        data[f"PC{i + 1}"] = pcs[:, i]
    return data, pipe.named_steps["pca"].explained_variance_ratio_


def pc_correlations(data: pd.DataFrame, target: str) -> pd.DataFrame:
    rows = []
    variables = [c for c in DESCRIPTOR_COLUMNS if c in data.columns] + [target]
    for pc_idx in range(1, 9):
        pc = f"PC{pc_idx}"
        for variable in variables:
            vals = data[[pc, variable]].dropna()
            if len(vals) > 2:
                r, p = pearsonr(vals[pc], vals[variable])
                rows.append({"PC": pc, "variable": variable, "label": LABELS.get(variable, variable), "pearson_r": r, "p_value": p})
    return pd.DataFrame(rows)


def scatter_panel(fig: plt.Figure, ax: plt.Axes, data: pd.DataFrame, color_col: str, title: str, cbar_label: str) -> None:
    sc = ax.scatter(data["PC1"], data["PC2"], c=data[color_col], cmap="viridis", s=34, alpha=0.9, edgecolor="white", linewidth=0.3)
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel("Foundation PC1", fontweight="bold")
    ax.set_ylabel("Foundation PC2", fontweight="bold")
    ax.grid(True, color="0.88", linewidth=0.6)
    cb = fig.colorbar(sc, ax=ax, shrink=0.82, pad=0.025)
    cb.set_label(cbar_label, fontweight="bold")


def plot(data: pd.DataFrame, corr: pd.DataFrame, explained: np.ndarray, args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="paper")
    plt.rcParams.update({"pdf.fonttype": 42, "font.family": "sans-serif", "font.size": 10})
    fig = plt.figure(figsize=(11.5, 13.5))
    gs = fig.add_gridspec(3, 2, wspace=0.25, hspace=0.32)
    axes = [fig.add_subplot(gs[i, j]) for i in range(3) for j in range(2)]
    scatter_panel(fig, axes[0], data, args.target, "A. Foundation PCA colored by G' at 1 Hz", "log10 G' at 1 Hz")
    if "oil_area_fraction" in data.columns:
        scatter_panel(fig, axes[1], data, "oil_area_fraction", "B. Oil phase fraction", "Oil phase fraction")
    if "protein_connection_density" in data.columns:
        scatter_panel(fig, axes[2], data, "protein_connection_density", "C. Protein network connectivity", "Protein connectivity")
    if "mean_normalized_NND" in data.columns:
        scatter_panel(fig, axes[3], data, "mean_normalized_NND", "D. Size-normalized NND", "Size-normalized NND")

    heat = corr.pivot(index="label", columns="PC", values="pearson_r").reindex(columns=[f"PC{i}" for i in range(1, 9)])
    sns.heatmap(heat, ax=axes[4], cmap="RdBu_r", vmin=-1, vmax=1, center=0, annot=True, fmt=".2f", linewidths=0.5, cbar_kws={"label": "Pearson r"})
    axes[4].set_title("E. Foundation PCs correlate with physical descriptors", fontweight="bold")
    axes[4].set_xlabel("")
    axes[4].set_ylabel("")
    axes[4].tick_params(axis="x", rotation=0)

    pcs = np.arange(1, len(explained) + 1)
    axes[5].bar(pcs, explained * 100, color="#5B8DB8", label="Individual PC variance")
    axes[5].plot(pcs, np.cumsum(explained) * 100, color="#B45A4A", marker="o", label="Cumulative variance")
    axes[5].set_title(f"F. Variance captured by {args.model_label} PCs", fontweight="bold")
    axes[5].set_xlabel("Foundation PCA component", fontweight="bold")
    axes[5].set_ylabel("Explained variance (%)", fontweight="bold")
    axes[5].legend(frameon=True, loc="center right")
    fig.savefig(args.output_dir / "foundation_pca_mechanism_figure.pdf", bbox_inches="tight")
    fig.savefig(args.output_dir / "foundation_pca_mechanism_figure.png", dpi=600, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data, emb_cols = load_data(args.metadata, args.embedding)
    data, explained = compute_pca(data, emb_cols)
    corr = pc_correlations(data, args.target)
    data.to_csv(args.output_dir / "foundation_pca_scores.csv", index=False)
    corr.to_csv(args.output_dir / "foundation_pc_descriptor_correlations.csv", index=False)
    plot(data, corr, explained, args)
    print(f"Saved PCA mechanism outputs to: {args.output_dir}")


if __name__ == "__main__":
    main()
