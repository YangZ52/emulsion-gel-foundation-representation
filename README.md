# Emulsion Gel Foundation Representation Benchmark

This repository provides a lightweight, review-oriented code package for evaluating segmentation-free foundation-model image representations for emulsion-gel rheology prediction from confocal laser scanning microscopy (CLSM) images.

The workflow compares conventional formulation and handcrafted descriptor baselines against frozen foundation-model image embeddings from CLIP, DINOv2, DINOv3, SAM, and SAM2 encoders. Regression models are evaluated using formulation-grouped cross-validation so that replicate images from the same formulation are always kept in the same split.

## Repository Structure

```text
.
├── src/
│   ├── foundation_rheology_benchmark.py
│   └── foundation_pca_mechanism_figure.py
├── examples/
│   ├── foundation_model_variants.csv
│   ├── expected_metadata_columns.csv
│   └── confocal_images/
├── data/
│   └── raw/
│       └── README.md
├── docs/
│   └── reviewer_reproducibility_notes.md
├── requirements.txt
├── LICENSE
└── README.md
```

## What Is Included

This public package includes:

- configurable Python code for grouped-CV rheology prediction;
- code for comparing formulation, descriptor, foundation, and fused feature sets;
- code for PCA-based visualization of foundation latent space;
- a list of foundation-model variants evaluated in the manuscript;
- a template describing the expected metadata columns;
- optional folders for small example CLSM images.

The full raw CLSM dataset, complete rheology workbook, and high-dimensional foundation embeddings are not included here by default because they may be large and are best deposited through a persistent research data repository. The scripts are written so these files can be supplied locally through command-line arguments.

## Data Expected

The benchmark expects an image-level metadata table with one row per CLSM image. The table should include formulation identifiers, rheology measurements, image filenames, and any available handcrafted descriptors. See:

```text
examples/expected_metadata_columns.csv
```

Recommended input organization:

```text
data/raw/
├── emulsion_gel_metadata.csv
├── descriptors.csv
├── embeddings/
│   ├── embeddings_DINOv3-B16.csv
│   └── ...
└── images/
    ├── picture1.tif
    └── ...
```

## Installation

Create an environment and install dependencies:

```bash
pip install -r requirements.txt
```

For embedding extraction with pretrained encoders, install PyTorch and the relevant model packages for your hardware. If embeddings have already been extracted, the prediction scripts only require the tabular embedding CSV files.

## Run the Main Benchmark

```bash
python src/foundation_rheology_benchmark.py \
  --metadata data/raw/emulsion_gel_metadata.csv \
  --embedding-dir data/raw/embeddings \
  --output-dir outputs/foundation_benchmark \
  --target log_Gp \
  --folds 10
```

The script evaluates feature sets such as formulation only, descriptor only, foundation only, formulation + descriptor, formulation + foundation, and all combined features using grouped cross-validation by formulation.

## Generate the PCA Mechanism Figure

```bash
python src/foundation_pca_mechanism_figure.py \
  --metadata data/raw/emulsion_gel_metadata.csv \
  --embedding data/raw/embeddings/embeddings_DINOv3-B16.csv \
  --output-dir outputs/pca_mechanism
```

This figure projects frozen DINOv3-B16 image embeddings into PCA space and compares foundation principal components with physical descriptors and rheology.

## Validation Strategy

All model evaluation is performed at the formulation-group level. Images from the same formulation are never split across training and validation folds. This grouped-CV design is important because multiple CLSM images can correspond to the same independent formulation and rheology measurement.

## Notes for Reviewers and Readers

This repository is intended to document the analysis structure and provide reproducible code templates for the manuscript. Full reproduction requires the associated raw CLSM image dataset, rheology metadata, and/or precomputed embedding tables. Once deposited, the dataset DOI can be added here.

