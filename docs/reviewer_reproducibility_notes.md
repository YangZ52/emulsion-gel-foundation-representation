# Reviewer Reproducibility Notes

This repository provides the analysis structure used for the foundation-representation benchmark.

Key reproducibility points:

- each CLSM image is linked to a formulation group;
- grouped cross-validation is performed by formulation group;
- replicate images from the same formulation are never divided between train and validation folds;
- foundation encoders are used as frozen image feature extractors;
- SAM and SAM2 are evaluated through their image encoders only, not their segmentation decoders;
- downstream regressors are trained on fixed tabular feature representations;
- rheological targets are modeled in log10 space.

The complete raw image dataset and full embedding tables can be large. For full reproduction, use the associated dataset DOI or provide equivalent local paths through the script command-line arguments.

