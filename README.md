# FTD-MultiModal-Transformer

A research-oriented multimodal deep learning framework for Frontotemporal Dementia (FTD) detection using structural MRI and Transformer-based feature fusion.

## Features

- MRI preprocessing pipeline
- HD-BET skull stripping
- MNI registration
- ROI feature extraction
- Transformer-based multimodal fusion
- Deep learning classification
- Explainable neuroimaging workflow

## Project Structure

.
├── scripts/
├── docs/
├── README.md
├── requirements.txt
└── LICENSE

## Dataset

The dataset is not included due to privacy and storage constraints.

Expected dataset directories:

data/
├── nifti/
├── skull_stripped/
├── mni_registered/
├── roi_features/
├── preprocessed/

## Pipeline

MRI Scan
→ Skull Stripping
→ Registration
→ Feature Extraction
→ Transformer Fusion
→ Classification

## Future Work

- Vision Mamba integration
- Self-supervised MRI pretraining
- Explainable AI visualizations
- Federated medical learning

## Author

Manikandan
