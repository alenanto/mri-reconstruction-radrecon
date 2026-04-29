# RadRecon Usage Guide

## Workflow

### 1. Data Preprocessing

First, preprocess the raw fastMRI dataset:

python scripts/fastMRI_breast_preprocessing.py \
    --dataset-path data/fastMRI_org \
    --slice-indices-path data/slice_indices.xlsx \
    --output-path data/fastMRI_processed

---

### 2. Training

Configure the training parameters in the configuration files (configs/).

Then run:

python src/train.py

---

### 3. Prediction

Set the checkpoint path in the configuration file:

ckpt_path: /path/to/your/checkpoint.ckpt

Then run:

python src/predict.py

---

## Notes

- Ensure hyperparameters are set correctly before training
- Prediction requires a trained checkpoint
- Preprocessing is mandatory before training