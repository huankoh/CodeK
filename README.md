### CodeK: Contrastive Optimization and Distillation for Exiguous Knowledge

CodeK is an interpretable deep learning framework tailored for data-limited biomedical settings. It unites a Mixer-based encoder with a dual-branch training scheme that combines supervised contrastive learning and classification. CodeK also supports robust, model-agnostic feature attribution for clinician-facing interpretability.

### Repository structure

- **main.py**: Train/evaluate CodeK on supported datasets
- **model.py**: MLPMixer encoder, supervised contrastive loss, downstream classifier
- **trainer.py**: Training loop, ROC/AUC evaluation, model checkpointing
- **data.py**: CSV loading, encoding, imputation, scaling, split, augmentation utilities
- **baseline.py**: Logistic Regression, Random Forest, SVM baselines
- **interpretation.py**: LIME-based feature attribution for saved/pretrained models
- **scripts/**: Example commands for datasets (`mimic.sh`, `framingham.sh`, `eicu.sh`, `wgs_group2.sh`, `wgs_group3.sh`)
- **data/**: Example CSVs used in the paper/experiments

### Requirements

- Python 3.8+
- PyTorch (tested with recent 1.x/2.x)
- scikit-learn, numpy, pandas, matplotlib, seaborn, tqdm, scipy, lime

Install example:

```bash
python -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install torch torchvision torchaudio \
    numpy pandas scikit-learn matplotlib seaborn tqdm scipy lime
```

### Datasets

CSV datasets are expected under `data/`. Please download all the CSVs in the [Google Drive](https://drive.google.com/drive/folders/1KkgMHpsJLUp3bFpaZm29FOuhUoCmqoFo?usp=sharing) and put it into the `data/` folder for quick runs:

- **eicu**: `data/eicu_cohort_modify.csv` (label: `aki_stage`)
- **mimic**: `data/mimic.csv` (label: `aki_stage`)
- **framingham**: `data/framingham.csv` (label: `TenYearCHD`)
- **wgs_group1**: `data/wgs_dummy_dataset.csv` (label: `Y`)
- **wgs_group2**: `data/group2_dummy_dataset.csv` (label: `Y`)
- **wgs_group3**: `data/group3_dummy_dataset.csv` (label: `Y`)

Data preprocessing (encoding, drop, scale, imputation) is configured inside `main.py` / `data.py` per dataset. To add new datasets, mirror this pattern: specify `label_feature`, which columns to encode/drop/scale, and any dataset-specific normalization/binning.

### Quickstart

From the project root:

```bash
python main.py --data framingham --device cpu \
  --train_prop 0.7 --test_prop 0.3 \
  --batch_size 512 --epoch 200 --lr 5e-4 \
  --base_tau 0.8 --tau 0.1 --hid_dim 150 --feat_hid_dim 300 --down_hid_dim 16 \
  --dropout 0.3 --patch 1 --num_block 1 --seed 2
```

Example runs (see `scripts/`)

Notes:

- Use `--device cuda` to utilize a GPU if available.
- The script prints training loss and validation AUC; the best model is saved as `best_model.pt` in the project root.

### Command-line arguments (main)

- **--data**: one of `eicu | mimic | framingham | wgs_group1 | wgs_group2 | wgs_group3` (required)
- **--train_prop / --test_prop**: train/test split fractions
- **--epoch, --batch_size, --lr, --weight_decay**: training hyperparameters
- **--hid_dim, --feat_hid_dim, --down_hid_dim**: embedding, contrastive, and downstream dims
- **--base_tau, --tau**: temperatures for supervised contrastive loss
- **--dropout**: dropout in encoders
- **--patch, --num_block**: Mixer encoder patching and block count
- **--seed**: random seed; see `utils.set_random_seed`


### Data augmentation

CodeK includes a class-wise GMM augmentation utility in `data.augmentation`. The CLI exposes `--aug`, `--aug_size`, and `--n_component` flags; the augmentation call in `main.py` is currently commented out. To use augmentation, enable the augmentation block in `main.py` and pass the flags shown in the example scripts.

### Interpretability (LIME)

You can generate instance-level feature attributions using `interpretation.py` with a saved or pretrained model:

```bash
python interpretation.py --data mimic --train_prop 0.7 --test_prop 0.3 --num_samples 10 --random_seed 0
```

By default, `interpretation.py` loads `./pretrained/mimic_best_model_seed3.pt`. To use your own trained model, change `load_model()` to point to your checkpoint (e.g., `best_model.pt`). The script creates `lime_explanation_sample_*.html` files with per-instance explanations.


### Using your own data

To adapt CodeK to a new CSV dataset:

1) Add your CSV to `data/`.
2) Define dataset-specific preprocessing in `main.py` (encoding/drop/scale lists and `label_feature`).
3) Update CLI choices if you add a new dataset key.
4) Verify feature dimensions match the network input size.