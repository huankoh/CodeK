# CodeK

**Contrastive Optimization and Distillation for Exiguous Knowledge for Data-Limited Biomedical Applications**

CodeK is an interpretable deep-learning framework for small and imbalanced biomedical datasets. It combines class-conditional data augmentation, a Mixer-based encoder, supervised contrastive learning, classification, and post hoc feature attribution.

## Archival status

The permanent, citable version of the code and accompanying reproducibility materials is deposited in Zenodo: [https://doi.org/10.5281/zenodo.21487356](https://doi.org/10.5281/zenodo.21487356).

The GitHub repository at <https://github.com/huankoh/CodeK> is an additional development and distribution resource. GitHub is not the archival version of record.

## Repository layout

- `main.py`: train and evaluate CodeK on supported datasets.
- `model.py`: Mixer encoder, supervised contrastive loss, and classifier.
- `trainer.py`: training loop, ROC/AUC evaluation, and checkpointing.
- `data.py`: loading, encoding, imputation, scaling, splitting, and augmentation.
- `baseline.py`: logistic regression, random forest, and SVM baselines.
- `interpretation*.py`: LIME-based attribution and feature-ranking utilities.
- `scripts/`: example dataset-specific commands and the archive verifier.
- `paper_figure_plots/`: notebooks and source tables used to create manuscript figures.
- `data/`: data manifest and access instructions; restricted clinical data are not included.
- `DATA_AVAILABILITY.md`: provenance, access conditions, and redistribution status for each dataset.

## Installation

Python 3.8 or newer is recommended. Create an isolated environment and install the declared dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The original experiments used PyTorch 1.12. Later PyTorch 1.x and 2.x versions may also work, but should be recorded in any rerun report.

## Data setup

The data files are not bundled indiscriminately with the source code. MIMIC and eICU records are credentialed datasets and must never be redistributed through GitHub or a public general-purpose archive. Other third-party datasets retain their source licenses and access conditions.

Follow [`DATA_AVAILABILITY.md`](DATA_AVAILABILITY.md) and [`data/README.md`](data/README.md) to obtain the inputs and place them under `data/` using the expected filenames.

## Quick start

From the repository root:

```bash
python main.py --data framingham --device cpu \
  --train_prop 0.7 --test_prop 0.3 \
  --batch_size 512 --epoch 200 --lr 5e-4 \
  --base_tau 0.8 --tau 0.1 --hid_dim 150 --feat_hid_dim 300 \
  --down_hid_dim 16 --dropout 0.3 --patch 1 --num_block 1 --seed 2
```

Supported command-line dataset keys are `eicu`, `mimic`, `framingham`, `wgs_group1`, `wgs_group2`, and `wgs_group3`. The manuscript figure notebooks document additional notebook-based analyses.

The CLI exposes `--aug`, `--aug_size`, and `--n_component`. In this snapshot, the augmentation call in `main.py` is commented out; enable that block before requesting an augmented training run and record the change in the run metadata.

## Interpretability

Generate instance-level feature attributions with a compatible saved model:

```bash
python interpretation.py --data mimic --train_prop 0.7 --test_prop 0.3 \
  --num_samples 10 --random_seed 0
```

`interpretation.py` contains the default checkpoint path. Change it to the checkpoint created by your run when needed.

## Verify an archival copy

Run the repository checks before creating a release or uploading a snapshot:

```bash
python scripts/verify_archive.py
```

## Citation and license

Use the citation metadata in [`CITATION.cff`](CITATION.cff) and cite the permanent DOI once published. The software is licensed under the [Apache License 2.0](LICENSE). Dataset licenses and data-use agreements are separate from the software license.
