# Data availability and provenance

This repository does not use GitHub or a personal cloud-storage folder as the archival authority for research data. The permanent software snapshot is deposited independently in Zenodo at [https://doi.org/10.5281/zenodo.21487356](https://doi.org/10.5281/zenodo.21487356), while source datasets remain governed by their original repositories, licenses, and data-use agreements.

## Clinical and genomic benchmarks

| Dataset | Permanent or authoritative source | Access and redistribution status | Expected local file |
| --- | --- | --- | --- |
| MIMIC-III | [PhysioNet MIMIC-III Clinical Database v1.4](https://physionet.org/content/mimiciii/1.4/), DOI: [10.13026/C2XW26](https://doi.org/10.13026/C2XW26) | Credentialed access. The source data and patient-level derivatives are not included in this public archive. Users must complete PhysioNet requirements and create the analysis-ready cohort under the applicable data-use agreement. | `mimic.csv` |
| eICU-CRD (supported by the code but not one of the five principal manuscript benchmarks) | [PhysioNet eICU Collaborative Research Database v2.0](https://physionet.org/content/eicu-crd/2.0/), DOI: [10.13026/C2WM1R](https://doi.org/10.13026/C2WM1R) | Credentialed access. Not redistributed. | `eicu_cohort_modify.csv` |
| Framingham | [Kaggle source record](https://www.kaggle.com/datasets/aasheesh200/framingham-heart-study-dataset) | The Kaggle record reports an unknown license. The input is therefore not redistributed here; users must obtain it from the source and comply with its terms. | `framingham.csv` |
| METABRIC breast cancer | [cBioPortal METABRIC study](https://www.cbioportal.org/study/summary?id=brca_metabric) and [Kaggle transformation used by the analysis](https://www.kaggle.com/datasets/raghadalharbi/breast-cancer-gene-expression-profiles-metabric) | The Kaggle record identifies the database under the Open Database License and contents under the Database Contents License. Obtain the study from the cited source and retain source attribution. | `gene_breast_normalized.csv` |
| Heart disease | [UCI Heart Disease dataset](https://doi.org/10.24432/C52P4X) and [Kaggle integration used by the analysis](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction) | Third-party data; not relicensed under the CodeK software license. Obtain from the cited source and follow its license and attribution requirements. | `heart_normalized.csv` |
| Stroke prediction | [Kaggle source record](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset) | The source describes the data files as copyright of the original authors. The input is not redistributed without separate permission. | `stroke_normalized.csv` |
| Personal Genome Project WGS | [Harvard Personal Genome Project](https://my.pgp-hms.org/) | Public participant-contributed data remain subject to the PGP access terms, participant consent, and responsible-use requirements. The local model-ready matrices are not bundled with the software archive. | `wgs_dummy_dataset.csv`, `group2_dummy_dataset.csv`, `group3_dummy_dataset.csv` |

## Reproducibility policy

- Place locally obtained inputs under `data/` with the expected filenames. Raw data are ignored by Git to prevent accidental publication.
- Keep restricted clinical data and patient-level derivatives only in approved storage.
- Record the source version, retrieval date, preprocessing script or notebook, and SHA-256 checksum in `data/MANIFEST.tsv` for each local input.
- The repository's Apache-2.0 license applies to CodeK software only. It does not override a dataset license, participant consent, or a data-use agreement.
- Generated aggregate tables and manuscript figures may be included in the software archive when they contain no row-level or identifying information.

## Known archival limitation requiring author confirmation

The Framingham and Stroke Kaggle records do not provide sufficiently clear redistribution permission for the analysis-ready files. Those files should not be uploaded to a public DOI deposit unless the authors document permission or replace them with an equivalently processed input from a source that permits redistribution. This is a data-provenance issue, not a GitHub issue.
