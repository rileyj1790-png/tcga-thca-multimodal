# TCGA-THCA multimodal analysis

This repository contains the analysis code for a research feasibility study of documented thyroid carcinoma recurrence or progression within 730 days. It is not clinical software.

## Setup and inputs

Use Python 3.12 and install `requirements.txt`. Run commands from the repository root. Download TCGA-THCA source data separately from the [NCI Genomic Data Commons](https://portal.gdc.cancer.gov/). No clinical records, slide images, tiles, embeddings, or patient-level predictions are included.

The revision scripts expect these prepared inputs from the study workflow:

- `data/processed/clinical_features.tsv`: one row per eligible patient, including `PatientID` and `label_2y`.
- `data/processed/balanced_clinical_features.parquet`: one row per imaging-cohort patient with the same join key and label.
- `processed/balanced_imaging_2y/tiles/`: patient folders containing JPEG pathology tiles. Script 10 documents slide tiling; it reads the local slide table at `SLIDE_TABLE` and uses the slide directory defined in the script. Run it with `--help` for tiling options.
- Script 28 additionally expects the local labels, patient master table, balanced cohort CSV, and slide audit CSV listed at the top of that script.

These scripts document the revision from prepared inputs. They do not reconstruct the exact cohort from a fresh GDC download by themselves.

## Run the revision

```text
python scripts/19_extract_resnet18_patient_embeddings.py --output-csv outputs/revision1/features/image_features_resnet18_random.csv --output-parquet outputs/revision1/features/patient_resnet18_random.parquet --metadata-path outputs/revision1/reports/resnet18_random_embedding_metadata.json --batch-size 64 --num-threads 8
python scripts/26_train_full_cohort_clinical.py
python scripts/27_run_revision_multimodal_analysis.py
python scripts/28_revision_cohort_and_site_audit.py
python scripts/29_site_grouped_and_paired_sensitivity.py
```

Model settings, feature exclusions, fold design, and sampling seeds are defined in the scripts. Outputs are written under `outputs/revision1`. Some outputs are patient-level and should remain local.
