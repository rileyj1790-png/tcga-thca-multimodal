from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, mannwhitneyu


LABELS = Path("data") / "processed" / "labels_2y.tsv"
MASTER = (
    Path("biospecimen.project-tcga-thca.2026-05-27")
    / "tcga_thca_output"
    / "tcga_thca_patient_master_with_recurrence_labels.tsv"
)
BALANCED = Path("processed") / "balanced_imaging_2y" / "balanced_patients.csv"
SLIDE_AUDIT = Path("processed") / "balanced_imaging_2y" / "svs_slide_audit.csv"
OUTPUT_DIR = Path("outputs") / "revision1" / "audits"


def categorical_test(table: pd.DataFrame, column: str) -> dict[str, object]:
    values = table[["cohort", column]].fillna("Missing")
    contingency = pd.crosstab(values["cohort"], values[column])
    if contingency.shape[0] < 2 or contingency.shape[1] < 2:
        p_value = np.nan
    else:
        _, p_value, _, _ = chi2_contingency(contingency)
    return {
        "variable": column,
        "test": "chi_square",
        "included_summary": table.loc[table["cohort"].eq("included"), column]
        .fillna("Missing")
        .value_counts()
        .to_dict(),
        "excluded_summary": table.loc[table["cohort"].eq("excluded"), column]
        .fillna("Missing")
        .value_counts()
        .to_dict(),
        "p_value": p_value,
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    labels = pd.read_csv(LABELS, sep="\t")
    master = pd.read_csv(MASTER, sep="\t", low_memory=False).rename(
        columns={"patient_barcode": "PatientID"}
    )
    columns = [
        "PatientID",
        "gender",
        "race",
        "age_at_diagnosis_years",
        "ajcc_pathologic_stage",
        "ajcc_pathologic_t",
        "ajcc_pathologic_n",
        "ajcc_pathologic_m",
    ]
    table = labels[["PatientID", "label_inclusion_status", "label_exclusion_reason"]].merge(
        master[[column for column in columns if column in master.columns]], on="PatientID", how="left"
    )
    table["cohort"] = np.where(table["label_inclusion_status"].eq("included"), "included", "excluded")

    age = pd.to_numeric(table["age_at_diagnosis_years"], errors="coerce")
    included_age = age[table["cohort"].eq("included")].dropna()
    excluded_age = age[table["cohort"].eq("excluded")].dropna()
    rows = [
        {
            "variable": "age_at_diagnosis_years",
            "test": "mann_whitney_u",
            "included_summary": f"{included_age.mean():.2f} +/- {included_age.std():.2f} (n={len(included_age)})",
            "excluded_summary": f"{excluded_age.mean():.2f} +/- {excluded_age.std():.2f} (n={len(excluded_age)})",
            "p_value": mannwhitneyu(included_age, excluded_age, alternative="two-sided").pvalue,
        }
    ]
    for column in ["gender", "race", "ajcc_pathologic_stage", "ajcc_pathologic_t", "ajcc_pathologic_n", "ajcc_pathologic_m"]:
        if column in table.columns:
            rows.append(categorical_test(table, column))
    pd.DataFrame(rows).to_csv(OUTPUT_DIR / "included_vs_excluded.tsv", sep="\t", index=False)
    table["label_exclusion_reason"].fillna("included").value_counts().rename_axis("group").reset_index(
        name="n_patients"
    ).to_csv(OUTPUT_DIR / "cohort_exclusion_counts.tsv", sep="\t", index=False)

    balanced = pd.read_csv(BALANCED)
    label_column = "recurrence_progression_within_2y"
    balanced["tissue_source_site"] = balanced["PatientID"].str.split("-").str[1]
    site_counts = pd.crosstab(balanced["tissue_source_site"], balanced[label_column])
    site_counts.to_csv(OUTPUT_DIR / "balanced_tissue_source_site_by_label.tsv", sep="\t")

    slides = pd.read_csv(SLIDE_AUDIT)
    magnification = slides.groupby(["objective_power", "mpp_x", "mpp_y"], dropna=False).size().reset_index(name="n_slides")
    magnification.to_csv(OUTPUT_DIR / "slide_magnification_summary.tsv", sep="\t", index=False)
    print(f"Wrote cohort, site, and magnification audits to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
