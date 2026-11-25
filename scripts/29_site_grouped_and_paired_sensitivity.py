from __future__ import annotations

import importlib
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold


revision = importlib.import_module("27_run_revision_multimodal_analysis")
OUTPUT_DIR = Path("outputs") / "revision1" / "sensitivity"
RANDOM_SEED = 42


def site_grouped_analysis() -> None:
    clinical = pd.read_parquet(revision.CLINICAL_PATH)
    image = pd.read_csv(revision.IMAGE_PATH)
    image_columns = [column for column in image.columns if column.startswith("img_feat_")]
    clinical_columns = [
        column
        for column in clinical.columns
        if column not in revision.EXCLUDED and clinical[column].notna().any()
    ]
    table = clinical.merge(image[["PatientID"] + image_columns], on="PatientID", how="inner")
    table["label_2y"] = table["label_2y"].astype(int)
    table["tissue_source_site"] = table["PatientID"].str.split("-").str[1]
    feature_sets = {
        "clinical_only": clinical_columns,
        "image_only": image_columns,
        "early_fusion": clinical_columns + image_columns,
    }
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    rows = []
    for fold, (train_idx, test_idx) in enumerate(
        cv.split(table, table["label_2y"], groups=table["tissue_source_site"]), start=1
    ):
        train, test = table.iloc[train_idx], table.iloc[test_idx]
        for modality, columns in feature_sets.items():
            for model_name in ["logistic_regression", "random_forest"]:
                pipeline = revision.make_pipeline(train[columns], model_name)
                pipeline.fit(train[columns], train["label_2y"])
                scores = pipeline.predict_proba(test[columns])[:, 1]
                for patient_id, site, truth, score in zip(
                    test["PatientID"], test["tissue_source_site"], test["label_2y"], scores
                ):
                    rows.append(
                        {
                            "PatientID": patient_id,
                            "tissue_source_site": site,
                            "fold": fold,
                            "modality": modality,
                            "model": model_name,
                            "label_2y": int(truth),
                            "predicted_probability": float(score),
                        }
                    )
    predictions = pd.DataFrame(rows)
    predictions = revision.add_late_fusion(
        predictions.assign(repeat=1)[
            ["PatientID", "label_2y", "repeat", "fold", "modality", "model", "predicted_probability"]
        ]
    )
    summaries = []
    for (modality, model), group in predictions.groupby(["modality", "model"]):
        summaries.append(
            {
                "modality": modality,
                "model": model,
                **revision.score_row(group["label_2y"].to_numpy(), group["predicted_probability"].to_numpy()),
            }
        )
    predictions.to_csv(OUTPUT_DIR / "site_grouped_predictions.tsv", sep="\t", index=False)
    pd.DataFrame(summaries).to_csv(OUTPUT_DIR / "site_grouped_metrics.tsv", sep="\t", index=False)


def paired_bootstrap() -> None:
    predictions = pd.read_csv(
        Path("outputs") / "revision1" / "multimodal_repeated_cv" / "patient_predictions.tsv",
        sep="\t",
    )
    predictions = predictions[predictions["repeat"].eq(1)]
    comparisons = [
        ("late_fusion", "random_forest", "clinical_only", "random_forest"),
        ("early_fusion", "random_forest", "clinical_only", "random_forest"),
        ("late_fusion", "logistic_regression", "clinical_only", "logistic_regression"),
    ]
    rng = np.random.default_rng(RANDOM_SEED)
    rows = []
    for modality_a, model_a, modality_b, model_b in comparisons:
        a = predictions[(predictions["modality"] == modality_a) & (predictions["model"] == model_a)]
        b = predictions[(predictions["modality"] == modality_b) & (predictions["model"] == model_b)]
        paired = a.merge(b, on=["PatientID", "label_2y"], suffixes=("_a", "_b"))
        y = paired["label_2y"].to_numpy()
        score_a = paired["predicted_probability_a"].to_numpy()
        score_b = paired["predicted_probability_b"].to_numpy()
        for metric_name, metric in [("roc_auc", roc_auc_score), ("pr_auc", average_precision_score)]:
            observed = metric(y, score_a) - metric(y, score_b)
            differences = []
            for _ in range(2000):
                indices = rng.integers(0, len(y), len(y))
                if np.unique(y[indices]).size < 2:
                    continue
                differences.append(metric(y[indices], score_a[indices]) - metric(y[indices], score_b[indices]))
            differences = np.asarray(differences)
            rows.append(
                {
                    "model_a": f"{modality_a}_{model_a}",
                    "model_b": f"{modality_b}_{model_b}",
                    "metric": metric_name,
                    "observed_difference_a_minus_b": observed,
                    "bootstrap_ci_low": np.quantile(differences, 0.025),
                    "bootstrap_ci_high": np.quantile(differences, 0.975),
                    "two_sided_p": min(1.0, 2 * min(np.mean(differences <= 0), np.mean(differences >= 0))),
                    "n_bootstrap": len(differences),
                }
            )
    pd.DataFrame(rows).to_csv(OUTPUT_DIR / "paired_bootstrap_comparisons.tsv", sep="\t", index=False)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    site_grouped_analysis()
    paired_bootstrap()
    print(f"Wrote site-grouped and paired sensitivity results to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
