from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


INPUT_PATH = Path("data") / "processed" / "clinical_features.tsv"
OUTPUT_DIR = Path("outputs") / "revision1" / "full_cohort_clinical"
RANDOM_SEED = 42

CLINICAL_FEATURES = [
    "gender",
    "race",
    "ethnicity",
    "age_at_diagnosis_years",
    "primary_diagnosis",
    "morphology",
    "tumor_grade",
    "classification_of_tumor",
    "tissue_or_organ_of_origin",
    "site_of_resection_or_biopsy",
    "prior_malignancy",
    "synchronous_malignancy",
    "ajcc_pathologic_stage",
    "ajcc_pathologic_t",
    "ajcc_pathologic_n",
    "ajcc_pathologic_m",
    "ajcc_clinical_stage",
    "ajcc_clinical_t",
    "ajcc_clinical_n",
    "ajcc_clinical_m",
    "tobacco_smoking_status",
    "alcohol_history",
    "bmi",
    "height",
    "weight",
]

FEATURE_SETS = {
    "all_clinical": CLINICAL_FEATURES,
    "stage_only": ["ajcc_pathologic_stage"],
    "tnm_only": ["ajcc_pathologic_t", "ajcc_pathologic_n", "ajcc_pathologic_m"],
    "basic_stage_tnm": [
        "age_at_diagnosis_years",
        "gender",
        "ajcc_pathologic_stage",
        "ajcc_pathologic_t",
        "ajcc_pathologic_n",
        "ajcc_pathologic_m",
    ],
}


def make_pipeline(x_train: pd.DataFrame, model_name: str) -> Pipeline:
    numeric = x_train.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical = [column for column in x_train.columns if column not in numeric]
    preprocess = ColumnTransformer(
        [
            (
                "numeric",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric,
            ),
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                categorical,
            ),
        ]
    )
    if model_name == "logistic_regression":
        model = LogisticRegression(
            C=0.25,
            max_iter=5000,
            class_weight="balanced",
            random_state=RANDOM_SEED,
        )
    else:
        model = RandomForestClassifier(
            n_estimators=500,
            min_samples_leaf=2,
            max_features="sqrt",
            class_weight="balanced",
            random_state=RANDOM_SEED,
            n_jobs=-1,
        )
    return Pipeline([("preprocess", preprocess), ("model", model)])


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(INPUT_PATH, sep="\t")
    data = data[data["label_2y"].notna()].copy()
    data["label_2y"] = data["label_2y"].astype(int)
    data = data.replace({"": np.nan, "'--": np.nan, "--": np.nan, "Not Reported": np.nan})

    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=RANDOM_SEED)
    rows: list[dict[str, object]] = []
    y = data["label_2y"].to_numpy()

    for feature_set, requested_columns in FEATURE_SETS.items():
        columns = [
            column
            for column in requested_columns
            if column in data.columns and data[column].notna().any()
        ]
        x = data[columns].copy()
        for model_name in ["logistic_regression", "random_forest"]:
            for fold, (train_idx, test_idx) in enumerate(cv.split(x, y), start=1):
                pipeline = make_pipeline(x.iloc[train_idx], model_name)
                pipeline.fit(x.iloc[train_idx], y[train_idx])
                scores = pipeline.predict_proba(x.iloc[test_idx])[:, 1]
                predictions = (scores >= 0.5).astype(int)
                rows.append(
                    {
                        "feature_set": feature_set,
                        "model": model_name,
                        "fold": fold,
                        "n_train": len(train_idx),
                        "n_test": len(test_idx),
                        "accuracy": accuracy_score(y[test_idx], predictions),
                        "precision": precision_score(y[test_idx], predictions, zero_division=0),
                        "recall": recall_score(y[test_idx], predictions, zero_division=0),
                        "f1": f1_score(y[test_idx], predictions, zero_division=0),
                        "roc_auc": roc_auc_score(y[test_idx], scores),
                        "pr_auc": average_precision_score(y[test_idx], scores),
                    }
                )

    fold_metrics = pd.DataFrame(rows)
    metric_columns = ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]
    summary = (
        fold_metrics.groupby(["feature_set", "model"])[metric_columns]
        .agg(["mean", "std"])
        .reset_index()
    )
    summary.columns = ["_".join(column).rstrip("_") for column in summary.columns.to_flat_index()]
    fold_metrics.to_csv(OUTPUT_DIR / "fold_metrics.tsv", sep="\t", index=False)
    summary.to_csv(OUTPUT_DIR / "metric_summary.tsv", sep="\t", index=False)
    pd.DataFrame(
        {
            "n_patients": [len(data)],
            "n_negative": [(data["label_2y"] == 0).sum()],
            "n_positive": [(data["label_2y"] == 1).sum()],
            "cv": ["RepeatedStratifiedKFold(5 folds x 5 repeats, seed=42)"],
        }
    ).to_csv(OUTPUT_DIR / "cohort_summary.tsv", sep="\t", index=False)
    print(summary.to_string(index=False))
    print(f"\nWrote revision outputs to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
