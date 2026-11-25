from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


CLINICAL_PATH = Path("data") / "processed" / "balanced_clinical_features.parquet"
IMAGE_PATH = Path("outputs") / "revision1" / "features" / "image_features_resnet18_random.csv"
OUTPUT_DIR = Path("outputs") / "revision1" / "multimodal_repeated_cv"
TARGET_PREVALENCE = 29 / 304
RANDOM_SEED = 42

EXCLUDED = {
    "PatientID",
    "label_2y",
    "prior_treatment",
    "year_of_diagnosis",
    "diagnostic_slide_file_count",
    "biospecimen_slide_metadata_row_count",
}


def make_pipeline(x: pd.DataFrame, model_name: str) -> Pipeline:
    numeric = x.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical = [column for column in x.columns if column not in numeric]
    preprocess = ColumnTransformer(
        [
            (
                "numeric",
                Pipeline(
                    [("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
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
            C=0.25, max_iter=5000, class_weight="balanced", random_state=RANDOM_SEED
        )
    else:
        model = RandomForestClassifier(
            n_estimators=200,
            min_samples_leaf=2,
            max_features="sqrt",
            class_weight="balanced",
            random_state=RANDOM_SEED,
            n_jobs=1,
        )
    return Pipeline([("preprocess", preprocess), ("model", model)])


def score_row(y: np.ndarray, score: np.ndarray) -> dict[str, float]:
    pred = (score >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    sensitivity = recall_score(y, pred, zero_division=0)
    specificity = tn / (tn + fp) if tn + fp else np.nan
    pi = TARGET_PREVALENCE
    ppv = sensitivity * pi / (sensitivity * pi + (1 - specificity) * (1 - pi))
    npv = specificity * (1 - pi) / ((1 - sensitivity) * pi + specificity * (1 - pi))
    return {
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": sensitivity,
        "specificity": specificity,
        "f1": f1_score(y, pred, zero_division=0),
        "roc_auc": roc_auc_score(y, score),
        "pr_auc": average_precision_score(y, score),
        "brier_score": brier_score_loss(y, score),
        "ppv_at_observed_prevalence": ppv,
        "npv_at_observed_prevalence": npv,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def add_late_fusion(predictions: pd.DataFrame) -> pd.DataFrame:
    additions = []
    for model_name in ["logistic_regression", "random_forest"]:
        clinical = predictions[
            (predictions["modality"] == "clinical_only") & (predictions["model"] == model_name)
        ]
        image = predictions[
            (predictions["modality"] == "image_only") & (predictions["model"] == model_name)
        ]
        merged = clinical.merge(
            image,
            on=["PatientID", "label_2y", "repeat", "fold"],
            suffixes=("_clinical", "_image"),
        )
        for row in merged.itertuples():
            additions.append(
                {
                    "PatientID": row.PatientID,
                    "label_2y": row.label_2y,
                    "repeat": row.repeat,
                    "fold": row.fold,
                    "modality": "late_fusion",
                    "model": model_name,
                    "predicted_probability": (
                        row.predicted_probability_clinical + row.predicted_probability_image
                    )
                    / 2,
                }
            )
    return pd.concat([predictions, pd.DataFrame(additions)], ignore_index=True)


def summarize_repeats(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for (modality, model, repeat), group in predictions.groupby(["modality", "model", "repeat"]):
        rows.append(
            {
                "modality": modality,
                "model": model,
                "repeat": repeat,
                **score_row(group["label_2y"].to_numpy(), group["predicted_probability"].to_numpy()),
            }
        )
    repeated = pd.DataFrame(rows)
    metrics = [
        "accuracy", "precision", "recall", "specificity", "f1", "roc_auc", "pr_auc",
        "brier_score", "ppv_at_observed_prevalence", "npv_at_observed_prevalence",
    ]
    summary_rows = []
    for (modality, model), group in repeated.groupby(["modality", "model"]):
        row = {"modality": modality, "model": model}
        for metric in metrics:
            mean = group[metric].mean()
            std = group[metric].std(ddof=1)
            margin = 2.776 * std / np.sqrt(len(group))
            row.update(
                {
                    f"{metric}_mean": mean,
                    f"{metric}_std": std,
                    f"{metric}_ci_low": max(0.0, mean - margin),
                    f"{metric}_ci_high": min(1.0, mean + margin),
                }
            )
        summary_rows.append(row)
    return repeated, pd.DataFrame(summary_rows)


def create_calibration_figure(predictions: pd.DataFrame) -> None:
    repeat_one = predictions[predictions["repeat"] == 1]
    fig, ax = plt.subplots(figsize=(7, 5))
    styles = {
        "clinical_only": ("#2364aa", "Clinical only"),
        "image_only": ("#d1495b", "Image only"),
        "early_fusion": ("#2a9d8f", "Early fusion"),
        "late_fusion": ("#7b2cbf", "Late fusion"),
    }
    for modality, (color, label) in styles.items():
        group = repeat_one[
            (repeat_one["modality"] == modality)
            & (repeat_one["model"] == "logistic_regression")
        ]
        observed, predicted = calibration_curve(
            group["label_2y"], group["predicted_probability"], n_bins=5, strategy="quantile"
        )
        ax.plot(predicted, observed, marker="o", color=color, label=label)
    ax.plot([0, 1], [0, 1], linestyle="--", color="#555555", label="Ideal")
    ax.set(xlabel="Mean predicted probability", ylabel="Observed event proportion", xlim=(0, 1), ylim=(0, 1))
    ax.set_title("Calibration in the balanced 58-patient cohort")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "calibration_balanced_cohort.png", dpi=300)
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    clinical = pd.read_parquet(CLINICAL_PATH)
    image = pd.read_csv(IMAGE_PATH)
    image_columns = [column for column in image.columns if column.startswith("img_feat_")]
    clinical_columns = [
        column for column in clinical.columns if column not in EXCLUDED and clinical[column].notna().any()
    ]
    table = clinical.merge(image[["PatientID"] + image_columns], on="PatientID", how="inner")
    table["label_2y"] = table["label_2y"].astype(int)
    feature_sets = {
        "clinical_only": clinical_columns,
        "image_only": image_columns,
        "early_fusion": clinical_columns + image_columns,
    }

    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=RANDOM_SEED)
    rows = []
    for split_index, (train_idx, test_idx) in enumerate(cv.split(table, table["label_2y"]), start=0):
        repeat = split_index // 5 + 1
        fold = split_index % 5 + 1
        print(f"Repeat {repeat}/5, fold {fold}/5", flush=True)
        train = table.iloc[train_idx]
        test = table.iloc[test_idx]
        for modality, columns in feature_sets.items():
            for model_name in ["logistic_regression", "random_forest"]:
                pipeline = make_pipeline(train[columns], model_name)
                pipeline.fit(train[columns], train["label_2y"])
                scores = pipeline.predict_proba(test[columns])[:, 1]
                for patient_id, truth, score in zip(test["PatientID"], test["label_2y"], scores):
                    rows.append(
                        {
                            "PatientID": patient_id,
                            "label_2y": int(truth),
                            "repeat": repeat,
                            "fold": fold,
                            "modality": modality,
                            "model": model_name,
                            "predicted_probability": float(score),
                        }
                    )

    predictions = add_late_fusion(pd.DataFrame(rows))
    repeated_metrics, summary = summarize_repeats(predictions)
    predictions.to_csv(OUTPUT_DIR / "patient_predictions.tsv", sep="\t", index=False)
    repeated_metrics.to_csv(OUTPUT_DIR / "repeat_metrics.tsv", sep="\t", index=False)
    summary.to_csv(OUTPUT_DIR / "metric_summary_with_95ci.tsv", sep="\t", index=False)
    pd.DataFrame(
        {
            "clinical_feature": clinical_columns,
            "included": True,
        }
    ).to_csv(OUTPUT_DIR / "clinical_features_used.tsv", sep="\t", index=False)
    create_calibration_figure(predictions)
    print(summary[["modality", "model", "f1_mean", "roc_auc_mean", "pr_auc_mean"]].to_string(index=False))
    print(f"\nWrote revision analysis to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
