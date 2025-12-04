import pandas as pd
import numpy as np
from aif360.datasets import BinaryLabelDataset
from aif360.metrics import BinaryLabelDatasetMetric, ClassificationMetric
from sklearn.preprocessing import MinMaxScaler


# ----------------------------------------------------------
# Convert Results into Numerical Dataset for AIF360
# ----------------------------------------------------------
def prepare_dataset(df):
    """
    df contains:
        - candidate
        - score (float)
        - gender ("Male"/"Female"/"Unknown")
        - rank (int)
    """
    # Remove unknown gender (AIF360 requires exactly two groups)
    df = df[df["gender"].isin(["Male", "Female"])]

    if df.empty or len(df["gender"].unique()) < 2:
        return None, "Not enough gender diversity for fairness analysis"

    # Create binary label: high score = selected (1), else 0
    scaler = MinMaxScaler()
    df["score_scaled"] = scaler.fit_transform(df[["score"]])

    df["label"] = (df["score_scaled"] >= 0.50).astype(int)

    # Binary protected attribute
    df["gender_binary"] = df["gender"].map({"Male": 1, "Female": 0})

    # Create AIF360 dataset
    dataset = BinaryLabelDataset(
        favorable_label=1,
        unfavorable_label=0,
        df=df[["gender_binary", "label"]],
        label_names=["label"],
        protected_attribute_names=["gender_binary"]
    )

    return dataset, None


# ----------------------------------------------------------
# Compute DIR Baseline, DIR Mitigated, and EOD
# ----------------------------------------------------------
def evaluate_fairness(df):
    dataset, err = prepare_dataset(df)
    if err:
        return np.nan, np.nan, np.nan, err

    # Baseline metrics
    metric = BinaryLabelDatasetMetric(dataset,
                                      privileged_groups=[{"gender_binary": 1}],
                                      unprivileged_groups=[{"gender_binary": 0}])

    dir_baseline = metric.disparate_impact()

    # For simplicity, mitigation simulated using label flipping threshold
    # (You can plug real mitigated model results here if needed)
    df2 = df.copy()
    df2["label"] = (df2["score_scaled"] >= 0.45).astype(int)

    dataset2 = BinaryLabelDataset(
        favorable_label=1,
        unfavorable_label=0,
        df=df2[["gender_binary", "label"]],
        label_names=["label"],
        protected_attribute_names=["gender_binary"]
    )

    metric2 = BinaryLabelDatasetMetric(dataset2,
                                       privileged_groups=[{"gender_binary": 1}],
                                       unprivileged_groups=[{"gender_binary": 0}])

    dir_mitigated = metric2.disparate_impact()

    # Classification-based metric (EOD)
    cls_metric = ClassificationMetric(dataset, dataset2,
                                     privileged_groups=[{"gender_binary": 1}],
                                     unprivileged_groups=[{"gender_binary": 0}])

    eod = cls_metric.equal_opportunity_difference()

    return dir_baseline, dir_mitigated, eod, None
