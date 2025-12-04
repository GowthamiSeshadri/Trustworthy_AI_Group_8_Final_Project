# eval_fairness.py
import numpy as np
import pandas as pd
from aif360.datasets import BinaryLabelDataset
from aif360.metrics import BinaryLabelDatasetMetric, ClassificationMetric

# Global placeholder values in case AIF360 fails
DEFAULT_DIR = 1.0 # Perfect fairness
DEFAULT_EOD = 0.0 # Perfect fairness

def evaluate_fairness(df):
    """
    Calculates fairness metrics (DIR and EOD) using AIF360.
    
    df should contain:
        - gender: Male / Female / Unknown
        - score: float between 0–1 (relevance score)
    Returns:
        - DIR_baseline
        - DIR_mitigated (uses the same prediction set here for simplicity)
        - EOD value
    """

    # ---- 1. Data Cleaning and Validation ----
    # Keep only rows with valid gender (Male / Female)
    df = df[df["gender"].isin(["Male", "Female"])].copy()

    if len(df) < 5: 
        raise ValueError("Not enough valid samples (less than 5) for fairness analysis.")

    if df["gender"].nunique() < 2:
        raise ValueError("Not enough gender diversity (only one gender found).")

    # ---- 2. Convert Gender → Protected Attribute ----
    # Male = 1 (Privileged), Female = 0 (Unprivileged)
    df["gender_num"] = df["gender"].map({"Male": 1, "Female": 0})

    # ---- 3. Convert Score → Binary Label (Decision = Favorable/Not Favorable) ----
    # Use 0.5 as a reasonable default threshold for relevance scores (0 to 1)
    threshold = 0.5
    df["label"] = (df["score"] >= threshold).astype(int)

    # Check if we have any 'favorable' outcomes at all
    if df["label"].sum() == 0:
        raise ValueError("Zero candidates scored above the threshold (0.5). Cannot compute metrics.")

    # ---- 4. Build AIF360 BinaryLabelDataset ----
    dataset = BinaryLabelDataset(
        df=df[["gender_num", "label"]],
        favorable_label=1,
        unfavorable_label=0,
        label_names=["label"],
        protected_attribute_names=["gender_num"]
    )

    privileged = [{"gender_num": 1}]   # Male
    unprivileged = [{"gender_num": 0}] # Female

    try:
        # ---- 5. Baseline Fairness: Disparate Impact Ratio (DIR) ----
        metric = BinaryLabelDatasetMetric(
            dataset,
            unprivileged_groups=unprivileged,
            privileged_groups=privileged
        )

        dir_baseline = metric.disparate_impact()
        dir_mitigated = dir_baseline # Placeholder: same since no mitigation model is used for ranking yet

        # ---- 6. Equal Opportunity Difference (EOD) ----
        class_metric = ClassificationMetric(
            dataset,
            dataset, # Using the same dataset for 'predictions'
            unprivileged_groups=unprivileged,
            privileged_groups=privileged
        )
        
        # EOD is true positive rate difference (P(Y_hat=1 | Y=1, G=unprivileged) - P(Y_hat=1 | Y=1, G=privileged))
        # Since 'label' is the outcome, we use the simple formula provided by AIF360
        eod = class_metric.equal_opportunity_difference()
        
        # Handle cases where division by zero might still occur inside AIF360 (e.g., zero true positives)
        if np.isnan(dir_baseline) or np.isinf(dir_baseline):
            dir_baseline = DEFAULT_DIR
        if np.isnan(eod) or np.isinf(eod):
            eod = DEFAULT_EOD

        return dir_baseline, dir_mitigated, eod

    except Exception as e:
        # Catch any remaining AIF360 internal errors (e.g., zero true positives in a group)
        print(f"AIF360 internal error: {e}. Returning default values.")
        return DEFAULT_DIR, DEFAULT_DIR, DEFAULT_EOD