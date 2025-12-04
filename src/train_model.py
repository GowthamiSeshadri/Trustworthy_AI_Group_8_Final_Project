import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.utils import resample


# ---------------------------------------------
# 1. LOAD DATA
# ---------------------------------------------
df = pd.read_csv("../data/HR-Employee.csv")

# Convert Attrition → hired (1 = No attrition)
df["hired"] = df["Attrition"].map({"No": 1, "Yes": 0})

# Drop unused or non-predictive columns
df = df.drop(columns=["Attrition", "EmployeeCount", "EmployeeNumber", "Over18", "StandardHours"])

# Target variable
y = df["hired"]
X = df.drop(columns=["hired"])


# ---------------------------------------------
# 2. DEFINE CATEGORICAL + NUMERIC FEATURES
# ---------------------------------------------
categorical_cols = [
    "BusinessTravel", "Department", "EducationField",
    "Gender", "JobRole", "MaritalStatus", "OverTime"
]

numeric_cols = [col for col in X.columns if col not in categorical_cols]


# ---------------------------------------------
# 3. PREPROCESSOR PIPELINE
# ---------------------------------------------
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
    ]
)

# ---------------------------------------------
# 4. ORIGINAL MODEL (NO MITIGATION)
# ---------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

pipeline_orig = Pipeline([
    ("preprocessor", preprocessor),
    ("clf", LogisticRegression(max_iter=500))
])

pipeline_orig.fit(X_train, y_train)

# Save original model + preprocessor
joblib.dump(pipeline_orig, "../models/model_orig.pkl")
joblib.dump(preprocessor, "../models/preprocessor.pkl")

print("✔ Original Model Trained & Saved")


# ---------------------------------------------
# 5. MITIGATION: BALANCE THE DATASET
# ---------------------------------------------
df_majority = df[df.hired == 1]
df_minority = df[df.hired == 0]

df_minority_up = resample(df_minority, replace=True, n_samples=len(df_majority), random_state=42)
df_balanced = pd.concat([df_majority, df_minority_up])

y2 = df_balanced["hired"]
X2 = df_balanced.drop(columns=["hired"])

pipeline_mit = Pipeline([
    ("preprocessor", preprocessor),
    ("clf", LogisticRegression(max_iter=500))
])

pipeline_mit.fit(X2, y2)

joblib.dump(pipeline_mit, "../models/model_mit.pkl")

print("✔ Mitigated Model Trained & Saved")
