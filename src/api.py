from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import joblib
import pandas as pd
import numpy as np

app = FastAPI(
    title="AI Hiring Fairness Evaluation API",
    description="Predict hiring decisions with/without fairness mitigation.",
    version="1.0.0"
)

# ---------------------------
# Load models & preprocessor
# ---------------------------
try:
    model_orig = joblib.load("../models/model_orig.pkl")
    model_mit = joblib.load("../models/model_mit.pkl")
    preprocessor = joblib.load("../models/preprocessor.pkl")
    print("Models loaded successfully.")
except:
    raise ImportError("Models not found. Run train_model.py first.")


# ---------------------------
# Input Schema (MATCHES HR CSV)
# ---------------------------
class CandidateInput(BaseModel):
    Age: float
    BusinessTravel: str
    DailyRate: float
    Department: str
    DistanceFromHome: float
    Education: float
    EducationField: str
    EnvironmentSatisfaction: float
    Gender: str
    HourlyRate: float
    JobInvolvement: float
    JobLevel: float
    JobRole: str
    JobSatisfaction: float
    MaritalStatus: str
    MonthlyIncome: float
    NumCompaniesWorked: float
    OverTime: str
    PercentSalaryHike: float
    PerformanceRating: float
    RelationshipSatisfaction: float
    StockOptionLevel: float
    TotalWorkingYears: float
    TrainingTimesLastYear: float
    WorkLifeBalance: float
    YearsAtCompany: float
    YearsInCurrentRole: float
    YearsSinceLastPromotion: float
    YearsWithCurrManager: float

    mitigate: Optional[bool] = False
