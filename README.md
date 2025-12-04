1. Project Overview

This final project evaluates the trustworthiness of the AI Resume Ranking System developed in the midterm project.
The system matches candidate resumes to a job description (JD) using Sentence-BERT embeddings and cosine similarity to generate a ranked list of suitable candidates.

For the final project, we focus on evaluating and improving the system’s Fairness & Bias using quantitative fairness metrics from AIF360.

How to Run the AI Resume Ranking System (Final Project)

STEP 1: Create & Activate Virtual Environment

Windows
cd "HR PROJECT"
python -m venv venv
venv\Scripts\activate

macOS / Linux
cd "HR PROJECT"
python3 -m venv venv
source venv/bin/activate

STEP 2: Install All Dependencies

Run inside your activated virtual environment:

pip install -r requirements.txt
This installs:
Streamlit (UI)
Sentence-BERT
scikit-learn
PyMuPDF (fitz)
AIF360 (fairness metrics)
docx / python-docx

STEP 3: Make Sure Models Exist

Before running the app, you must have:

models/
│── model_orig.pkl
│── model_mit.pkl
│── preprocessor.pkl


If they are missing, run:

python src/train_model.py

STEP 4: Run the Streamlit UI

Use this exact command (because the app is inside /src):
streamlit run src/app.py

STEP 5: Using the Application
Upload Job Description
Paste JD text OR upload a PDF/DOCX/TXT file
Upload Candidate Resumes
Upload multiple resumes (PDF, DOCX, TXT)
Assign Gender
Select gender for each candidate (required for fairness evaluation)
Run Screening
System computes embedding similarity
Generates ranking
Computes fairness metrics:
Metric	Meaning
DIR (Baseline)	Bias before mitigation
DIR (Mitigated)	After reweighing
EOD	Equal Opportunity Difference
These appear at the top-right of the UI.

STEP 6: (Optional) Run FastAPI Backend

If you want to test the API separately:
uvicorn src.api:app --reload

STEP 7: Fairness Evaluation Script (Optional)

If you want to test fairness metrics manually:

python src/eval_fairness.py

STEP 8: Deactivate Environment

When finished:

deactivate
