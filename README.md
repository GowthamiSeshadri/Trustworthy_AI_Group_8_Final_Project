Project :  Fair AI Resume Evaluator (Trustworthy Hiring System)

1. Overview

   This project evaluates the trustworthiness (Fairness & Bias) of our midterm AI application that ranks job candidates based on semantic similarity between the       Job Description (JD) and resumes.
   The final project focuses on fairness analysis, bias detection, bias mitigation, and transparent reporting using AIF360.

2. Features:
   Resume and JD text extraction (PDF/DOCX/TXT)
   Semantic similarity scoring using Sentence-BERT (all-MiniLM-L6-v2)
   Candidate ranking and score visualization
   Gender detection (keyword + heuristic)

3 :Fairness metrics:
   Disparate Impact Ratio (DIR)
   Equal Opportunity Difference (EOD)
   Automatic handling when fairness cannot be computed due to lack of diversity
   Streamlit-based interactive UI

4. How to Install & Run
   Step 1 — Create Virtual Environment
        python -m venv venv
        venv\Scripts\activate
   Step 2 — Install Dependencies
         pip install -r src/requirements.txt
   Step 3 — Run the Application
         streamlit run src/app.py
       
   

