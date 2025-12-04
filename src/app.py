import streamlit as st
import pandas as pd
import numpy as np
from utils import extract_text, compute_similarity, detect_gender
from eval_fairness import evaluate_fairness

st.set_page_config(page_title="AI Candidate Ranking", layout="wide")

# ---------------------------
# Load Sentence Transformer
# ---------------------------
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("all-MiniLM-L6-v2")


# ----------------------------------------------------
# UI Header
# ----------------------------------------------------
st.title("AI Candidate Ranking System")
st.subheader("Streamlit Dashboard for HR Professionals")

st.write("---")


# ----------------------------------------------------
# SECTION 1: Job Description Upload
# ----------------------------------------------------
st.header("Job Description")

jd_text = st.text_area("Paste JD Here", height=180)

jd_file = st.file_uploader("Upload JD (PDF/DOCX/TXT)", type=["pdf", "docx", "txt"])

if jd_file:
    jd_text = extract_text(jd_file)


# ----------------------------------------------------
# SECTION 2: Candidate Resume Upload
# ----------------------------------------------------
st.header("Candidate Resumes")

resume_files = st.file_uploader(
    "Upload Resumes",
    type=["pdf", "docx", "txt"],
    accept_multiple_files=True
)

# Run system only if JD and resumes uploaded
if jd_text and resume_files:

    st.write("### Candidate Ranking Results")

    results = []
    jd_vec = model.encode([jd_text])[0]

    for file in resume_files:
        resume_text = extract_text(file)
        resume_vec = model.encode([resume_text])[0]

        score = compute_similarity(jd_vec, resume_vec)
        gender = detect_gender(resume_file.name, resume_text)


        results.append({
            "candidate": file.name,
            "score": round(float(score), 4),
            "gender": gender
        })

    # Create DataFrame
    results_df = pd.DataFrame(results)
    results_df["rank"] = results_df["score"].rank(ascending=False).astype(int)
    results_df = results_df.sort_values("rank")

    st.dataframe(results_df, use_container_width=True)

    st.write("---")

    # ----------------------------------------------------
    # SECTION 3: FAIRNESS AUDIT (Gender Bias)
    # ----------------------------------------------------
    st.header("Fairness Audit (Gender Bias)")

    with st.spinner("Calculating Fairness Metrics..."):
        dir_base, dir_mitigated, eod, err = evaluate_fairness(results_df)

    col1, col2, col3 = st.columns(3)

    if err:
        col1.warning("DIR Baseline: Not Available")
        col2.warning("DIR Mitigated: Not Available")
        col3.warning("EOD: Not Available")
        st.info(err)
    else:
        col1.metric("DIR (Baseline)", round(dir_base, 3))
        col2.metric("DIR (Mitigated)", round(dir_mitigated, 3))
        col3.metric("EOD", round(eod, 3))


else:
    st.info("Please upload both a Job Description and candidate resumes to begin.")
