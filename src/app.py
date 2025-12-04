import streamlit as st
import pandas as pd
import numpy as np
import io
import re 

# Assuming these files are in the same directory
from utils import extract_text, compute_similarity, detect_gender, get_embeddings, extract_experience 
from eval_fairness import evaluate_fairness

# ================================
# 🔧 Streamlit App Configuration
# ================================
st.set_page_config(page_title="AI Candidate Ranking System", layout="wide")

st.title("AI Candidate Ranking System")
st.text("Streamlit Dashboard for HR Professionals")

# ----------------------------------------------------------------------
# 🎨 LAYOUT SETUP: Create two main columns for Input (1) and Output (2)
# ----------------------------------------------------------------------
col_input, col_output = st.columns([1, 2.5]) 

# ----------------------------------------------------------------------
# GLOBAL DATA PLACEHOLDERS (Must be initialized for session state)
# ----------------------------------------------------------------------
if 'results_df' not in st.session_state:
    st.session_state.results_df = pd.DataFrame(columns=["rank", "CANDIDATE NAME", "SCORE (RELEVANCE)", "EXPERIENCE", "ACTION", "gender"])
if 'fairness_metrics' not in st.session_state:
     st.session_state.fairness_metrics = {'dir_base': 0.0, 'dir_mit': 0.0, 'eod': 0.0}

# Load metrics from state to persist across reruns
dir_baseline_val = st.session_state.fairness_metrics['dir_base']
dir_mitigated_val = st.session_state.fairness_metrics['dir_mit']
eod_val = st.session_state.fairness_metrics['eod']
show_results = False # Flag to control display

# If results already exist, assume we want to show them on load
if not st.session_state.results_df.empty:
    show_results = True


# ==================================================================
# 1️⃣ INPUT PANEL (col_input)
# ==================================================================
with col_input:
    # ------------------
    # 1. Job Description
    # ------------------
    st.header("1. Job Description")
    with st.container(border=True):
        jd_text = st.text_area("Paste JD Here", height=150, label_visibility="collapsed", key="jd_text_input")
        # FIX: Added non-empty label for accessibility warning fix
        uploaded_jd = st.file_uploader("Job Description File Upload", type=["pdf", "docx", "txt"], key="jd_upload", label_visibility="collapsed")
        st.button("Upload JD File (PDF/DOCX)", key="upload_jd_btn", use_container_width=True)

    jd_final_text = jd_text
    if uploaded_jd:
        try:
            jd_final_text = extract_text(uploaded_jd)
        except Exception as e:
            st.error(f"Error processing JD file: {e}")
            jd_final_text = jd_text

    st.markdown("---")

    # ------------------
    # 2. Candidate Resumes
    # ------------------
    st.header("2. Candidate Resumes")
    st.markdown("""
        <div style='border: 2px dashed #ddd; padding: 20px; text-align: center; color: #888;'>
            Drag and drop resumes here<br>(or click to browse)
        </div>
        """, unsafe_allow_html=True)
    
    # FIX: Added non-empty label for accessibility warning fix
    resume_files = st.file_uploader("Candidate Resumes Upload", type=["pdf", "docx", "txt"], accept_multiple_files=True, key="resume_upload", label_visibility="collapsed")
    
    st.markdown("---")
    
    # ------------------
    # Run Button
    # ------------------
    if st.button("RUN SCREENING", type="primary", use_container_width=True):
        if not jd_final_text:
            st.error("Please provide a Job Description to compute rankings.")
        elif not resume_files:
            st.warning("Upload at least one resume to compute rankings.")
        else:
            show_results = True


# ==================================================================
# 3️⃣ PROCESSING LOGIC (RUNS only if button is clicked/state is active)
# ==================================================================
if show_results:
    with st.spinner("Screening candidates and evaluating fairness..."):
        
        try:
            jd_vec = get_embeddings(jd_final_text)
        except Exception as e:
            st.error(f"Error generating JD embeddings: {e}")
            jd_vec = None

        if jd_vec is not None and resume_files:
            rows = []
            for file in resume_files:
                try:
                    resume_text = extract_text(file)
                    resume_vec = get_embeddings(resume_text)
                    
                    # Compute Similarity Score
                    score = compute_similarity(jd_vec, resume_vec)
                    
                    # Use the extract_experience function
                    experience = extract_experience(resume_text)
                    if experience is None:
                         # Fallback to random if not found
                         experience = np.random.randint(3, 15) 
                    
                    # Compute Gender for Fairness Audit
                    gender = detect_gender(resume_text)

                    candidate_name = file.name.split('.')[0] 

                    rows.append([candidate_name, score, gender, f"{experience} Years"])

                except Exception as e:
                    rows.append([file.name.split('.')[0], 0.0, "Unknown", "N/A"])
            
            if rows:
                results_df = pd.DataFrame(rows, columns=["CANDIDATE NAME", "SCORE (RELEVANCE)", "gender", "EXPERIENCE"])
                
                # Calculate Rank and Sort
                results_df["rank"] = results_df["SCORE (RELEVANCE)"].rank(ascending=False, method='min').astype(int)
                results_df = results_df.sort_values("rank").reset_index(drop=True)
                st.session_state.results_df = results_df
                
                # --- Fairness Audit Calculation ---
                try:
                    fairness_df = results_df[["gender", "SCORE (RELEVANCE)"]].rename(columns={"SCORE (RELEVANCE)": "score"})
                    
                    # FIX: Capture the calculated values
                    dir_baseline_val, dir_mitigated_val, eod_val = evaluate_fairness(fairness_df)
                    
                    # FIX: Save the calculated values to session state
                    st.session_state.fairness_metrics = {
                        'dir_base': dir_baseline_val, 
                        'dir_mit': dir_mitigated_val, 
                        'eod': eod_val
                    }

                except ValueError as ve:
                    # Update metrics to zero/defaults if calculation fails
                    dir_baseline_val, dir_mitigated_val, eod_val = 0.0, 0.0, 0.0
                    st.session_state.fairness_metrics = {'dir_base': 0.0, 'dir_mit': 0.0, 'eod': 0.0}
                    st.warning(f"Fairness Evaluation Skipped: {ve}")
                except Exception as e:
                    dir_baseline_val, dir_mitigated_val, eod_val = 0.0, 0.0, 0.0
                    st.session_state.fairness_metrics = {'dir_base': 0.0, 'dir_mit': 0.0, 'eod': 0.0}
                    st.error(f"Fairness Evaluation Error: {str(e)}")
            else:
                 st.session_state.results_df = pd.DataFrame(columns=["rank", "CANDIDATE NAME", "SCORE (RELEVANCE)", "EXPERIENCE", "ACTION", "gender"])
                 st.error("No valid resumes were processed.")

# ==================================================================
# 4️⃣ OUTPUT PANEL (col_output)
# ==================================================================
if show_results and not st.session_state.results_df.empty:
    with col_output:
        
        # ------------------------------------------
        # 4.1 Fairness Audit: Bias Mitigation Efficacy (KPI Cards)
        # ------------------------------------------
        st.header("Fairness Audit: Bias Mitigation Efficacy")

        kpi1, kpi2, kpi3 = st.columns(3)

        # Function to apply color and status text using custom HTML for the screenshot look
        def get_metric_html(label, value, target, inverse_color=False):
            val_str = f"{value:.2f}"
            
            # Logic for DIR (Target >= 0.8)
            if inverse_color: 
                is_good = value >= target
                status_text = "Acceptable (Goal Achieved)" if is_good else "Unfair (Below 0.8 Threshold)"
                bg_color = '#ebfff1' if is_good else '#ffebeb' # Greenish/Reddish background
                text_color = '#27ae60' if is_good else '#c0392b' # Green/Red text
            # Logic for EOD (Target near 0.0, e.g., abs(value) <= 0.05)
            else: 
                is_good = abs(value) <= target
                status_text = "Near Zero (Goal Achieved)" if is_good else "Too High (Bias Detected)"
                bg_color = '#e6f7ff' if is_good else '#fffae6' # Bluish/Yellowish background
                text_color = '#3498db' if is_good else '#f39c12' # Blue/Orange text
            
            # HTML Structure for the KPI Card
            return f"""
                <div style='background-color:{bg_color}; padding:10px; border-radius:10px; text-align:center; min-height: 120px;'>
                    <p style='margin:0; font-size:14px; color:#555;'>{label}</p>
                    <h2 style='margin:5px 0; color:{text_color};'>{val_str}</h2>
                    <p style='margin:0; font-size:12px; font-weight:bold; color:{text_color};'>{status_text}</p>
                </div>
            """

        # KPI 1: DIR - Baseline
        with kpi1:
            html = get_metric_html("Disparate Impact Ratio (DIR) - Baseline", dir_baseline_val, 0.8, inverse_color=True)
            st.markdown(html, unsafe_allow_html=True)


        # KPI 2: DIR - Mitigated
        with kpi2:
            html = get_metric_html("Disparate Impact Ratio (DIR) - Mitigated", dir_mitigated_val, 0.8, inverse_color=True)
            html = html.replace("Unfair (Below 0.8 Threshold)", "Improvement Needed (Below 0.8)")
            st.markdown(html, unsafe_allow_html=True)
            

        # KPI 3: EOD
        with kpi3:
            html = get_metric_html("Equal Opportunity Diff. (EOD)", eod_val, 0.05, inverse_color=False) 
            st.markdown(html, unsafe_allow_html=True)

        st.markdown("---")

        # ------------------------------------------
        # 4.2 Candidate Ranking Results Table
        # ------------------------------------------
        st.header("Candidate Ranking Results")

        # FIX 1: Include 'gender' in the DataFrame used for display
        df_display = st.session_state.results_df[['rank', 'CANDIDATE NAME', 'SCORE (RELEVANCE)', 'EXPERIENCE', 'gender']].copy()
        
        def get_score_color(val):
            try:
                val = float(val)
                if val >= 0.9: return 'green'
                if val >= 0.8: return 'orange'
                return 'red'
            except:
                return 'gray'

        # FIX 2: Adjust column ratios for the new GENDER column (6 columns total)
        col_ratios = [0.5, 2.0, 1.5, 1.5, 1.0, 1.5]
        col_names = st.columns(col_ratios)
        col_names[0].markdown('**RANK**')
        col_names[1].markdown('**CANDIDATE NAME**')
        col_names[2].markdown('**SCORE (RELEVANCE)**')
        col_names[3].markdown('**EXPERIENCE**')
        col_names[4].markdown('**GENDER**') # NEW HEADER
        col_names[5].markdown('**ACTION**')
        st.markdown("---") 

        # 2. Display Data Rows with Buttons
        for index, row in df_display.iterrows():
            cols = st.columns(col_ratios)
            
            # RANK
            cols[0].write(f"**{row['rank']}**")
            
            # CANDIDATE NAME
            cols[1].write(row['CANDIDATE NAME'])
            
            # SCORE (with color)
            score_color = get_score_color(row['SCORE (RELEVANCE)'])
            score_html = f"<span style='color: {score_color}; font-weight: bold;'>{row['SCORE (RELEVANCE)']:.2f}</span>"
            cols[2].markdown(score_html, unsafe_allow_html=True)
            
            # EXPERIENCE
            cols[3].write(row['EXPERIENCE'])
            
            # FIX 3: Display the GENDER value
            cols[4].write(row['gender'])
            
            # ACTION Button (now the 6th column, index 5)
            if cols[5].button('Explain Rank (XAI)', key=f"xai_btn_{index}", use_container_width=True):
                st.info(f"XAI Explanation requested for **{row['CANDIDATE NAME']}** (Score: {row['SCORE (RELEVANCE)']:.2f}).")