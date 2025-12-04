import fitz  # PyMuPDF
import docx2txt
import re
import gender_guesser.detector as gender
from sentence_transformers import SentenceTransformer, util

# Load SBERT Model once
model = SentenceTransformer("all-MiniLM-L6-v2")

# Load gender detector
gender_detector = gender.Detector()


# -------------------------------
# 1. Extract text from PDF or DOCX
# -------------------------------
def extract_text(file):
    text = ""

    # If PDF
    if file.name.lower().endswith(".pdf"):
        pdf = fitz.open(stream=file.read(), filetype="pdf")
        for page in pdf:
            text += page.get_text()

    # If DOCX
    elif file.name.lower().endswith(".docx"):
        text = docx2txt.process(file)

    else:
        try:
            text = file.read().decode("utf-8", errors="ignore")
        except:
            text = ""

    return text.strip()


# -------------------------------
# 2. Compute SBERT cosine similarity
# -------------------------------
def compute_similarity(jd_text, resume_text):
    if not jd_text or not resume_text:
        return 0.0

    jd_emb = model.encode(jd_text, convert_to_tensor=True)
    res_emb = model.encode(resume_text, convert_to_tensor=True)

    score = float(util.cos_sim(jd_emb, res_emb)[0])

    return round(score, 4)


# -------------------------------
# 3. Extract name from filename
# -------------------------------
def extract_name(filename):
    name = filename.replace(".pdf", "").replace(".docx", "")
    name = re.sub(r"[_\-0-9]+", " ", name).strip()
    return name


# -------------------------------
# 4. Gender Detection
# -------------------------------
def detect_gender(filename, resume_text):
    # Try detecting from filename first
    name = extract_name(filename)
    first = name.split()[0]

    g = gender_detector.get_gender(first)

    if g in ["male", "mostly_male"]:
        return "Male"
    if g in ["female", "mostly_female"]:
        return "Female"

    # If still unknown → search resume text for pronouns
    resume_text = resume_text.lower()

    female_keywords = [" she ", " her ", " ms ", " mrs ", " woman", " female"]
    male_keywords = [" he ", " him ", " mr ", " man ", " male"]

    if any(k in resume_text for k in female_keywords):
        return "Female"
    if any(k in resume_text for k in male_keywords):
        return "Male"

    return "Unknown"
