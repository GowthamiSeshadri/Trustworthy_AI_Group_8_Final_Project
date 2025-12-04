import fitz  # PyMuPDF
import docx2txt
import numpy as np
import re
from sentence_transformers import SentenceTransformer, util
import io
import streamlit as st # <-- Necessary import

# ==========================================
# 🧠 Load SBERT Model Once (Using st.session_state)
# ==========================================
# Check if the model is already loaded (to prevent Streamlit re-loading on every rerun)
if 'model' not in st.session_state:
    try:
        # st.spinner is used to show a loading message in the Streamlit app
        with st.spinner("Loading AI Ranking Model..."):
            st.session_state.model = SentenceTransformer("all-MiniLM-L6-v2")
            print("✅ SentenceTransformer model loaded successfully.")
    except Exception as e:
        print(f"Error loading SentenceTransformer model: {e}")
        st.session_state.model = None

model = st.session_state.model

# ==========================================
# 📄 Extract text from PDF/DOCX/TXT
# ==========================================
def extract_text(uploaded_file):
    if uploaded_file is None:
        return ""

    # Reset file pointer to the beginning for reading
    uploaded_file.seek(0)
    filename = uploaded_file.name.lower()
    
    # Convert file object to a BytesIO stream for better compatibility
    file_stream = io.BytesIO(uploaded_file.read())

    # ---------- PDF ----------
    if filename.endswith(".pdf"):
        try:
            # fitz.open requires the file to be a path or a byte stream
            # We read the file stream into bytes before opening with fitz
            text = ""
            with fitz.open(stream=file_stream.read(), filetype="pdf") as pdf:
                for page in pdf:
                    text += page.get_text()
                return text.strip()
        except Exception:
            return ""

    # ---------- DOCX ----------
    if filename.endswith(".docx"):
        try:
            # docx2txt can handle a file stream directly
            text = docx2txt.process(file_stream)
            return text.strip()
        except Exception:
            return ""

    # ---------- TXT ----------
    if filename.endswith(".txt"):
        try:
            # Must read bytes and decode
            return file_stream.read().decode("utf-8", errors="ignore").strip()
        except Exception:
            return ""

    return ""


# ==========================================
# 🧠 Sentence Embedding Function
# ==========================================
def get_embeddings(text):
    if model is None:
        raise ConnectionError("Sentence Transformer model failed to load.")
        
    # Remove excessive newlines and clean text
    cleaned_text = re.sub(r'\n+', ' ', text).strip()
    if not cleaned_text:
        return np.array([])
        
    # Use the model loaded from session state
    return model.encode(cleaned_text, convert_to_tensor=True)


# ==========================================
# 📊 Cosine Similarity
# ==========================================
def compute_similarity(jd_vec, resume_vec):
    if jd_vec.size == 0 or resume_vec.size == 0:
        return 0.0
        
    try:
        # Cosine similarity returns a tensor, we convert it to a float
        sim = util.cos_sim(jd_vec, resume_vec)
        # Scale to 0-1 range and return
        return float(sim[0][0])
    except:
        return 0.0

# ==========================================
# 📅 Experience Extraction (Basic Regex)
# ==========================================
def extract_experience(text):
    if not text:
        return None
        
    text_lower = text.lower()

    # Look for patterns like "5 years experience" or "10+ years in"
    match = re.search(r'(\d+)\+?\s*years?\s*(?:of|in|exp|experience)', text_lower)
    
    if match:
        return int(match.group(1))
    
    # Simple fallback
    simple_match = re.search(r'(\d{1,2})\s*years?\b', text_lower)
    if simple_match:
        return int(simple_match.group(1))

    return None

# ==========================================
# 👤 Gender Detection (IMPROVED NAME FOCUS)
# ==========================================

# Expanded name lists (add more names common to your sample data)
FEMALE_NAMES = set([
    "emma", "olivia", "ava", "sophia", "isabella", "mia", "charlotte", "amelia",
    "ella", "grace", "sarah", "emily", "hannah", "sofia", "layla",
    "pallavi", "sudha", "gowthami", "mary", "priya", "sita", "kavya", 
    "fatima", "aisha", "samantha", "jessica", "maya", "sara", "anjali"
])

MALE_NAMES = set([
    "liam", "noah", "oliver", "elijah", "james", "william", "benjamin",
    "lucas", "henry", "alex", "john", "michael", "robert", "david",
    "akil", "akhil", "mike", "rafi", "prushotham", "suresh", "ram", 
    "ahmed", "ali", "chris", "thomas", "ryan", "jay", "vikram"
])


def detect_gender(text):
    if not text:
        return "Unknown"

    text_lower = text.lower()
    
    # Strategy: Assume the first non-empty line contains the name
    first_line = text.strip().split('\n', 1)[0]
    
    # Clean the line to get the first word (likely the first name)
    words = re.findall(r'\b[a-z]+\b', first_line.lower())
    
    if not words:
        return "Unknown"
        
    # Check the first word in the first line against the name lists
    first_name = words[0]
    
    if first_name in FEMALE_NAMES:
        return "Female"
    elif first_name in MALE_NAMES:
        return "Male"
    else:
        # Fallback to check if any known name appears near the beginning
        # (Useful for resumes where the name might be capitalized or formatted oddly)
        # Check first 500 characters
        search_text = text_lower[:500]
        
        for name in FEMALE_NAMES:
            if name in search_text:
                return "Female"

        for name in MALE_NAMES:
            if name in search_text:
                return "Male"
                
        return "Unknown"