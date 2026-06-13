import streamlit as st
import pickle
import re
from PIL import Image
import pytesseract
from scipy.sparse import hstack, csr_matrix
import numpy as np

# ── Windows Tesseract path (update if yours differs) ─────────────────────────
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Fake Review Detector", page_icon="🕵️", layout="centered")

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Syne:wght@700;800&family=DM+Sans:wght@400;500&display=swap');
    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
    h1, h2, h3 { font-family: 'Syne', sans-serif; }
    .stApp { background: #0d0d0d; color: #f0ede8; }
    .card { background: #181818; border: 1px solid #2a2a2a; border-radius: 16px; padding: 2rem; margin-bottom: 1.5rem; }
    .result-genuine { background: linear-gradient(135deg,#0a2e1a,#0d3d22); border:1px solid #1a6b3a; border-radius:12px; padding:1.5rem 2rem; text-align:center; }
    .result-fake    { background: linear-gradient(135deg,#2e0a0a,#3d0d0d); border:1px solid #6b1a1a; border-radius:12px; padding:1.5rem 2rem; text-align:center; }
    .result-uncertain { background: linear-gradient(135deg,#1a1a0a,#2e2a0d); border:1px solid #6b5e1a; border-radius:12px; padding:1.5rem 2rem; text-align:center; }
    .result-title { font-family:'Syne',sans-serif; font-size:1.6rem; font-weight:800; }
    .result-sub { font-size:0.9rem; opacity:0.7; margin-top:0.3rem; }
    .extracted-box { background:#111; border:1px dashed #333; border-radius:10px; padding:1rem 1.2rem; font-size:0.85rem; color:#aaa; white-space:pre-wrap; max-height:160px; overflow-y:auto; }
    .stButton>button { background:#f0ede8; color:#0d0d0d; border:none; border-radius:8px; font-family:'Syne',sans-serif; font-weight:700; font-size:1rem; padding:0.6rem 2rem; width:100%; }
    .stButton>button:hover { opacity:0.85; }
    .stTextArea textarea { background:#111 !important; color:#f0ede8 !important; border:1px solid #2a2a2a !important; border-radius:10px !important; }
    .confidence-bar-bg { background:#222; border-radius:100px; height:8px; margin-top:0.8rem; }
    .confidence-bar-fill { height:8px; border-radius:100px; }
</style>
""", unsafe_allow_html=True)


# ── Load model artefacts ──────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    model           = pickle.load(open("model.pkl",           "rb"))
    vectorizer      = pickle.load(open("vectorizer.pkl",      "rb"))
    char_vectorizer = pickle.load(open("char_vectorizer.pkl", "rb"))
    use_rating      = pickle.load(open("use_rating.pkl",      "rb"))
    return model, vectorizer, char_vectorizer, use_rating

model, vectorizer, char_vectorizer, use_rating = load_model()


# ── Helpers ───────────────────────────────────────────────────────────────────
def clean_text(text: str) -> str:
    text = str(text).lower()
    contractions = {
        "won't":"will not","can't":"cannot","n't":" not",
        "'re":" are","'s":" is","'d":" would",
        "'ll":" will","'ve":" have","'m":" am"
    }
    for k, v in contractions.items():
        text = text.replace(k, v)
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def predict(text: str, rating: float = 3.0):
    cleaned   = clean_text(text)
    word_vec  = vectorizer.transform([cleaned])
    char_vec  = char_vectorizer.transform([cleaned])
    combined  = hstack([word_vec, char_vec])
    if use_rating:
        rating_vec = csr_matrix(np.array([[rating]]))
        combined   = hstack([combined, rating_vec])
    label      = model.predict(combined)[0]
    proba      = model.predict_proba(combined)[0]
    classes    = model.classes_.tolist()
    confidence = proba[classes.index(label)]
    return label, confidence


def extract_text_from_image(image_file) -> str:
    img = Image.open(image_file)
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")
    return pytesseract.image_to_string(img).strip()


def render_result(label: str, confidence: float, source: str = ""):
    pct = int(confidence * 100)
    low_confidence = confidence < 0.75

    if low_confidence:
        css_class = "result-uncertain"
        icon      = "⚠️"
        verdict   = "Uncertain — Review is borderline"
        bar_color = "#b8960c"
    elif label == "CG":
        css_class = "result-genuine"
        icon      = "✅"
        verdict   = "Genuine Review"
        bar_color = "#1a9950"
    else:
        css_class = "result-fake"
        icon      = "❌"
        verdict   = "Fake Review"
        bar_color = "#c0392b"

    st.markdown(f"""
    <div class="{css_class}">
        <div class="result-title">{icon} {verdict}</div>
        <div class="result-sub">{source}Model confidence: {pct}%</div>
        <div class="confidence-bar-bg">
            <div class="confidence-bar-fill" style="width:{pct}%; background:{bar_color};"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if low_confidence:
        st.caption("The model isn't confident enough to make a strong call. Consider checking the review manually.")


# ── UI ────────────────────────────────────────────────────────────────────────
st.markdown("<h1 style='font-size:2.4rem; margin-bottom:0.2rem;'>🕵️ Fake Review Detector</h1>", unsafe_allow_html=True)
st.markdown("<p style='color:#888; margin-bottom:2rem;'>Paste a review or upload a screenshot — we'll tell you if it's real.</p>", unsafe_allow_html=True)

tab_text, tab_image = st.tabs(["✏️  Type / Paste Review", "🖼️  Upload Screenshot"])

# ── Tab 1: Text ───────────────────────────────────────────────────────────────
with tab_text:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    user_input = st.text_area("Review text", placeholder="Paste the review here…", height=160, label_visibility="collapsed")

    rating_input = 3.0
    if use_rating:
        rating_input = st.slider("Star rating of the review", 1.0, 5.0, 3.0, 0.5)

    check_text = st.button("Analyse Review", key="btn_text")
    st.markdown("</div>", unsafe_allow_html=True)

    if check_text:
        if not user_input.strip():
            st.warning("Please enter some review text first.")
        else:
            label, confidence = predict(user_input, rating_input)
            render_result(label, confidence)

# ── Tab 2: Image ──────────────────────────────────────────────────────────────
with tab_image:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<p style='color:#888; font-size:0.9rem; margin-bottom:1rem;'>Upload any screenshot (PNG, JPG, WEBP). Text is extracted automatically via OCR.</p>", unsafe_allow_html=True)
    uploaded = st.file_uploader("Upload screenshot", type=["png", "jpg", "jpeg", "webp"], label_visibility="collapsed")
    if uploaded:
        st.image(uploaded, caption="Uploaded image", use_column_width=True)

    img_rating = 3.0
    if use_rating:
        img_rating = st.slider("Star rating (if visible in screenshot)", 1.0, 5.0, 3.0, 0.5, key="img_rating")

    check_image = st.button("Extract & Analyse", key="btn_image")
    st.markdown("</div>", unsafe_allow_html=True)

    if check_image:
        if not uploaded:
            st.warning("Please upload an image first.")
        else:
            with st.spinner("Reading text from image…"):
                try:
                    extracted = extract_text_from_image(uploaded)
                except Exception as e:
                    st.error(f"OCR failed: {e}")
                    st.stop()

            if not extracted.strip():
                st.warning("Could not extract text. Try a clearer screenshot.")
            else:
                st.markdown("**Extracted text:**")
                st.markdown(f"<div class='extracted-box'>{extracted}</div>", unsafe_allow_html=True)
                st.markdown("<br>", unsafe_allow_html=True)
                label, confidence = predict(extracted, img_rating)
                render_result(label, confidence, source="OCR → ")

st.markdown("<br><p style='text-align:center;color:#444;font-size:0.8rem;'>Ensemble: LinearSVC + Logistic Regression + Naive Bayes · OCR via Tesseract</p>", unsafe_allow_html=True)
