# ==============================================================================
# AI-Powered Resume Parser & Skill Matcher
#
# INSTRUCTION FOR SPACY:
# To download the required spaCy model, run the following command in your terminal:
#     python -m spacy download en_core_web_sm
# ==============================================================================

import json
import streamlit as st
import pandas as pd
from utils.parser import (
    extract_text_from_file,
    extract_contact_info,
    extract_name,
    extract_skills,
    get_spacy_model,
    DEFAULT_SKILLS
)
from utils.matcher import (
    calculate_semantic_score,
    compare_skills
)

# Page Configuration
st.set_page_config(
    page_title="AI Resume Parser & Skill Matcher",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------------------------------------------------------------------
# Refined Glassmorphism Light UI Theme (Zero Neon, Elegant Palette)
# ------------------------------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

    /* Global Light Glassmorphism Base */
    html, body, [class*="css"], .stApp {
        background-color: #F8FAFC !important;
        background-image: 
            radial-gradient(at 12% 15%, rgba(219, 234, 254, 0.5) 0px, transparent 55%),
            radial-gradient(at 88% 18%, rgba(224, 231, 255, 0.45) 0px, transparent 50%),
            radial-gradient(at 50% 85%, rgba(241, 245, 249, 0.6) 0px, transparent 65%) !important;
        color: #0F172A !important;
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
    }

    /* Force Light Theme on all Streamlit Typography */
    h1, h2, h3, h4, h5, h6, p, span, label, div {
        color: #0F172A;
    }

    /* Sidebar Glassmorphism */
    [data-testid="stSidebar"] {
        background: rgba(255, 255, 255, 0.78) !important;
        backdrop-filter: blur(20px) !important;
        -webkit-backdrop-filter: blur(20px) !important;
        border-right: 1px solid rgba(226, 232, 240, 0.8) !important;
        box-shadow: 2px 0 20px rgba(15, 23, 42, 0.02) !important;
    }

    /* Glassmorphic Surface Cards */
    .glass-panel {
        background: rgba(255, 255, 255, 0.72) !important;
        backdrop-filter: blur(16px) !important;
        -webkit-backdrop-filter: blur(16px) !important;
        border: 1px solid rgba(255, 255, 255, 0.85) !important;
        border-radius: 16px !important;
        box-shadow: 0 4px 20px -2px rgba(15, 23, 42, 0.04), inset 0 1px 0 rgba(255, 255, 255, 0.9) !important;
        padding: 1.5rem;
        transition: all 0.3s cubic-bezier(0.16, 1, 0.3, 1);
    }
    .glass-panel:hover {
        background: rgba(255, 255, 255, 0.85) !important;
        border-color: rgba(203, 213, 225, 0.9) !important;
        box-shadow: 0 10px 30px -4px rgba(15, 23, 42, 0.07), inset 0 1px 0 rgba(255, 255, 255, 1) !important;
        transform: translateY(-2px);
    }

    /* Hero Banner */
    .hero-banner {
        background: rgba(255, 255, 255, 0.8) !important;
        backdrop-filter: blur(20px) !important;
        -webkit-backdrop-filter: blur(20px) !important;
        border: 1px solid rgba(255, 255, 255, 0.9) !important;
        border-radius: 20px !important;
        padding: 2.2rem 2.4rem;
        margin-bottom: 1.8rem;
        box-shadow: 0 8px 32px -4px rgba(15, 23, 42, 0.05), inset 0 1px 0 rgba(255, 255, 255, 0.95);
        position: relative;
        overflow: hidden;
    }
    .hero-banner::before {
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: linear-gradient(90deg, #3B82F6, #6366F1, #0EA5E9);
    }
    .hero-tag {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(239, 246, 255, 0.9);
        color: #1D4ED8;
        border: 1px solid #BFDBFE;
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.03em;
        text-transform: uppercase;
        margin-bottom: 0.8rem;
    }
    .hero-title {
        font-size: 2.3rem;
        font-weight: 800;
        color: #0F172A;
        letter-spacing: -0.03em;
        line-height: 1.2;
        margin-bottom: 0.5rem;
    }
    .hero-accent {
        color: #2563EB;
    }
    .hero-desc {
        font-size: 1.02rem;
        color: #475569;
        max-width: 760px;
        line-height: 1.55;
        margin: 0;
    }

    /* -------------------------------------------------------------------------
       FORCING STREAMLIT NATIVE INPUTS TO ELEGANT LIGHT GLASS STYLES
       ------------------------------------------------------------------------- */
    /* File Uploader Dropzone */
    [data-testid="stFileUploader"] {
        background: rgba(255, 255, 255, 0.65) !important;
        backdrop-filter: blur(14px) !important;
        -webkit-backdrop-filter: blur(14px) !important;
        border-radius: 16px !important;
        border: 1px solid rgba(226, 232, 240, 0.85) !important;
        padding: 0.6rem !important;
        box-shadow: 0 2px 12px rgba(15, 23, 42, 0.02) !important;
    }
    [data-testid="stFileUploader"] section {
        background: rgba(255, 255, 255, 0.7) !important;
        border: 1.5px dashed #CBD5E1 !important;
        border-radius: 12px !important;
        padding: 1.2rem 1rem !important;
    }
    [data-testid="stFileUploader"] section:hover {
        border-color: #3B82F6 !important;
        background: rgba(255, 255, 255, 0.95) !important;
    }
    [data-testid="stFileUploader"] * {
        color: #334155 !important;
    }
    [data-testid="stFileUploader"] button {
        background: #FFFFFF !important;
        color: #0F172A !important;
        border: 1px solid #CBD5E1 !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04) !important;
    }
    [data-testid="stFileUploader"] button:hover {
        border-color: #3B82F6 !important;
        color: #2563EB !important;
    }

    /* Text Area */
    [data-testid="stTextArea"] textarea {
        background: rgba(255, 255, 255, 0.8) !important;
        backdrop-filter: blur(14px) !important;
        -webkit-backdrop-filter: blur(14px) !important;
        border: 1px solid rgba(203, 213, 225, 0.85) !important;
        border-radius: 14px !important;
        color: #0F172A !important;
        font-size: 0.93rem !important;
        line-height: 1.55 !important;
        box-shadow: 0 2px 8px rgba(15, 23, 42, 0.02) !important;
        transition: all 0.2s ease !important;
    }
    [data-testid="stTextArea"] textarea:focus {
        border-color: #3B82F6 !important;
        box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.15) !important;
        background: #FFFFFF !important;
    }

    /* Text Inputs */
    .stTextInput input {
        background: rgba(255, 255, 255, 0.85) !important;
        border: 1px solid #CBD5E1 !important;
        border-radius: 10px !important;
        color: #0F172A !important;
    }
    .stTextInput input:focus {
        border-color: #3B82F6 !important;
        box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.15) !important;
        background: #FFFFFF !important;
    }

    /* Buttons: Secondary / Demo Buttons */
    .stButton button {
        background: rgba(255, 255, 255, 0.85) !important;
        backdrop-filter: blur(12px) !important;
        -webkit-backdrop-filter: blur(12px) !important;
        color: #1E293B !important;
        border: 1px solid #CBD5E1 !important;
        border-radius: 11px !important;
        font-weight: 600 !important;
        font-size: 0.9rem !important;
        padding: 0.55rem 1.2rem !important;
        box-shadow: 0 1px 3px rgba(15, 23, 42, 0.03) !important;
        transition: all 0.2s cubic-bezier(0.16, 1, 0.3, 1) !important;
    }
    .stButton button:hover {
        background: #FFFFFF !important;
        color: #1D4ED8 !important;
        border-color: #93C5FD !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 18px rgba(37, 99, 235, 0.1) !important;
    }

    /* Primary Analyze Button */
    .stButton button[kind="primary"] {
        background: linear-gradient(135deg, #2563EB 0%, #1D4ED8 100%) !important;
        color: #FFFFFF !important;
        border: none !important;
        box-shadow: 0 4px 14px rgba(37, 99, 235, 0.28) !important;
        font-weight: 700 !important;
    }
    .stButton button[kind="primary"]:hover {
        background: linear-gradient(135deg, #1D4ED8 0%, #1E40AF 100%) !important;
        color: #FFFFFF !important;
        box-shadow: 0 8px 22px rgba(37, 99, 235, 0.38) !important;
        transform: translateY(-2px) !important;
    }

    /* Skill Badges (Non-Neon, Sophisticated Palette) */
    .skill-chip {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 6px 13px;
        border-radius: 9999px;
        font-size: 0.86rem;
        font-weight: 600;
        margin: 3px;
        transition: all 0.2s cubic-bezier(0.34, 1.56, 0.64, 1);
        cursor: default;
        user-select: none;
    }
    .skill-chip:hover {
        transform: translateY(-2px) scale(1.04);
        box-shadow: 0 4px 12px rgba(15, 23, 42, 0.06);
    }

    /* Matched: Calm Muted Sage */
    .chip-matched {
        background: rgba(240, 253, 244, 0.9) !important;
        color: #166534 !important;
        border: 1px solid #BBF7D0 !important;
    }

    /* Missing: Soft Terracotta Rose */
    .chip-missing {
        background: rgba(254, 242, 242, 0.9) !important;
        color: #991B1B !important;
        border: 1px solid #FECACA !important;
    }

    /* Bonus: Subtle Slate */
    .chip-bonus {
        background: rgba(248, 250, 252, 0.9) !important;
        color: #334155 !important;
        border: 1px solid #CBD5E1 !important;
    }

    /* Stat Box */
    .stat-pill {
        background: rgba(255, 255, 255, 0.75);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(226, 232, 240, 0.9);
        border-radius: 14px;
        padding: 1.1rem;
        text-align: center;
        box-shadow: 0 2px 10px rgba(15, 23, 42, 0.02);
        transition: transform 0.2s ease, background 0.2s ease;
    }
    .stat-pill:hover {
        transform: translateY(-2px);
        background: #FFFFFF;
        box-shadow: 0 8px 20px rgba(15, 23, 42, 0.05);
    }
    .stat-val {
        font-size: 1.85rem;
        font-weight: 800;
        color: #0F172A;
        line-height: 1.1;
    }
    .stat-lbl {
        font-size: 0.78rem;
        font-weight: 700;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        margin-top: 5px;
    }

    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: rgba(241, 245, 249, 0.6);
        padding: 6px;
        border-radius: 12px;
        border: 1px solid rgba(226, 232, 240, 0.8);
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 8px 16px;
        font-weight: 600;
        color: #64748B;
        background: transparent;
    }
    .stTabs [aria-selected="true"] {
        background: #FFFFFF !important;
        color: #1D4ED8 !important;
        box-shadow: 0 2px 8px rgba(15, 23, 42, 0.05) !important;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def load_nlp():
    """Cached loader for the spaCy language model."""
    return get_spacy_model("en_core_web_sm")


def render_skill_chips(items, chip_type="matched"):
    """Renders calm, elegant, non-neon skill chips."""
    if not items:
        return '<div style="color: #94A3B8; font-style: italic; padding: 0.4rem 0;">None detected</div>'

    icon_map = {
        "matched": "✓",
        "missing": "✕",
        "bonus": "•"
    }
    class_map = {
        "matched": "chip-matched",
        "missing": "chip-missing",
        "bonus": "chip-bonus"
    }

    icon = icon_map.get(chip_type, "•")
    css_class = class_map.get(chip_type, "chip-matched")

    chips = []
    for item in items:
        chips.append(
            f'<span class="skill-chip {css_class}">'
            f'<span style="opacity: 0.7; font-size: 0.85em;">{icon}</span> {item}'
            f'</span>'
        )
    return "".join(chips)


def render_radial_gauge(score: float):
    """Generates an elegant, non-neon circular compatibility gauge."""
    circumference = 2 * 3.14159 * 52
    clamped_score = max(0.0, min(100.0, score))
    offset = circumference - (clamped_score / 100.0) * circumference

    if clamped_score >= 70:
        stroke_color = "#2563EB"  # Classic royal blue
        badge_bg = "rgba(239, 246, 255, 0.9)"
        badge_text = "#1D4ED8"
        label = "Strong Fit"
    elif clamped_score >= 40:
        stroke_color = "#D97706"  # Soft amber
        badge_bg = "rgba(254, 243, 199, 0.9)"
        badge_text = "#92400E"
        label = "Moderate Fit"
    else:
        stroke_color = "#DC2626"  # Muted crimson
        badge_bg = "rgba(254, 242, 242, 0.9)"
        badge_text = "#991B1B"
        label = "Low Fit"

    svg_html = f"""
    <div style="display: flex; flex-direction: column; align-items: center; justify-content: center; padding: 0.5rem 0;">
        <div style="position: relative; width: 130px; height: 130px; display: flex; align-items: center; justify-content: center;">
            <svg width="130" height="130" viewBox="0 0 130 130" style="transform: rotate(-90deg);">
                <circle cx="65" cy="65" r="52" stroke="#E2E8F0" stroke-width="10" fill="transparent" />
                <circle cx="65" cy="65" r="52" stroke="{stroke_color}" stroke-width="10" 
                        stroke-dasharray="{circumference}" stroke-dashoffset="{offset}" 
                        stroke-linecap="round" fill="transparent" 
                        style="transition: stroke-dashoffset 1.2s cubic-bezier(0.16, 1, 0.3, 1);" />
            </svg>
            <div style="position: absolute; text-align: center;">
                <div style="font-size: 1.85rem; font-weight: 800; color: #0F172A; line-height: 1;">
                    {clamped_score:.1f}<span style="font-size: 1rem; font-weight: 600; color: #64748B;">%</span>
                </div>
            </div>
        </div>
        <div style="margin-top: 10px; background: {badge_bg}; color: {badge_text}; padding: 4px 14px; border-radius: 9999px; font-weight: 700; font-size: 0.82rem; border: 1px solid rgba(0,0,0,0.05);">
            {label}
        </div>
    </div>
    """
    return svg_html


def generate_text_report(name, email, phone, score, matched_skills, missing_skills, resume_skills):
    """Generates an exportable plain-text summary report."""
    return f"""==================================================
AI RESUME PARSER & SKILL MATCH REPORT
==================================================

CANDIDATE PROFILE:
------------------
Name:  {name}
Email: {email}
Phone: {phone}

MATCH OVERVIEW:
---------------
Overall Compatibility Score: {score}%
Matched Skills Count:        {len(matched_skills)}
Missing Skills Count:        {len(missing_skills)}

SKILLS BREAKDOWN:
-----------------
Matched Skills ({len(matched_skills)}):
{", ".join(matched_skills) if matched_skills else "None"}

Missing Skills ({len(missing_skills)}):
{", ".join(missing_skills) if missing_skills else "None"}

All Identified Candidate Skills ({len(resume_skills)}):
{", ".join(resume_skills) if resume_skills else "None"}

==================================================
Generated via AI Resume Parser & Skill Matcher
==================================================
"""


def main():
    # --------------------------------------------------------------------------
    # Sidebar
    # --------------------------------------------------------------------------
    with st.sidebar:
        st.markdown("### ⚙️ System Engine")
        st.markdown(
            """
            <div style="background: rgba(240, 253, 244, 0.8); border: 1px solid #BBF7D0; padding: 0.75rem 1rem; border-radius: 12px; margin-bottom: 1rem;">
                <div style="display: flex; align-items: center; gap: 8px;">
                    <span style="display: inline-block; width: 8px; height: 8px; border-radius: 50%; background: #16A34A;"></span>
                    <span style="font-weight: 700; color: #166534; font-size: 0.85rem;">NLP Pipeline Active</span>
                </div>
                <div style="font-size: 0.75rem; color: #15803D; margin-top: 3px;">spaCy NER & TF-IDF Matching</div>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.subheader("💡 Setup Command")
        st.code("python -m spacy download en_core_web_sm", language="bash")
        st.caption("Required for Named Entity Recognition.")

        st.divider()

        st.subheader("📚 Skill Repository")
        st.markdown(f"**{len(DEFAULT_SKILLS)}** standardized technical competencies available:")
        search_filter = st.text_input("🔍 Search Skill Repository", placeholder="e.g. Python, Docker, React")
        filtered_catalog = [s for s in DEFAULT_SKILLS if search_filter.lower() in s.lower()] if search_filter else DEFAULT_SKILLS

        with st.expander(f"Browse Skills ({len(filtered_catalog)})", expanded=False):
            st.write(", ".join(sorted(filtered_catalog)))

        st.divider()
        st.caption("Resume Parser & Matcher • Clean Glassmorphic Theme")

    # --------------------------------------------------------------------------
    # Hero Banner
    # --------------------------------------------------------------------------
    st.markdown(
        """
        <div class="hero-banner">
            <div class="hero-tag">✨ Talent Intelligence AI</div>
            <div class="hero-title">Resume Parser <span class="hero-accent">&amp; Skill Matcher</span></div>
            <p class="hero-desc">
                Evaluate candidate documents against target job specifications with intelligent NLP parsing,
                entity extraction, and vector-based semantic compatibility scoring.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Demo quick-loader buttons
    demo_cols = st.columns([1.6, 1.2, 1.2], gap="small")
    with demo_cols[0]:
        st.markdown("<p style='font-weight:600; color:#475569; margin-top:8px;'>⚡ Quick Test with Demo Data:</p>", unsafe_allow_html=True)
    with demo_cols[1]:
        demo_fs = st.button("🚀 Demo: Full-Stack Dev", use_container_width=True)
    with demo_cols[2]:
        demo_ds = st.button("🧠 Demo: Data Scientist", use_container_width=True)

    if "sample_resume_text" not in st.session_state:
        st.session_state.sample_resume_text = None
    if "sample_jd_text" not in st.session_state:
        st.session_state.sample_jd_text = ""

    if demo_fs:
        st.session_state.sample_resume_text = (
            "Sophia Martinez\n"
            "sophia.martinez@clouddev.io | +1 (415) 782-9901 | San Francisco, CA\n\n"
            "PROFESSIONAL SUMMARY\n"
            "Lead Full Stack Engineer with 7+ years of experience engineering scalable microservices.\n"
            "Proficient in Python, Django, React, TypeScript, and PostgreSQL.\n\n"
            "TECHNICAL SKILLS\n"
            "Languages: Python, TypeScript, JavaScript, SQL, HTML, CSS, Bash\n"
            "Frameworks: Django, FastAPI, React, Node.js, Express, Next.js, Redux\n"
            "Cloud & DevOps: Docker, Kubernetes, AWS, CI/CD, GitHub Actions, Terraform\n"
            "Databases: PostgreSQL, Redis, MongoDB"
        )
        st.session_state.sample_jd_text = (
            "We are looking for a Senior Full Stack Software Engineer to build robust web services.\n"
            "Requirements:\n"
            "- Strong proficiency in Python, Django, and React.\n"
            "- Experience with Docker, Kubernetes, AWS, and PostgreSQL.\n"
            "- Knowledge of Redis caching and Terraform infrastructure.\n"
            "- Familiarity with GraphQL and GraphQL APIs is a big plus."
        )
        st.toast("Loaded Full-Stack Developer demo data!", icon="⚡")

    if demo_ds:
        st.session_state.sample_resume_text = (
            "Dr. David Chen\n"
            "david.chen@aimodel.org | +1 (206) 555-8823 | Seattle, WA\n\n"
            "PROFESSIONAL SUMMARY\n"
            "Senior Machine Learning Scientist specializing in NLP, Deep Learning, and LLMs.\n"
            "Extensive experience deploying models on AWS with PyTorch, Scikit-Learn, and Docker.\n\n"
            "CORE COMPETENCIES\n"
            "AI/ML: Machine Learning, Deep Learning, Natural Language Processing, NLP, LLMs, LangChain\n"
            "Libraries: PyTorch, TensorFlow, Scikit-Learn, Pandas, NumPy, Hugging Face, spaCy\n"
            "Cloud: AWS, Docker, Linux, Git, CI/CD\n"
            "Databases: PostgreSQL, Elasticsearch"
        )
        st.session_state.sample_jd_text = (
            "We are seeking an AI/ML Engineer to lead our NLP initiatives.\n"
            "Requirements:\n"
            "- Hands-on expertise in Machine Learning, Deep Learning, and NLP.\n"
            "- Strong coding skills in Python, PyTorch, Scikit-Learn, and Pandas.\n"
            "- Experience deploying with Docker, Kubernetes, and AWS.\n"
            "- Knowledge of OpenCV and Computer Vision is desirable."
        )
        st.toast("Loaded Data Scientist demo data!", icon="🧠")

    st.markdown("<br>", unsafe_allow_html=True)

    # --------------------------------------------------------------------------
    # Input Area: Upload & Job Description
    # --------------------------------------------------------------------------
    col_upload, col_jd = st.columns([1, 1], gap="large")

    with col_upload:
        st.markdown(
            """
            <div style="display:flex; align-items:center; gap:8px; margin-bottom: 0.5rem;">
                <span style="font-size:1.25rem;">📄</span>
                <span style="font-weight:700; font-size:1.1rem; color:#0F172A;">1. Candidate Resume</span>
            </div>
            """,
            unsafe_allow_html=True
        )

        uploaded_file = st.file_uploader(
            "Upload resume file",
            type=["pdf", "docx"],
            help="Accepts .pdf (using pdfminer.six) or .docx (using python-docx)",
            label_visibility="collapsed"
        )

        if uploaded_file:
            st.success(f"✓ Uploaded: **{uploaded_file.name}** ({uploaded_file.size / 1024:.1f} KB)")
        elif st.session_state.sample_resume_text:
            st.info("ℹ️ Using loaded demo resume text. (Upload a file above to analyze your own resume)")

    with col_jd:
        st.markdown(
            """
            <div style="display:flex; align-items:center; gap:8px; margin-bottom: 0.5rem;">
                <span style="font-size:1.25rem;">📋</span>
                <span style="font-weight:700; font-size:1.1rem; color:#0F172A;">2. Target Job Description</span>
            </div>
            """,
            unsafe_allow_html=True
        )

        default_jd_value = st.session_state.sample_jd_text or (
            "We are looking for a Software Engineer proficient in Python, React, and PostgreSQL. "
            "Experience with Docker, Kubernetes, AWS, and CI/CD pipelines is required."
        )

        jd_text = st.text_area(
            "Paste Job Description text",
            value=default_jd_value,
            height=165,
            placeholder="Paste job qualifications, required tech stack, and responsibilities...",
            label_visibility="collapsed"
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # Action Button
    analyze_btn = st.button("🚀 Analyze & Match Resume", type="primary", use_container_width=True)

    # --------------------------------------------------------------------------
    # Trigger Analysis
    # --------------------------------------------------------------------------
    if analyze_btn:
        has_file = uploaded_file is not None
        has_sample = bool(st.session_state.sample_resume_text)
        has_resume = has_file or has_sample
        has_jd = bool(jd_text.strip())

        # Edge Case 1: Display warning if inputs are missing without throwing tracebacks
        if not has_resume and not has_jd:
            st.warning("⚠️ Please provide a resume (upload PDF/DOCX or select a Demo button) and paste a Job Description.")
            return

        if not has_resume:
            st.warning("⚠️ Please upload a resume file (PDF or DOCX) to proceed with the analysis.")
            return

        if not has_jd:
            st.warning("⚠️ Please paste a Job Description before analyzing.")
            return

        with st.spinner("Extracting candidate information, running NER, and calculating semantic match..."):
            # Step 1: Text extraction
            try:
                if has_file:
                    resume_text = extract_text_from_file(uploaded_file)
                else:
                    resume_text = st.session_state.sample_resume_text
            except ValueError as ve:
                st.error(f"❌ File Parsing Error: {ve}")
                return
            except Exception as e:
                st.error(f"❌ Unexpected error reading file: {e}")
                return

            if not resume_text.strip():
                st.error("❌ Could not extract readable text from the document.")
                return

            # Step 2: Information & Contact extraction
            nlp = load_nlp()
            candidate_name = extract_name(resume_text, nlp=nlp)
            contact_info = extract_contact_info(resume_text)

            # Step 3: Skill extraction
            resume_skills = extract_skills(resume_text, DEFAULT_SKILLS)
            parsed_jd_skills = extract_skills(jd_text, DEFAULT_SKILLS)

            # Edge Case 2: Fallback if no specific skills were found in JD
            if not parsed_jd_skills:
                st.info("ℹ️ No specific skill keywords detected in the Job Description. Benchmarking against standard technical catalog.")
                effective_jd_skills = DEFAULT_SKILLS
            else:
                effective_jd_skills = parsed_jd_skills

            # Step 4: Semantic Scoring & Skill Comparison
            semantic_score = calculate_semantic_score(resume_text, jd_text)
            matched_skills, missing_skills = compare_skills(resume_skills, effective_jd_skills)
            additional_skills = [s for s in resume_skills if s not in matched_skills]

        # --------------------------------------------------------------------------
        # Results Dashboard
        # --------------------------------------------------------------------------
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(
            """
            <div style="display:flex; align-items:center; justify-content:space-between; margin-bottom: 1.2rem;">
                <h2 style="font-weight: 800; color: #0F172A; margin: 0; font-size: 1.55rem;">
                    🎯 Candidate Analysis Dashboard
                </h2>
                <span style="background: rgba(239, 246, 255, 0.9); color: #1D4ED8; border: 1px solid #BFDBFE; padding: 4px 12px; border-radius: 9999px; font-weight: 700; font-size: 0.8rem;">
                    Screening Completed
                </span>
            </div>
            """,
            unsafe_allow_html=True
        )

        # Overview Metrics Grid
        stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
        with stat_col1:
            st.markdown(
                f"""
                <div class="stat-pill">
                    <div class="stat-val" style="color: #2563EB;">{semantic_score}%</div>
                    <div class="stat-lbl">Compatibility Score</div>
                </div>
                """,
                unsafe_allow_html=True
            )
        with stat_col2:
            st.markdown(
                f"""
                <div class="stat-pill">
                    <div class="stat-val" style="color: #166534;">{len(matched_skills)}</div>
                    <div class="stat-lbl">Matched Skills</div>
                </div>
                """,
                unsafe_allow_html=True
            )
        with stat_col3:
            st.markdown(
                f"""
                <div class="stat-pill">
                    <div class="stat-val" style="color: #991B1B;">{len(missing_skills)}</div>
                    <div class="stat-lbl">Missing Skills</div>
                </div>
                """,
                unsafe_allow_html=True
            )
        with stat_col4:
            st.markdown(
                f"""
                <div class="stat-pill">
                    <div class="stat-val" style="color: #475569;">{len(additional_skills)}</div>
                    <div class="stat-lbl">Additional Skills</div>
                </div>
                """,
                unsafe_allow_html=True
            )

        st.markdown("<br>", unsafe_allow_html=True)

        # Candidate Profile & Gauge Column
        dash_c1, dash_c2 = st.columns([1.2, 1], gap="medium")

        with dash_c1:
            st.markdown(
                f"""
                <div class="glass-panel" style="height: 100%;">
                    <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 1.1rem;">
                        <div style="background: rgba(239, 246, 255, 0.9); color: #2563EB; width: 40px; height: 40px; border-radius: 12px; display: flex; align-items: center; justify-content: center; font-size: 1.2rem; border: 1px solid #BFDBFE;">
                            👤
                        </div>
                        <div>
                            <h3 style="margin: 0; font-size: 1.2rem; font-weight: 800; color: #0F172A;">Candidate Profile</h3>
                            <span style="font-size: 0.8rem; color: #64748B;">Identified via spaCy NER</span>
                        </div>
                    </div>
                    
                    <div style="background: rgba(255, 255, 255, 0.6); border-radius: 12px; padding: 1.1rem; border: 1px solid rgba(226, 232, 240, 0.9);">
                        <div style="margin-bottom: 0.8rem;">
                            <span style="font-size: 0.74rem; text-transform: uppercase; font-weight: 700; color: #64748B;">Candidate Name</span>
                            <div style="font-size: 1.25rem; font-weight: 800; color: #0F172A;">{candidate_name}</div>
                        </div>
                        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 12px;">
                            <div>
                                <span style="font-size: 0.74rem; text-transform: uppercase; font-weight: 700; color: #64748B;">Email Address</span>
                                <div style="font-weight: 600; color: #1E293B; word-break: break-all; font-size: 0.92rem;">{contact_info['email']}</div>
                            </div>
                            <div>
                                <span style="font-size: 0.74rem; text-transform: uppercase; font-weight: 700; color: #64748B;">Phone Number</span>
                                <div style="font-weight: 600; color: #1E293B; font-size: 0.92rem;">{contact_info['phone']}</div>
                            </div>
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        with dash_c2:
            st.markdown(
                f"""
                <div class="glass-panel" style="height: 100%; text-align: center;">
                    <div style="display: flex; align-items: center; justify-content: center; gap: 8px; margin-bottom: 0.4rem;">
                        <span style="font-size: 1.15rem;">🎯</span>
                        <h3 style="margin: 0; font-size: 1.2rem; font-weight: 800; color: #0F172A;">Compatibility Meter</h3>
                    </div>
                    {render_radial_gauge(semantic_score)}
                    <div style="color: #64748B; font-size: 0.8rem; margin-top: 4px;">
                        TF-IDF & Cosine Similarity analysis
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        # ----------------------------------------------------------------------
        # Skills Gap Analysis (Glassmorphic Columns)
        # ----------------------------------------------------------------------
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("### 🔍 Skill Gap Analysis")

        # Interactive filter
        filter_col, _ = st.columns([2, 3])
        with filter_col:
            skill_query = st.text_input("🔍 Filter Analyzed Skills", placeholder="Type a skill name to filter...")

        filtered_matched = [s for s in matched_skills if skill_query.lower() in s.lower()] if skill_query else matched_skills
        filtered_missing = [s for s in missing_skills if skill_query.lower() in s.lower()] if skill_query else missing_skills

        sc1, sc2 = st.columns(2, gap="medium")

        with sc1:
            st.markdown(
                f"""
                <div class="glass-panel" style="border-top: 3px solid #16A34A;">
                    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom: 0.9rem;">
                        <h4 style="margin: 0; color: #166534; font-weight: 800; font-size: 1.1rem;">
                            ✅ Matched Skills ({len(filtered_matched)})
                        </h4>
                        <span style="background: rgba(240, 253, 244, 0.9); color: #15803D; font-weight: 700; font-size: 0.78rem; padding: 2px 10px; border-radius: 9999px; border: 1px solid #BBF7D0;">
                            Present in Resume
                        </span>
                    </div>
                    <div>{render_skill_chips(filtered_matched, chip_type='matched')}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

        with sc2:
            st.markdown(
                f"""
                <div class="glass-panel" style="border-top: 3px solid #DC2626;">
                    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom: 0.9rem;">
                        <h4 style="margin: 0; color: #991B1B; font-weight: 800; font-size: 1.1rem;">
                            ❌ Missing Skills ({len(filtered_missing)})
                        </h4>
                        <span style="background: rgba(254, 242, 242, 0.9); color: #B91C1C; font-weight: 700; font-size: 0.78rem; padding: 2px 10px; border-radius: 9999px; border: 1px solid #FECACA;">
                            Required by JD
                        </span>
                    </div>
                    <div>{render_skill_chips(filtered_missing, chip_type='missing')}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

        if additional_skills:
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(
                f"""
                <div class="glass-panel" style="border-top: 3px solid #64748B;">
                    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom: 0.8rem;">
                        <h4 style="margin: 0; color: #1E293B; font-weight: 800; font-size: 1.05rem;">
                            📋 Additional Candidate Competencies ({len(additional_skills)})
                        </h4>
                        <span style="background: rgba(248, 250, 252, 0.9); color: #475569; font-weight: 700; font-size: 0.78rem; padding: 2px 10px; border-radius: 9999px; border: 1px solid #CBD5E1;">
                            Bonus Skills
                        </span>
                    </div>
                    <div>{render_skill_chips(additional_skills, chip_type='bonus')}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

        # ----------------------------------------------------------------------
        # Detailed Insights & Export Center
        # ----------------------------------------------------------------------
        st.markdown("<br>", unsafe_allow_html=True)
        tab_table, tab_text, tab_export = st.tabs([
            "📊 Tabular Breakdown",
            "📜 Raw Resume Document",
            "💾 Export Reports"
        ])

        with tab_table:
            table_rows = []
            for s in matched_skills:
                table_rows.append({"Skill Name": s, "Match Category": "Matched", "In Resume": "Yes", "Required in JD": "Yes"})
            for s in missing_skills:
                table_rows.append({"Skill Name": s, "Match Category": "Missing", "In Resume": "No", "Required in JD": "Yes"})
            for s in additional_skills:
                table_rows.append({"Skill Name": s, "Match Category": "Bonus Candidate Skill", "In Resume": "Yes", "Required in JD": "No"})

            if table_rows:
                df = pd.DataFrame(table_rows)
                st.dataframe(df, use_container_width=True, hide_index=True)

        with tab_text:
            st.text_area(
                "Cleaned Extracted Resume Text",
                value=resume_text,
                height=280,
                disabled=True
            )

        with tab_export:
            st.markdown("#### Download Candidate Assessment")
            exp_col1, exp_col2 = st.columns(2)

            report_txt = generate_text_report(
                name=candidate_name,
                email=contact_info["email"],
                phone=contact_info["phone"],
                score=semantic_score,
                matched_skills=matched_skills,
                missing_skills=missing_skills,
                resume_skills=resume_skills
            )
            safe_name = candidate_name.replace(" ", "_") if candidate_name != "Not found" else "candidate"

            with exp_col1:
                st.download_button(
                    label="📄 Download Plain Text Summary (.txt)",
                    data=report_txt,
                    file_name=f"{safe_name}_screening_report.txt",
                    mime="text/plain",
                    use_container_width=True
                )

            with exp_col2:
                json_data = json.dumps({
                    "candidate_name": candidate_name,
                    "email": contact_info["email"],
                    "phone": contact_info["phone"],
                    "compatibility_score": semantic_score,
                    "matched_skills": matched_skills,
                    "missing_skills": missing_skills,
                    "additional_skills": additional_skills,
                    "all_candidate_skills": resume_skills
                }, indent=2)

                st.download_button(
                    label="📦 Download JSON Analysis (.json)",
                    data=json_data,
                    file_name=f"{safe_name}_data.json",
                    mime="application/json",
                    use_container_width=True
                )


if __name__ == "__main__":
    main()
