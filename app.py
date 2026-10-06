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
    page_title="Resume Parser & Skill Matcher",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------------------------------------------------------------------
# Refined Glassmorphism Light UI Theme (Zero Neon, Clean & Readable)
# ------------------------------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

    /* Global Soft Light Background */
    html, body, .stApp {
        background-color: #F8FAFC !important;
        background-image: 
            radial-gradient(at 10% 12%, rgba(219, 234, 254, 0.45) 0px, transparent 50%),
            radial-gradient(at 90% 15%, rgba(224, 231, 255, 0.4) 0px, transparent 50%),
            radial-gradient(at 50% 90%, rgba(241, 245, 249, 0.6) 0px, transparent 60%) !important;
        color: #0F172A !important;
        font-family: 'Plus Jakarta Sans', -apple-system, sans-serif !important;
    }

    /* Force Clean Slate Text Everywhere */
    h1, h2, h3, h4, h5, h6, p, span, label, div {
        color: #0F172A;
    }

    /* Glassmorphism for Streamlit Border Containers */
    [data-testid="stVerticalBlockBorderWrapper"] {
        background: rgba(255, 255, 255, 0.75) !important;
        backdrop-filter: blur(16px) !important;
        -webkit-backdrop-filter: blur(16px) !important;
        border: 1px solid rgba(226, 232, 240, 0.9) !important;
        border-radius: 16px !important;
        box-shadow: 0 4px 20px -2px rgba(15, 23, 42, 0.04), inset 0 1px 0 rgba(255, 255, 255, 0.95) !important;
        padding: 1.25rem !important;
        transition: all 0.25s ease !important;
    }
    [data-testid="stVerticalBlockBorderWrapper"]:hover {
        background: rgba(255, 255, 255, 0.9) !important;
        border-color: rgba(203, 213, 225, 0.95) !important;
        box-shadow: 0 10px 28px -4px rgba(15, 23, 42, 0.07), inset 0 1px 0 rgba(255, 255, 255, 1) !important;
        transform: translateY(-2px);
    }

    /* Sidebar Frosted Glass */
    [data-testid="stSidebar"] {
        background: rgba(255, 255, 255, 0.8) !important;
        backdrop-filter: blur(20px) !important;
        -webkit-backdrop-filter: blur(20px) !important;
        border-right: 1px solid rgba(226, 232, 240, 0.85) !important;
    }

    /* File Uploader Glassmorphism */
    [data-testid="stFileUploader"] {
        background: rgba(255, 255, 255, 0.7) !important;
        backdrop-filter: blur(14px) !important;
        border-radius: 14px !important;
        border: 1px solid rgba(226, 232, 240, 0.85) !important;
        padding: 0.5rem !important;
    }
    [data-testid="stFileUploader"] section {
        background: rgba(255, 255, 255, 0.75) !important;
        border: 1.5px dashed #CBD5E1 !important;
        border-radius: 12px !important;
    }
    [data-testid="stFileUploader"] section:hover {
        border-color: #2563EB !important;
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
    }

    /* Text Area */
    [data-testid="stTextArea"] textarea {
        background: rgba(255, 255, 255, 0.8) !important;
        backdrop-filter: blur(14px) !important;
        border: 1px solid rgba(203, 213, 225, 0.9) !important;
        border-radius: 14px !important;
        color: #0F172A !important;
        font-size: 0.95rem !important;
        line-height: 1.55 !important;
        box-shadow: 0 2px 8px rgba(15, 23, 42, 0.02) !important;
    }
    [data-testid="stTextArea"] textarea:focus {
        border-color: #2563EB !important;
        box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.12) !important;
        background: #FFFFFF !important;
    }

    /* Input text fields */
    .stTextInput input {
        background: rgba(255, 255, 255, 0.85) !important;
        border: 1px solid #CBD5E1 !important;
        border-radius: 10px !important;
        color: #0F172A !important;
    }

    /* Secondary / Demo Buttons */
    .stButton button {
        background: rgba(255, 255, 255, 0.85) !important;
        backdrop-filter: blur(12px) !important;
        color: #1E293B !important;
        border: 1px solid #CBD5E1 !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
        padding: 0.55rem 1.2rem !important;
        box-shadow: 0 1px 3px rgba(15, 23, 42, 0.03) !important;
        transition: all 0.2s ease !important;
    }
    .stButton button:hover {
        background: #FFFFFF !important;
        color: #1D4ED8 !important;
        border-color: #93C5FD !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.1) !important;
    }

    /* Primary Analyze Button */
    .stButton button[kind="primary"] {
        background: linear-gradient(135deg, #2563EB 0%, #1D4ED8 100%) !important;
        color: #FFFFFF !important;
        border: none !important;
        font-weight: 700 !important;
        box-shadow: 0 4px 14px rgba(37, 99, 235, 0.28) !important;
    }
    .stButton button[kind="primary"]:hover {
        background: linear-gradient(135deg, #1D4ED8 0%, #1E40AF 100%) !important;
        color: #FFFFFF !important;
        box-shadow: 0 8px 22px rgba(37, 99, 235, 0.38) !important;
        transform: translateY(-2px) !important;
    }

    /* Metrics Styling */
    [data-testid="stMetricValue"] {
        color: #0F172A !important;
        font-weight: 800 !important;
        font-size: 1.9rem !important;
    }
    [data-testid="stMetricLabel"] {
        color: #64748B !important;
        font-weight: 600 !important;
        font-size: 0.85rem !important;
        text-transform: uppercase !important;
        letter-spacing: 0.03em !important;
    }

    /* Hero Tag */
    .hero-tag {
        display: inline-block;
        background: rgba(239, 246, 255, 0.9);
        color: #1D4ED8;
        border: 1px solid #BFDBFE;
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.03em;
        text-transform: uppercase;
        margin-bottom: 0.5rem;
    }

    /* Calm Non-Neon Skill Chips */
    .skill-chip {
        display: inline-block;
        padding: 6px 14px;
        border-radius: 9999px;
        font-size: 0.86rem;
        font-weight: 600;
        margin: 3px;
        transition: all 0.2s ease;
    }
    .skill-chip:hover {
        transform: translateY(-2px);
    }
    .chip-matched {
        background: #F0FDF4;
        color: #166534;
        border: 1px solid #BBF7D0;
    }
    .chip-missing {
        background: #FEF2F2;
        color: #991B1B;
        border: 1px solid #FECACA;
    }
    .chip-bonus {
        background: #F8FAFC;
        color: #334155;
        border: 1px solid #CBD5E1;
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


def render_skill_chips_html(items, chip_type="matched"):
    """Returns a flat, unindented HTML string of calm chips."""
    if not items:
        return '<p style="color: #94A3B8; font-style: italic; margin: 0;">None detected</p>'

    css_class = f"chip-{chip_type}"
    icon = "✓" if chip_type == "matched" else ("✕" if chip_type == "missing" else "•")

    chips = [f'<span class="skill-chip {css_class}">{icon} {item}</span>' for item in items]
    return f'<div style="margin-top: 4px;">{"".join(chips)}</div>'


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
        st.markdown("### ⚙️ Engine Status")
        st.success("NLP Pipeline Active (spaCy & TF-IDF)")

        st.subheader("💡 SpaCy Command")
        st.code("python -m spacy download en_core_web_sm", language="bash")
        st.caption("Required for Named Entity Recognition.")

        st.divider()

        st.subheader("📚 Skill Repository")
        st.write(f"Total standard skills in catalog: **{len(DEFAULT_SKILLS)}**")
        search_filter = st.text_input("🔍 Search Catalog", placeholder="e.g. Python, Docker, React")
        filtered_catalog = [s for s in DEFAULT_SKILLS if search_filter.lower() in s.lower()] if search_filter else DEFAULT_SKILLS

        with st.expander(f"Browse Skills ({len(filtered_catalog)})", expanded=False):
            st.write(", ".join(sorted(filtered_catalog)))

        st.divider()
        st.caption("AI Resume Parser & Skill Matcher")

    # --------------------------------------------------------------------------
    # Hero Header (Clean Container)
    # --------------------------------------------------------------------------
    with st.container(border=True):
        st.markdown('<span class="hero-tag">✨ Talent Intelligence AI</span>', unsafe_allow_html=True)
        st.markdown("## 📄 Resume Parser & Skill Matcher")
        st.markdown(
            "Evaluate candidate resumes against target job specifications with intelligent NLP parsing, "
            "contact information extraction, and semantic skill gap scoring."
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # Demo quick-loader buttons
    demo_cols = st.columns([1.6, 1.2, 1.2], gap="small")
    with demo_cols[0]:
        st.markdown("<p style='font-weight:600; color:#475569; margin-top:8px;'>⚡ Quick Test with Demo Profiles:</p>", unsafe_allow_html=True)
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
        st.markdown("### 1. Upload Resume")
        uploaded_file = st.file_uploader(
            "Upload resume file",
            type=["pdf", "docx"],
            help="Accepts .pdf (via pdfminer.six) or .docx (via python-docx)",
            label_visibility="collapsed"
        )

        if uploaded_file:
            st.success(f"✓ Uploaded: **{uploaded_file.name}** ({uploaded_file.size / 1024:.1f} KB)")
        elif st.session_state.sample_resume_text:
            st.info("ℹ️ Using loaded demo resume text. (Upload a file above to test your own document)")

    with col_jd:
        st.markdown("### 2. Job Description")
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
            st.warning("⚠️ Please provide a resume (upload PDF/DOCX or select a Demo profile) and paste a Job Description.")
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
        # Results Dashboard (100% Clean & Readable, No Raw Code)
        # --------------------------------------------------------------------------
        st.markdown("<br>", unsafe_allow_html=True)
        st.subheader("🎯 Analysis Results")

        # Top Stat Metrics Cards
        stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
        with stat_col1:
            with st.container(border=True):
                st.metric(label="Match Score", value=f"{semantic_score}%")
        with stat_col2:
            with st.container(border=True):
                st.metric(label="Matched Skills", value=len(matched_skills))
        with stat_col3:
            with st.container(border=True):
                st.metric(label="Missing Skills", value=len(missing_skills))
        with stat_col4:
            with st.container(border=True):
                st.metric(label="Bonus Skills", value=len(additional_skills))

        st.markdown("<br>", unsafe_allow_html=True)

        # Profile & Fit Assessment Columns
        dash_c1, dash_c2 = st.columns([1.2, 1], gap="medium")

        with dash_c1:
            with st.container(border=True):
                st.markdown("### 👤 Candidate Profile")
                st.caption("Extracted via spaCy Named Entity Recognition")
                st.divider()

                col_name, col_contact = st.columns([1, 1])
                with col_name:
                    st.markdown("**Candidate Name**")
                    st.markdown(f"#### {candidate_name}")
                with col_contact:
                    st.markdown("**Email Address**")
                    st.write(contact_info['email'])
                    st.markdown("**Phone Number**")
                    st.write(contact_info['phone'])

        with dash_c2:
            with st.container(border=True):
                st.markdown("### 🎯 Compatibility Rating")
                st.caption("TF-IDF Vectorization & Cosine Similarity Match")
                st.divider()

                score_col, badge_col = st.columns([1, 1])
                with score_col:
                    st.metric(label="Semantic Match", value=f"{semantic_score}%")
                with badge_col:
                    st.write("")
                    if semantic_score >= 70:
                        st.success("🟢 Strong Candidate Fit")
                    elif semantic_score >= 40:
                        st.warning("🟡 Moderate Candidate Fit")
                    else:
                        st.error("🔴 Low Candidate Fit")

                st.progress(min(1.0, max(0.0, semantic_score / 100.0)))
                st.caption(f"{len(matched_skills)} of {len(effective_jd_skills)} required skills found in candidate profile.")

        # --------------------------------------------------------------------------
        # Skills Gap Analysis (Readable Chips)
        # --------------------------------------------------------------------------
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
            with st.container(border=True):
                st.markdown(f"#### ✅ Matched Skills ({len(filtered_matched)})")
                st.caption("Skills found in candidate resume")
                st.divider()
                st.markdown(render_skill_chips_html(filtered_matched, chip_type='matched'), unsafe_allow_html=True)

        with sc2:
            with st.container(border=True):
                st.markdown(f"#### ❌ Missing Skills ({len(filtered_missing)})")
                st.caption("Required by Job Description but missing in resume")
                st.divider()
                st.markdown(render_skill_chips_html(filtered_missing, chip_type='missing'), unsafe_allow_html=True)

        if additional_skills:
            st.markdown("<br>", unsafe_allow_html=True)
            with st.container(border=True):
                st.markdown(f"#### 📋 Additional Candidate Competencies ({len(additional_skills)})")
                st.caption("Bonus skills found on candidate resume not explicitly required by JD")
                st.divider()
                st.markdown(render_skill_chips_html(additional_skills, chip_type='bonus'), unsafe_allow_html=True)

        # --------------------------------------------------------------------------
        # Detailed Insights & Export Center
        # --------------------------------------------------------------------------
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
