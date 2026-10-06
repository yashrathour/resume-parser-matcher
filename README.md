# AI-Powered Resume Parser & Skill Matcher 📄⚡

An intelligent, glassmorphism-styled Streamlit application that evaluates candidate resumes against target job descriptions using **spaCy NER**, **regex entity extraction**, and **TF-IDF semantic cosine similarity**.

---

## 🌟 Key Features

- **Multi-Format Resume Ingestion**: Supports `.pdf` (via `pdfminer.six`) and `.docx` (via `python-docx`) documents.
- **Candidate Entity Extraction**:
  - Automatically identifies candidate names via **spaCy Named Entity Recognition** (`en_core_web_sm`).
  - Reliably extracts email addresses and international/domestic phone numbers via regex.
- **Skill Extraction & Gap Analysis**:
  - Matches resume competencies against a comprehensive technical catalog (70+ skills).
  - Delivers a clear breakdown of **Matched Skills** vs. **Missing Skills**.
- **Vector-Based Compatibility Scoring**: Computes a semantic percentage match score using `scikit-learn`'s `TfidfVectorizer` and `cosine_similarity`.
- **Refined Glassmorphic Light UI**: Designed with clean frosted glass cards, SVG radial gauges, non-neon color accents, and interactive demo profiles.
- **Multi-Format Export**: 1-click downloads for plain text summary reports (`.txt`) and structured data (`.json`).

---

## 🚀 Quickstart & Local Setup

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/resume-parser-matcher.git
cd resume-parser-matcher

# 2. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download spaCy model
python -m spacy download en_core_web_sm

# 5. Launch the Streamlit application
streamlit run app.py
```

---

## ☁️ Deploying to Streamlit Community Cloud

1. Push this repository to your GitHub account.
2. Sign in to [share.streamlit.io](https://share.streamlit.io) with your GitHub account.
3. Click **New app**, select your repository, set the main file path to `app.py`, and click **Deploy**!
*(All dependencies including the spaCy model are pre-configured in `requirements.txt` for automatic installation).*

---

## 📁 Project Architecture

```text
nlp/
├── app.py                      # Main Streamlit dashboard & application entry point
├── requirements.txt            # Python dependencies + spaCy model wheel
├── .streamlit/
│   └── config.toml             # Streamlit light theme and server configuration
├── utils/
│   ├── parser.py               # Text extraction (PDF/DOCX), contact extraction, NER, skill extraction
│   └── matcher.py              # TF-IDF cosine similarity & skill gap analysis
├── tests/
│   └── test_parser_matcher.py  # Unit test verification suite
└── Dockerfile                  # Container deployment configuration
```
