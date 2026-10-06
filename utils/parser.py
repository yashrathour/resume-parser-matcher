"""
Parser module for extracting text and entities from resumes and job descriptions.
Handles PDF and DOCX extraction, regex-based contact info extraction,
spaCy-based candidate name extraction, and keyword-based skill extraction.
"""

import io
import re
from typing import Dict, List, Optional, Union
import docx
from pdfminer.high_level import extract_text as extract_pdf_text
import spacy

# Default comprehensive technical skills list
DEFAULT_SKILLS: List[str] = [
    # Programming Languages
    "Python", "Java", "C", "C++", "C#", "JavaScript", "TypeScript", "Go", "Golang",
    "Rust", "Ruby", "PHP", "Swift", "Kotlin", "Scala", "R", "Dart", "Bash", "Shell",
    "SQL", "HTML", "CSS", "SAS", "MATLAB", "Perl",

    # Web & Backend Frameworks
    "React", "React.js", "Angular", "Vue", "Vue.js", "Next.js", "Node.js", "Express",
    "Express.js", "Django", "Flask", "FastAPI", "Spring", "Spring Boot", "ASP.NET",
    ".NET", "Ruby on Rails", "Laravel", "Redux", "GraphQL", "REST API", "Tailwind CSS",
    "Bootstrap",

    # AI, ML & Data Science
    "Machine Learning", "Deep Learning", "Natural Language Processing", "NLP",
    "Computer Vision", "Data Science", "Data Analysis", "Artificial Intelligence",
    "TensorFlow", "PyTorch", "Keras", "Scikit-Learn", "Pandas", "NumPy", "SciPy",
    "NLTK", "spaCy", "OpenCV", "Hugging Face", "LLMs", "LangChain", "Transformers",
    "Big Data", "Hadoop", "Spark", "Apache Spark",

    # Cloud & DevOps
    "AWS", "Amazon Web Services", "Azure", "Microsoft Azure", "GCP", "Google Cloud",
    "Docker", "Kubernetes", "CI/CD", "Jenkins", "GitLab CI", "GitHub Actions",
    "Terraform", "Ansible", "Linux", "Unix", "Nginx", "Apache",

    # Databases
    "PostgreSQL", "MySQL", "MongoDB", "Redis", "SQLite", "Cassandra", "Oracle",
    "Microsoft SQL Server", "DynamoDB", "Elasticsearch", "Neo4j",

    # Tools & Methodologies
    "Git", "GitHub", "GitLab", "Bitbucket", "Jira", "Confluence", "Agile", "Scrum",
    "Microservices", "Unit Testing", "Test Driven Development", "TDD"
]

# Cached spaCy NLP pipeline instance
_nlp_model = None


def get_spacy_model(model_name: str = "en_core_web_sm"):
    """
    Loads and caches the spaCy language model.
    Attempts automatic loading and falls back to a blank pipeline if unavailable,
    ensuring zero runtime crash on cloud hosting.
    """
    global _nlp_model
    if _nlp_model is not None:
        return _nlp_model

    try:
        _nlp_model = spacy.load(model_name)
        return _nlp_model
    except Exception:
        pass

    try:
        import subprocess
        import sys
        subprocess.run(
            [sys.executable, "-m", "spacy", "download", model_name],
            check=False,
            capture_output=True
        )
        _nlp_model = spacy.load(model_name)
        return _nlp_model
    except Exception as e:
        print(f"Warning: Could not load or download '{model_name}': {e}. Using blank English pipeline.")

    _nlp_model = spacy.blank("en")
    if "sentencizer" not in _nlp_model.pipe_names:
        _nlp_model.add_pipe("sentencizer")

    return _nlp_model


def clean_extracted_text(text: str) -> str:
    """
    Cleans raw text by normalizing whitespaces, carriage returns, and blank lines.
    """
    if not text:
        return ""
    # Normalize Windows and Mac carriage returns
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Replace non-breaking spaces with standard space
    text = text.replace("\xa0", " ")
    # Replace tabs with spaces
    text = text.replace("\t", " ")
    # Collapse multiple consecutive horizontal spaces
    text = re.sub(r"[ ]{2,}", " ", text)
    # Collapse more than two consecutive newlines
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_text_from_pdf(pdf_source: Union[io.BytesIO, bytes, str, object]) -> str:
    """
    Extracts text from a PDF file using pdfminer.six.
    Supports Streamlit UploadedFile, BytesIO, raw bytes, or file path.

    Args:
        pdf_source: The PDF source object.

    Returns:
        Cleaned extracted text as a string.

    Raises:
        ValueError: If the file is corrupted, unreadable, or empty.
    """
    try:
        # If it's a Streamlit UploadedFile or file-like object
        if hasattr(pdf_source, "getvalue"):
            file_bytes = pdf_source.getvalue()
            stream = io.BytesIO(file_bytes)
            raw_text = extract_pdf_text(stream)
        elif hasattr(pdf_source, "read") and callable(pdf_source.read):
            if hasattr(pdf_source, "seek"):
                pdf_source.seek(0)
            stream = io.BytesIO(pdf_source.read())
            raw_text = extract_pdf_text(stream)
        elif isinstance(pdf_source, (bytes, bytearray)):
            stream = io.BytesIO(pdf_source)
            raw_text = extract_pdf_text(stream)
        elif isinstance(pdf_source, str):
            raw_text = extract_pdf_text(pdf_source)
        else:
            raise ValueError("Unsupported PDF input type.")

        cleaned = clean_extracted_text(raw_text)
        if not cleaned:
            raise ValueError("Extracted PDF content is empty or contains non-extractable scanned images.")
        return cleaned

    except Exception as exc:
        raise ValueError(f"Failed to read or parse PDF file: {exc}") from exc


def extract_text_from_docx(docx_source: Union[io.BytesIO, bytes, str, object]) -> str:
    """
    Extracts text from a DOCX file using python-docx.
    Supports Streamlit UploadedFile, BytesIO, raw bytes, or file path.

    Args:
        docx_source: The DOCX source object.

    Returns:
        Cleaned extracted text as a string.

    Raises:
        ValueError: If the file is corrupted, unreadable, or empty.
    """
    try:
        # Wrap into stream if UploadedFile or bytes
        if hasattr(docx_source, "getvalue"):
            stream = io.BytesIO(docx_source.getvalue())
            doc = docx.Document(stream)
        elif hasattr(docx_source, "read") and callable(docx_source.read):
            if hasattr(docx_source, "seek"):
                docx_source.seek(0)
            stream = io.BytesIO(docx_source.read())
            doc = docx.Document(stream)
        elif isinstance(docx_source, (bytes, bytearray)):
            stream = io.BytesIO(docx_source)
            doc = docx.Document(stream)
        elif isinstance(docx_source, str):
            doc = docx.Document(docx_source)
        else:
            raise ValueError("Unsupported DOCX input type.")

        text_parts: List[str] = []

        # Extract text from standard paragraphs
        for para in doc.paragraphs:
            stripped = para.text.strip()
            if stripped:
                text_parts.append(stripped)

        # Extract text from tables (many resumes format contact/skills in tables)
        for table in doc.tables:
            for row in table.rows:
                row_cells = [cell.text.strip() for cell in row.cells if cell.text.strip()]
                # Deduplicate identical adjacent cells (due to merged cells)
                unique_row = []
                for cell in row_cells:
                    if not unique_row or cell != unique_row[-1]:
                        unique_row.append(cell)
                if unique_row:
                    text_parts.append(" | ".join(unique_row))

        raw_text = "\n".join(text_parts)
        cleaned = clean_extracted_text(raw_text)
        if not cleaned:
            raise ValueError("Extracted DOCX content is empty.")
        return cleaned

    except Exception as exc:
        raise ValueError(f"Failed to read or parse DOCX file: {exc}") from exc


def extract_text_from_file(uploaded_file) -> str:
    """
    Convenience dispatcher function to extract text from a Streamlit UploadedFile
    based on its file extension.
    """
    if uploaded_file is None:
        raise ValueError("No file provided.")

    filename = getattr(uploaded_file, "name", "").lower()
    if filename.endswith(".pdf"):
        return extract_text_from_pdf(uploaded_file)
    elif filename.endswith(".docx"):
        return extract_text_from_docx(uploaded_file)
    else:
        raise ValueError(f"Unsupported file format for '{getattr(uploaded_file, 'name', 'unknown')}'. Please upload a PDF or DOCX file.")


def extract_contact_info(text: str) -> Dict[str, Optional[str]]:
    """
    Reliably extracts email addresses and phone numbers from resume text using regular expressions.

    Returns:
        Dictionary containing:
            - 'email': primary email string or 'Not found'
            - 'phone': primary phone string or 'Not found'
            - 'all_emails': list of all matched emails
            - 'all_phones': list of all matched phone numbers
    """
    if not text:
        return {
            "email": "Not found",
            "phone": "Not found",
            "all_emails": [],
            "all_phones": []
        }

    # Robust regex for email addresses
    email_regex = re.compile(
        r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+",
        re.IGNORECASE
    )
    emails = email_regex.findall(text)
    # Deduplicate while preserving order and strip punctuation
    unique_emails = []
    for em in emails:
        em_clean = em.strip(".,;:()<>[]\"'")
        if em_clean and em_clean not in unique_emails:
            unique_emails.append(em_clean)

    # Regex for international and domestic phone numbers:
    # Matches formats like +1-555-555-5555, (555) 555-5555, +91 9876543210, 555.555.5555, etc.
    phone_regex = re.compile(
        r"(?:(?:\+?\d{1,3}[\s.-]?)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}|\+?\d{1,4}[\s.-]?\d{10})",
        re.IGNORECASE
    )
    raw_phones = phone_regex.findall(text)
    unique_phones = []
    for ph in raw_phones:
        ph_clean = ph.strip(".,;:()<>[]\"' ")
        # Filter out numbers that are too short (like dates or zip codes)
        digits = re.sub(r"\D", "", ph_clean)
        if 10 <= len(digits) <= 15:
            if ph_clean not in unique_phones:
                unique_phones.append(ph_clean)

    return {
        "email": unique_emails[0] if unique_emails else "Not found",
        "phone": unique_phones[0] if unique_phones else "Not found",
        "all_emails": unique_emails,
        "all_phones": unique_phones
    }


def extract_name(text: str, nlp=None) -> str:
    """
    Extracts the candidate's name using spaCy Named Entity Recognition (NER: 'PERSON').
    Focuses on the resume header where candidate names typically reside.

    Args:
        text: The resume text.
        nlp: Optional preloaded spaCy model.

    Returns:
        The extracted candidate name, or 'Not found' if none detected.
    """
    if not text or not text.strip():
        return "Not found"

    if nlp is None:
        nlp = get_spacy_model()

    # Exclude common non-name headers and job-related keywords
    stopwords_title = {
        "curriculum", "vitae", "resume", "profile", "summary", "contact",
        "experience", "education", "skills", "projects", "certifications",
        "references", "engineer", "developer", "manager", "lead", "analyst",
        "scientist", "intern", "associate", "specialist", "page"
    }

    # In resumes, the candidate's name is almost always at the very top.
    # Take the first 10 non-empty lines or first 600 characters for highest accuracy.
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    header_text = "\n".join(lines[:8])

    doc_header = nlp(header_text)
    for ent in doc_header.ents:
        if ent.label_ == "PERSON":
            name = ent.text.strip()
            # Clean up unwanted punctuation
            name = re.sub(r"[^a-zA-Z\s.-]", "", name).strip()
            name_parts = name.split()
            # Valid candidate names typically consist of 2 to 4 words without digits or resume stopwords
            if 2 <= len(name_parts) <= 4:
                lower_parts = [p.lower() for p in name_parts]
                if not any(stop in lower_parts for stop in stopwords_title):
                    return name.title()

    # Fallback to searching the broader initial text (first 1500 chars) if header check failed
    full_preview = text[:1500]
    doc_full = nlp(full_preview)
    for ent in doc_full.ents:
        if ent.label_ == "PERSON":
            name = ent.text.strip()
            name = re.sub(r"[^a-zA-Z\s.-]", "", name).strip()
            name_parts = name.split()
            if 2 <= len(name_parts) <= 4:
                lower_parts = [p.lower() for p in name_parts]
                if not any(stop in lower_parts for stop in stopwords_title):
                    return name.title()

    # If NER fails, use a clean heuristic on the first line if it looks like a person's name
    if lines:
        first_line = re.sub(r"[^a-zA-Z\s.-]", "", lines[0]).strip()
        first_words = first_line.split()
        if 2 <= len(first_words) <= 3:
            lower_words = [w.lower() for w in first_words]
            if not any(w in stopwords_title for w in lower_words):
                return first_line.title()

    return "Not found"


def extract_skills(text: str, predefined_skills_list: Optional[List[str]] = None) -> List[str]:
    """
    Extracts technical skills by comparing the text against a predefined list case-insensitively.
    Uses regex delimiters to avoid false substring matches (e.g., 'C' inside 'React' or 'Go' in 'Good').

    Args:
        text: The resume or job description text.
        predefined_skills_list: List of technical skill names. Defaults to DEFAULT_SKILLS.

    Returns:
        Sorted list of unique matched skills with proper casing.
    """
    if not text:
        return []

    if predefined_skills_list is None:
        predefined_skills_list = DEFAULT_SKILLS

    matched_skills = set()

    for skill in predefined_skills_list:
        skill_clean = skill.strip()
        if not skill_clean:
            continue

        # Lookbehind and lookahead prevent false substring matches within words or identifiers,
        # while safely supporting symbols like C++, C#, .NET
        escaped_skill = re.escape(skill_clean)
        pattern = rf"(?<![A-Za-z0-9#+]){escaped_skill}(?![A-Za-z0-9#+])"

        if re.search(pattern, text, re.IGNORECASE):
            matched_skills.add(skill_clean)

    return sorted(list(matched_skills), key=lambda s: s.lower())
