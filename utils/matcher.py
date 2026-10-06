"""
Matcher module for calculating semantic similarity and comparing skill sets.
Provides TF-IDF based cosine similarity scoring and skill gap analysis.
"""

from typing import Any, Dict, List, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class SkillMatchResult(dict):
    """
    Dictionary subclass holding 'Matched Skills' and 'Missing Skills'.
    Supports tuple unpacking:
        matched, missing = compare_skills(resume_skills, jd_skills)
    as well as dictionary and attribute access:
        result['Matched Skills'], result.matched_skills
    """
    def __init__(self, matched_skills: List[str], missing_skills: List[str]):
        super().__init__({
            "Matched Skills": matched_skills,
            "Missing Skills": missing_skills
        })
        self.matched_skills = matched_skills
        self.missing_skills = missing_skills

    def __iter__(self):
        yield self["Matched Skills"]
        yield self["Missing Skills"]


def calculate_semantic_score(resume_text: str, jd_text: str) -> float:
    """
    Calculates a semantic match score between resume text and job description text
    using scikit-learn's TfidfVectorizer and cosine_similarity.

    Args:
        resume_text: Raw or cleaned text from the candidate's resume.
        jd_text: Raw or cleaned text from the target job description.

    Returns:
        A rounded float between 0.0 and 100.0 representing the percentage match.
    """
    if not resume_text or not jd_text:
        return 0.0

    resume_clean = resume_text.strip()
    jd_clean = jd_text.strip()

    if not resume_clean or not jd_clean:
        return 0.0

    try:
        # Standard TF-IDF vectorizer with English stop words
        vectorizer = TfidfVectorizer(
            stop_words="english"
        )
        tfidf_matrix = vectorizer.fit_transform([resume_clean, jd_clean])

        # Compute cosine similarity between the two document vectors
        similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]

        # Convert to percentage and round to 2 decimal places
        score = round(float(similarity) * 100.0, 2)
        return max(0.0, min(100.0, score))

    except ValueError:
        # Handles edge cases such as empty vocabulary or documents with only stopwords
        return 0.0
    except Exception as e:
        print(f"Error computing semantic score: {e}")
        return 0.0


def compare_skills(resume_skills: List[str], jd_skills: List[str]) -> SkillMatchResult:
    """
    Compares skills extracted from a resume against skills extracted from a job description.

    Args:
        resume_skills: List of skills found in the resume.
        jd_skills: List of skills found in or required by the job description.

    Returns:
        SkillMatchResult with:
            - 'Matched Skills': skills present in both the JD and the resume.
            - 'Missing Skills': skills required by the JD but absent in the resume.
        Supports tuple unpacking: matched, missing = compare_skills(...)
    """
    if not jd_skills:
        return SkillMatchResult(matched_skills=[], missing_skills=[])

    # Normalize resume skills for case-insensitive matching
    resume_skill_lookup = {skill.strip().lower(): skill.strip() for skill in (resume_skills or []) if skill.strip()}

    matched: List[str] = []
    missing: List[str] = []

    # Check each JD skill
    seen = set()
    for jd_skill in jd_skills:
        jd_skill_clean = jd_skill.strip()
        if not jd_skill_clean:
            continue

        lower_key = jd_skill_clean.lower()
        if lower_key in seen:
            continue
        seen.add(lower_key)

        if lower_key in resume_skill_lookup:
            # Preserve JD canonical casing or resume casing
            matched.append(jd_skill_clean)
        else:
            missing.append(jd_skill_clean)

    # Sort alphabetically for clean presentation
    matched_sorted = sorted(matched, key=lambda s: s.lower())
    missing_sorted = sorted(missing, key=lambda s: s.lower())

    return SkillMatchResult(matched_skills=matched_sorted, missing_skills=missing_sorted)
