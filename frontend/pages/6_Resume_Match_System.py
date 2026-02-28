import importlib
import os
import re
import sys
import tempfile
from pathlib import Path

import streamlit as st

st.set_page_config(page_title="Resume Match System", page_icon="📄", layout="wide")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_PATH = str(PROJECT_ROOT / "backend" / "models" / "portfolio_resume_match_model.joblib")
BACKEND_ROOT = str(PROJECT_ROOT / "backend")
if BACKEND_ROOT not in sys.path:
    sys.path.insert(0, BACKEND_ROOT)


def load_score_resume():
    module_candidates = [
        "app.services.portfolio_model",
        "resume_match_system.app.portfolio_model",
        "resume_match_system.portfolio_model",
        "portfolio_model",
    ]
    errors = []
    for module_name in module_candidates:
        try:
            module = importlib.import_module(module_name)
            if hasattr(module, "score_resume"):
                return module.score_resume, module_name, None
            errors.append(f"{module_name}: score_resume not found")
        except Exception as exc:
            errors.append(f"{module_name}: {exc}")
    return None, None, errors


def get_value(result, key, default=None):
    if hasattr(result, key):
        return getattr(result, key)
    if isinstance(result, dict):
        return result.get(key, default)
    return default


def _extract_text_from_path(file_path: str) -> str:
    suffix = Path(file_path).suffix.lower()
    if suffix == ".txt":
        return Path(file_path).read_text(encoding="utf-8", errors="ignore")
    if suffix == ".pdf":
        try:
            from pypdf import PdfReader

            reader = PdfReader(file_path)
            return "\n".join((p.extract_text() or "") for p in reader.pages)
        except Exception:
            return ""
    if suffix == ".docx":
        try:
            import docx2txt

            return docx2txt.process(file_path) or ""
        except Exception:
            return ""
    return ""


def _tokens(text: str):
    toks = re.findall(r"\b[a-z][a-z0-9+#.\-]{1,}\b", (text or "").lower())
    stop = {
        "the", "and", "for", "with", "that", "this", "from", "are", "you", "your", "our",
        "was", "were", "will", "have", "has", "into", "of", "in", "to", "a", "an", "or",
    }
    return [t for t in toks if t not in stop]


def fallback_score_resume(resume_path: str, jd_text: str, model_path: str = ""):
    resume_text = _extract_text_from_path(resume_path)
    jd_tokens = set(_tokens(jd_text))
    resume_tokens = set(_tokens(resume_text))

    jd_skill_seed = [
        "python", "sql", "excel", "power", "bi", "tableau", "analytics",
        "statistics", "etl", "pandas", "numpy", "communication",
    ]
    jd_skills = sorted({s for s in jd_skill_seed if s in jd_tokens})
    resume_skills = sorted({s for s in jd_skill_seed if s in resume_tokens})

    matching = sorted(set(jd_skills).intersection(resume_skills))
    missing = sorted(set(jd_skills).difference(resume_skills))

    skill_score = (len(matching) / len(jd_skills) * 100.0) if jd_skills else 0.0
    semantic_score = (len(jd_tokens & resume_tokens) / len(jd_tokens) * 100.0) if jd_tokens else 0.0
    years = re.findall(r"(\d+(?:\.\d+)?)\s*\+?\s*(?:yr|yrs|year|years)\b", jd_text.lower())
    required_exp = max([float(y) for y in years], default=0.0)
    experience_score = 100.0 if required_exp <= 0 else 0.0
    final_score = round(0.55 * semantic_score + 0.30 * skill_score + 0.15 * experience_score, 2)

    return {
        "final_score": round(final_score, 2),
        "semantic_score": round(semantic_score, 2),
        "skill_score": round(skill_score, 2),
        "experience_score": round(experience_score, 2),
        "matching_skills": matching,
        "missing_skills": missing,
        "resume_skills": resume_skills,
        "jd_skills": jd_skills,
    }


st.markdown(
    """
    <style>
    [data-testid="stMainBlockContainer"] { max-width: 1140px; padding-top: 1.1rem; }
    .page-kicker { font-size: 0.84rem; text-transform: uppercase; letter-spacing: 0.08em; color: #1f77b4; font-weight: 700; }
    .panel {
        background: var(--secondary-background-color);
        border: 1px solid rgba(49, 51, 63, 0.16);
        border-radius: 12px;
        box-shadow: 0 6px 16px rgba(15, 23, 42, 0.08);
        padding: 1rem 1.1rem;
        margin-bottom: 0.8rem;
    }
    .stButton > button { width: 100%; border-radius: 8px; font-weight: 600; }
    .skill-pill {
        display: inline-block;
        margin: 0.2rem 0.25rem 0 0;
        padding: 0.28rem 0.72rem;
        border-radius: 999px;
        font-size: 0.8rem;
        font-weight: 600;
        border: 1px solid rgba(49, 51, 63, 0.35);
        color: var(--text-color);
        background: rgba(43, 143, 216, 0.16);
    }
    .missing-pill {
        border-color: rgba(49, 51, 63, 0.35);
        color: var(--text-color);
        background: rgba(217, 74, 74, 0.16);
    }
    .skills-block {
        border: 1px solid rgba(49, 51, 63, 0.24);
        border-radius: 10px;
        padding: 0.8rem 0.9rem;
        min-height: 150px;
        background: rgba(127, 127, 127, 0.08);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown("<div class='page-kicker'>NLP Case Study</div>", unsafe_allow_html=True)
st.title("Resume Match System")
st.markdown(
    """
    <div class="panel">
      <h4 style="margin:0 0 0.35rem 0;">Reusable Model Contract</h4>
      <p><code>score_resume(resume_path, jd_text, model_path=\"backend/models/portfolio_resume_match_model.joblib\")</code></p>
      <p>Inputs: resume file path (.pdf/.docx), full JD text, optional model path.</p>
      <p>Outputs: final_score, semantic_score, skill_score, experience_score, matching_skills, missing_skills, resume_skills, jd_skills.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.expander("Pipeline Flow"):
    st.markdown(
        """
1. Parse resume from `resume_path` and extract text from PDF/DOCX.
2. Parse JD from `jd_text`, clean text, and extract skill-like phrases.
3. Build skill units via semantic clustering and canonical skill names.
4. Match JD vs resume skills using lexical overlap + embedding fallback.
5. Compute document semantic similarity (resume vs JD embeddings).
6. Compute experience score from inferred resume years vs JD required years.
7. Produce weighted `final_score` using semantic, skill, and experience components.
8. Return reusable output object for UI and downstream automation.
        """
    )

score_resume, module_name, import_errors = load_score_resume()
if score_resume is None:
    score_resume = fallback_score_resume
    st.warning("Backend model service is unavailable in this environment. Running fallback scorer.")
    with st.expander("Technical details"):
        st.code("\n".join(import_errors or []))

left_col, right_col = st.columns([1, 1], gap="large")
with left_col:
    resume_file = st.file_uploader("Upload Resume (.pdf or .docx)", type=["pdf", "docx"])
    resume_text_manual = st.text_area(
        "Or paste resume text (fallback)",
        height=140,
        placeholder="If PDF/DOCX extraction fails, paste resume text here.",
    )
with right_col:
    jd_text = st.text_area("Job Description", height=280, placeholder="Paste full job description text...")

model_path = DEFAULT_MODEL_PATH

if st.button("Run Match", type="primary"):
    if score_resume is None:
        st.stop()
    if resume_file is None and not resume_text_manual.strip():
        st.warning("Please upload a resume file (.pdf/.docx).")
        st.stop()
    if not jd_text.strip():
        st.warning("Please provide job description text.")
        st.stop()

    temp_path = None
    try:
        if resume_text_manual.strip():
            with tempfile.NamedTemporaryFile(delete=False, suffix=".txt", mode="w", encoding="utf-8") as temp_file:
                temp_file.write(resume_text_manual)
                temp_path = temp_file.name
        else:
            suffix = Path(resume_file.name).suffix.lower()
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
                temp_file.write(resume_file.getbuffer())
                temp_path = temp_file.name

        with st.spinner("Scoring resume against JD..."):
            result = score_resume(temp_path, jd_text, model_path=model_path)

        final_score = get_value(result, "final_score", 0)
        semantic_score = get_value(result, "semantic_score", 0)
        skill_score = get_value(result, "skill_score", 0)
        experience_score = get_value(result, "experience_score", 0)
        matching_skills = get_value(result, "matching_skills", []) or []
        missing_skills = get_value(result, "missing_skills", []) or []
        resume_skills = get_value(result, "resume_skills", []) or []
        jd_skills = get_value(result, "jd_skills", []) or []

        st.success("Resume matching completed.")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Final Score", str(final_score))
        m2.metric("Semantic", str(semantic_score))
        m3.metric("Skill", str(skill_score))
        m4.metric("Experience", str(experience_score))

        try:
            numeric_score = float(final_score)
        except Exception:
            numeric_score = 0.0
        st.progress(min(max(numeric_score / 100.0, 0.0), 1.0))

        c1, c2 = st.columns(2, gap="large")
        with c1:
            st.markdown("### Matching Skills")
            if matching_skills:
                st.markdown(
                    f"""
                    <div class="skills-block">
                        {"".join([f"<span class='skill-pill'>{s}</span>" for s in matching_skills])}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.info("No matching skills returned by model.")
        with c2:
            st.markdown("### Missing Skills")
            if missing_skills:
                st.markdown(
                    f"""
                    <div class="skills-block">
                        {"".join([f"<span class='skill-pill missing-pill'>{s}</span>" for s in missing_skills])}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.info("No missing skills returned by model.")

        with st.expander("Parsed Skill Sets"):
            st.write("Resume Skills:", resume_skills)
            st.write("JD Skills:", jd_skills)
        if not resume_skills:
            st.warning(
                "Resume skills are empty. Your PDF may be image-based or not extracted correctly. "
                "Use the 'Or paste resume text (fallback)' box and run again."
            )

    except Exception as exc:
        err = str(exc)
        if "en_core_web_sm" in err or "spaCy" in err or "spacy" in err:
            st.error(
                "NLP dependencies are missing. Install backend requirements and the spaCy model:\n\n"
                "1) pip install -r backend/requirements.txt\n"
                "2) python -m spacy download en_core_web_sm"
            )
        else:
            st.error(f"Model execution failed: {exc}")
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)

with st.sidebar:
    st.header("NLP Model")
    st.info(
        "Pipeline: Skill extraction + semantic similarity + experience scoring"
    )
