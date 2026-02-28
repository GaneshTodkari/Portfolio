import os
import re
from datetime import datetime
from pathlib import Path

from app.services.skill_phrase_filter import SkillPhraseFilter


def _load_nlp():
    try:
        import spacy

        return spacy.load("en_core_web_sm")
    except Exception:
        try:
            from spacy.cli import download

            download("en_core_web_sm")
            return spacy.load("en_core_web_sm")
        except Exception:
            return None


nlp = _load_nlp()


class ResumeParser:
    EXPERIENCE_HINTS = {
        "experience", "employment", "work", "worked", "intern", "internship",
        "position", "role", "company", "organization", "engineer", "developer",
        "analyst", "manager", "consultant", "executive", "associate", "trainee",
    }

    EDUCATION_HINTS = {
        "bachelor", "master", "m.tech", "b.tech", "phd", "university", "college",
        "school", "cgpa", "gpa", "semester", "academic", "education",
    }

    def __init__(self):
        if nlp is None:
            raise RuntimeError(
                "spaCy model `en_core_web_sm` is not available. "
                "Install with: python -m spacy download en_core_web_sm"
            )
        self.skill_filter = SkillPhraseFilter(nlp)

    def extract_text(self, file_path: str) -> str:
        ext = os.path.splitext(file_path)[1].lower()
        if ext == ".pdf":
            return self._extract_pdf(file_path)
        if ext == ".docx":
            return self._extract_docx(file_path)
        if ext == ".txt":
            return Path(file_path).read_text(encoding="utf-8", errors="ignore")
        raise ValueError("Unsupported file format")

    def _extract_pdf(self, file_path):
        # Priority: PyMuPDF -> pypdf -> PyPDF2
        try:
            import fitz

            doc = fitz.open(file_path)
            text = ""
            for page in doc:
                text += page.get_text("text")
            if text.strip():
                return text
        except Exception:
            pass

        try:
            from pypdf import PdfReader

            reader = PdfReader(file_path)
            text = "\n".join((page.extract_text() or "") for page in reader.pages)
            if text.strip():
                return text
        except Exception:
            pass

        try:
            from PyPDF2 import PdfReader

            reader = PdfReader(file_path)
            text = "\n".join((page.extract_text() or "") for page in reader.pages)
            if text.strip():
                return text
        except Exception:
            pass

        return ""

    def _extract_docx(self, file_path):
        try:
            import docx2txt

            return docx2txt.process(file_path)
        except Exception:
            return ""

    def parse_date(self, text):
        text = text.lower().strip()
        if text in ["present", "current"]:
            return datetime.now()
        for fmt in ("%b %Y", "%B %Y", "%Y"):
            try:
                return datetime.strptime(text, fmt)
            except Exception:
                continue
        return None

    def calculate_experience(self, date_entities):
        parsed_dates = []
        for d in date_entities:
            parsed = self.parse_date(d)
            if parsed:
                parsed_dates.append(parsed)
        parsed_dates = sorted(parsed_dates)
        if len(parsed_dates) < 2:
            return 0.0

        total_months = 0
        for i in range(0, len(parsed_dates) - 1, 2):
            start = parsed_dates[i]
            end = parsed_dates[i + 1]
            months = (end.year - start.year) * 12 + (end.month - start.month)
            if months > 0:
                total_months += months

        years = round(total_months / 12, 2)
        return min(years, 40.0)

    def extract_experience_dates(self, doc):
        exp_dates = []
        for ent in doc.ents:
            if ent.label_ != "DATE":
                continue
            sent_text = ent.sent.text.lower()
            has_exp_hint = any(h in sent_text for h in self.EXPERIENCE_HINTS)
            has_edu_hint = any(h in sent_text for h in self.EDUCATION_HINTS)
            if has_exp_hint and not has_edu_hint:
                exp_dates.append(ent.text)
        return exp_dates

    def parse(self, file_path: str):
        raw_text = self.extract_text(file_path)
        clean_text = raw_text.replace("\n", " ").strip()
        doc = nlp(clean_text)

        companies = []
        education = []
        certifications = []
        date_entities = []

        for ent in doc.ents:
            if ent.label_ == "ORG":
                companies.append(ent.text)
            if ent.label_ == "DATE":
                date_entities.append(ent.text)

        degree_keywords = ["bachelor", "master", "mba", "b.tech", "m.tech", "phd", "b.sc", "m.sc"]
        for token in doc:
            if token.text.lower() in degree_keywords:
                education.append(token.text)

        cert_keywords = ["certification", "certified", "associate", "professional"]
        for token in doc:
            if token.text.lower() in cert_keywords:
                certifications.append(token.text)

        candidate_phrases = self.skill_filter.extract_phrases(doc)
        experience_date_entities = self.extract_experience_dates(doc)
        experience_years = self.calculate_experience(experience_date_entities)

        return {
            "raw_text": raw_text,
            "clean_text": clean_text,
            "companies": list(set(companies)),
            "education": list(set(education)),
            "certifications": list(set(certifications)),
            "candidate_phrases": candidate_phrases,
            "dates": list(set(date_entities)),
            "experience_dates": list(set(experience_date_entities)),
            "experience_years": experience_years,
        }

    def _estimate_experience_fallback(self, text: str) -> float:
        current_year = datetime.now().year
        ranges = re.findall(r"(20\d{2})\s*[-–to]+\s*(20\d{2}|present|current)", text.lower())
        years = []
        for start, end in ranges:
            try:
                s = int(start)
                e = current_year if end in {"present", "current"} else int(end)
                if e >= s:
                    years.append(e - s)
            except Exception:
                continue
        if years:
            return min(max(years), 40.0)
        return 0.0
