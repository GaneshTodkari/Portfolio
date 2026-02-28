import re

import spacy

from app.services.skill_phrase_filter import SkillPhraseFilter


def _load_nlp():
    try:
        return spacy.load("en_core_web_sm")
    except Exception:
        return None


nlp = _load_nlp()


class JDParser:
    def __init__(self):
        if nlp is None:
            raise RuntimeError(
                "spaCy model `en_core_web_sm` is not available. "
                "Install with: python -m spacy download en_core_web_sm"
            )
        self.skill_filter = SkillPhraseFilter(nlp)
        self.direct_terms = [
            "python", "java", "javascript", "sql", "nosql", "pyspark", "spark",
            "power bi", "tableau", "excel", "git", "docker", "kubernetes",
            "aws", "azure", "gcp", "api", "etl", "elt", "data pipeline",
            "machine learning", "nlp", "computer vision",
            "master data management", "data migration", "sap mdm",
            "sap master data", "data extraction", "data processing",
            "data analytics", "data management", "process improvement",
            "data reporting", "sap",
            "autocad", "solidworks", "catia", "ansys", "nx cad", "creo",
            "gd&t", "tolerance stack-up", "dfm", "dfa", "fea", "cfd",
            "cnc", "cam", "cad", "sheet metal", "welding", "machining",
            "plm", "pdm", "maintenance", "rca", "lean", "six sigma",
            "revit", "staad", "etabs", "bim", "quantity estimation",
            "quantity surveying", "boq", "bar bending schedule", "primavera",
            "ms project", "site execution", "structural analysis",
            "geotechnical engineering", "surveying", "construction planning",
            "power systems", "substation", "switchgear", "relay protection",
            "load flow", "short circuit analysis", "electrical design",
            "earthing", "lighting design", "hvac", "mep",
            "embedded systems", "microcontroller", "pcb design", "vlsi",
            "verilog", "vhdl", "fpga", "rtos", "signal processing",
            "communication systems", "rf", "antenna", "telecom", "iot",
            "plc", "scada", "dcs", "pid control", "instrument calibration",
            "process instrumentation", "hmi",
            "process engineering", "p&id", "aspen hysys", "distillation",
            "heat exchanger", "mass transfer", "reactor design", "process safety",
            "aerospace", "aeronautical", "avionics", "biomedical",
            "biotechnology", "metallurgy", "mining engineering",
            "environmental engineering", "wastewater treatment", "gis",
        ]

    def calculate_experience(self, doc):
        years = []
        text = doc.text.lower()
        for m in re.finditer(r"(\d+(?:\.\d+)?)\s*\+?\s*(?:yr|yrs|year|years)\b", text):
            try:
                years.append(float(m.group(1)))
            except Exception:
                continue
        return max(years) if years else 0.0

    def extract_direct_skills(self, clean_text: str):
        text = clean_text.lower()
        hits = []
        for term in self.direct_terms:
            if term in text:
                hits.append(term)
        return hits

    def parse(self, jd_text: str):
        clean_text = jd_text.replace("\n", " ").strip()
        doc = nlp(clean_text)
        candidate_phrases = self.skill_filter.extract_phrases(doc)
        candidate_phrases.extend(self.extract_direct_skills(clean_text))
        candidate_phrases = sorted(set(candidate_phrases))
        required_experience = self.calculate_experience(doc)
        return {
            "raw_text": jd_text,
            "clean_text": clean_text,
            "candidate_phrases": candidate_phrases,
            "required_experience": required_experience,
        }
