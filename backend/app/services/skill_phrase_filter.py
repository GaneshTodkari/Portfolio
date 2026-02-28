import re


class SkillPhraseFilter:
    SECTION_NOISE = {
        "we", "we're", "hiring", "job", "title", "role", "about", "overview",
        "responsibilities", "responsibility", "requirements", "qualification",
        "qualifications", "preferred", "must", "minimum", "experience", "year",
        "years", "location", "employment", "type", "full-time", "part-time",
        "hybrid", "onsite", "remote", "candidate", "ideal", "advantage",
        "summary", "profile", "description", "company", "organization", "team",
        "key", "only"
    }

    PHRASE_NOISE_CONTAINS = {
        "job title", "employment type", "key responsibilities", "preferred skills",
        "technical skills", "about the role", "we re hiring", "we're hiring",
        "location", "only employment type", "industry type", "role category",
        "job description", "interview mode", "immediate joiners", "department"
    }

    PHRASE_NOISE_EXACT = {
        "complex business problems", "key responsibilities", "preferred skills",
        "technical skills", "stakeholders", "our analytics team",
        "cross-functional stakeholders", "analysis assist",
        "analysis proficiency", "business decision-making",
        "job description process", "procurement", "department",
        "recruitment staffing department"
    }

    CONTEXT_NOISE_WORDS = {
        "our", "team", "stakeholder", "stakeholders", "assist", "support",
        "collaboration", "loc", "salary", "shift", "regards", "interview",
        "mode", "department", "industry", "category", "joiners"
    }

    ROLE_TAXONOMY_WORDS = {
        "procurement", "recruitment", "staffing", "department", "industry",
        "category", "role", "process"
    }

    ALLOWED_SHORT_SKILLS = {
        "sql", "etl", "elt", "aws", "gcp", "api", "ml", "ai", "nlp", "bi",
        "crm", "erp", "hr", "ui", "ux", "qa", "kpi", "okr", "b2b", "b2c",
        "cad", "cam", "cnc", "plc", "scada", "hvac", "mep", "pmp", "bom",
        "bim", "boq", "rf", "pcb", "vlsi", "rtos", "fea", "cfd", "gd&t",
        "dcs", "dsa", "oop", "entc", "ece", "eee", "hvdc", "pid"
    }

    SKILL_KEYWORDS = {
        "python", "java", "javascript", "typescript", "sql", "pyspark", "spark",
        "excel", "tableau", "powerbi", "power", "bi", "git", "docker", "kubernetes",
        "cloud", "aws", "gcp", "azure", "api", "database", "query", "pipeline",
        "etl", "elt", "analytics", "analysis", "engineering", "model", "modeling",
        "architecture", "orchestration", "validation", "optimization", "reporting",
        "testing", "automation", "devops", "security",
        "communication", "presentation", "leadership", "stakeholder", "collaboration",
        "negotiation", "coordination", "facilitation", "planning", "organization",
        "management", "project", "program", "product", "operations", "compliance",
        "governance", "documentation", "strategy", "strategic", "research",
        "customer", "client", "service", "sales", "marketing", "procurement",
        "finance", "budgeting", "forecasting", "recruitment", "training", "coaching",
        "mentoring", "problem", "solving", "critical", "thinking", "decision",
        "quality", "audit", "risk", "process", "improvement", "support",
        "civil", "mechanical", "electrical", "electronics", "entc", "ece", "eee",
        "instrumentation", "production", "automobile", "automotive", "industrial",
        "manufacturing", "construction", "structural", "geotechnical", "transportation",
        "surveying", "quantity", "estimation", "mep", "bim", "revit", "autocad",
        "solidworks", "catia", "ansys", "matlab", "simulink", "cnc", "cad", "cam",
        "welding", "machining", "tolerance", "thermodynamics", "fluid", "mechanics",
        "maintenance", "commissioning", "boq", "billing", "site", "execution",
        "piping", "plumbing", "hvac", "load", "distribution", "power", "substation",
        "protection", "relay", "panel", "switchgear", "circuit", "pcb", "vlsi",
        "embedded", "microcontroller", "rtos", "plc", "scada", "dcs", "telecom",
        "rf", "antenna", "signal", "control", "mechatronics", "robotics", "lean",
        "six", "sigma", "safety", "chemical", "petrochemical", "refinery", "process",
        "p&id", "pid", "hysys", "aspen", "distillation", "reactor", "heat",
        "mass", "transfer", "metallurgy", "metallurgical", "material", "foundry",
        "mining", "geology", "marine", "naval", "aerospace", "aeronautical",
        "avionics", "biomedical", "biotechnology", "food", "agricultural",
        "environmental", "wastewater", "water", "sewage", "gis", "remote",
        "sensing", "computer", "software", "backend", "frontend", "fullstack",
        "networking", "cybersecurity", "cloud", "devops", "testing", "embedded",
        "iot", "fpga", "verilog", "vhdl"
    }

    SKILL_HEAD_WORDS = {
        "management", "service", "analysis", "communication", "leadership",
        "planning", "budgeting", "negotiation", "recruitment", "compliance",
        "operations", "marketing", "sales", "design", "research", "training",
        "support", "reporting", "documentation", "coordination", "strategy",
        "governance", "optimization", "engineering", "architecture",
        "construction", "manufacturing", "maintenance", "commissioning",
        "estimation", "surveying", "drafting", "automation", "instrumentation"
    }

    KNOWN_TOOL_BRANDS = {
        "salesforce", "workday", "jira", "sap", "oracle", "servicenow",
        "autocad", "revit", "solidworks", "catia", "ansys", "matlab",
        "simulink", "tableau", "powerbi"
    }

    HARD_SKILL_KEYWORDS = {
        "python", "sql", "pyspark", "spark", "excel", "tableau", "powerbi",
        "git", "database", "query", "pipeline", "etl", "elt", "analytics",
        "analysis", "model", "reporting", "visualization", "governance",
        "compliance", "security", "autocad", "revit", "solidworks", "catia",
        "ansys", "matlab", "simulink", "plc", "scada", "dcs", "vlsi",
        "embedded", "microcontroller", "rf", "telecom", "cnc", "cad", "cam",
        "hvac", "mep", "master", "mdm", "migration", "extraction",
        "transformation", "sap", "fea", "cfd", "bim", "boq", "piping",
        "substation", "switchgear", "pcb", "fpga", "verilog", "vhdl",
        "hysys", "aspen", "distillation", "reactor", "metallurgy", "gis",
        "avionics", "robotics", "mechatronics", "iot", "backend", "frontend"
    }

    TRAILING_NOISE_TOKENS = {"loc", "location", "salary", "shift", "mode", "regards"}

    def __init__(self, nlp):
        self.nlp = nlp
        self.stop_words = nlp.Defaults.stop_words

    def normalize(self, phrase: str) -> str:
        phrase = phrase.lower().strip()
        phrase = re.sub(r"[\(\)\[\]\{\},;:!?.]+", " ", phrase)
        phrase = re.sub(r"\s+", " ", phrase).strip()
        words = phrase.split()
        while len(words) > 1 and words[-1] in self.TRAILING_NOISE_TOKENS:
            words.pop()
        return " ".join(words)

    def is_skill_like(self, chunk) -> bool:
        phrase = self.normalize(chunk.text)
        words = phrase.split()
        lemmas = [tok.lemma_.lower() for tok in chunk]

        if not words or len(words) > 4:
            return False
        if any(any(ch.isdigit() for ch in word) for word in words):
            return False

        non_stop_words = [w for w in words if w not in self.stop_words]
        if not non_stop_words:
            return False
        if all(word in self.SECTION_NOISE for word in non_stop_words):
            return False
        if any(noise in phrase for noise in self.PHRASE_NOISE_CONTAINS):
            return False
        if phrase in self.PHRASE_NOISE_EXACT:
            return False

        if any(word in self.CONTEXT_NOISE_WORDS for word in words):
            if not any(lemma in self.HARD_SKILL_KEYWORDS for lemma in lemmas):
                return False

        if len(words) <= 2 and all(word in self.ROLE_TAXONOMY_WORDS for word in words):
            return False

        if " only " in f" {phrase} ":
            return False
        if any(tok.ent_type_ in {"GPE", "LOC", "FAC"} for tok in chunk):
            return False

        if len(words) == 1 and words[0] in self.ALLOWED_SHORT_SKILLS:
            return True
        if not any(tok.pos_ in {"NOUN", "PROPN", "ADJ"} for tok in chunk):
            return False
        if any(lemma in self.SKILL_KEYWORDS for lemma in lemmas):
            return True
        if words[-1] in self.SKILL_HEAD_WORDS:
            return True
        if any(lemma in self.KNOWN_TOOL_BRANDS for lemma in lemmas):
            return True

        return False

    def extract_phrases(self, doc):
        phrases = []
        for chunk in doc.noun_chunks:
            if self.is_skill_like(chunk):
                phrases.append(self.normalize(chunk.text))
        return sorted(set(phrases))
