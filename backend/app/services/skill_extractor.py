import re

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


class SkillExtractor:
    def __init__(self, embedding_service):
        self.embedding_service = embedding_service
        self.similarity_threshold = 0.60
        self.alias_map = {
            "version control tools": "git",
            "version control": "git",
            "query writing": "sql",
            "advanced query writing": "sql",
            "data validation": "validation",
            "data pipeline building": "pipeline",
            "workflow orchestration": "orchestration",
            "power bi": "powerbi",
            "power-bi": "powerbi",
            "ms excel": "excel",
            "m s excel": "excel",
            "master data management loc": "master data management",
            "sap master data": "master data management",
            "sap mdm": "master data management",
            "computer aided design": "cad",
            "computer-aided design": "cad",
            "computer aided manufacturing": "cam",
            "computer-aided manufacturing": "cam",
            "finite element analysis": "fea",
            "geometric dimensioning and tolerancing": "gd&t",
            "geometric dimensioning & tolerancing": "gd&t",
            "building information modeling": "bim",
            "quantity surveying": "quantity estimation",
            "bill of quantities": "boq",
            "programmable logic controller": "plc",
            "distributed control system": "dcs",
            "printed circuit board": "pcb",
            "very large scale integration": "vlsi",
            "embedded systems": "embedded",
            "signal processing": "signal",
            "control systems": "control",
            "process design": "process engineering",
            "process safety": "safety",
            "piping and instrumentation diagram": "p&id",
            "piping & instrumentation diagram": "p&id",
            "data structures and algorithms": "dsa",
            "object oriented programming": "oop",
            "object-oriented programming": "oop",
            "electronics and telecommunication": "entc",
            "electronic and telecommunication": "entc",
            "electronics and communication": "ece",
            "electrical and electronics": "eee",
        }

    def _normalize(self, text: str) -> str:
        text = text.lower().strip()
        text = re.sub(r"[\(\)\[\]\{\},;:!?.]+", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return self.alias_map.get(text, text)

    def _lexical_match(self, jd_skill: str, resume_skills: list[str]) -> bool:
        jd_norm = self._normalize(jd_skill)
        jd_tokens = set(jd_norm.split())
        if not jd_tokens:
            return False

        for rs in resume_skills:
            rs_norm = self._normalize(rs)
            if jd_norm == rs_norm:
                return True

            rs_tokens = set(rs_norm.split())
            if not rs_tokens:
                continue

            overlap = len(jd_tokens & rs_tokens) / max(len(jd_tokens), len(rs_tokens))
            if overlap >= 0.6:
                return True
            if jd_norm in rs_norm or rs_norm in jd_norm:
                return True
        return False

    def semantic_skill_match(self, jd_text: str, jd_skills: list[str], resume_skills: list[str]):
        if not jd_skills or not resume_skills:
            return [], jd_skills, 0.0

        jd_doc_embedding = self.embedding_service.generate_embedding(jd_text)
        jd_embeddings = self.embedding_service.generate_batch_embeddings(jd_skills)
        resume_embeddings = self.embedding_service.generate_batch_embeddings(resume_skills)

        weighted_match_score = 0.0
        total_weight = 0.0
        matching = []
        missing = []

        for i, jd_emb in enumerate(jd_embeddings):
            importance = max(cosine_similarity([jd_emb], [jd_doc_embedding])[0][0], 0.0)
            total_weight += importance

            lexical_hit = self._lexical_match(jd_skills[i], resume_skills)
            similarities = cosine_similarity([jd_emb], resume_embeddings)[0]
            max_sim = float(np.max(similarities)) if len(similarities) else 0.0

            if lexical_hit or max_sim >= self.similarity_threshold:
                matching.append(jd_skills[i])
                weighted_match_score += importance
            else:
                missing.append(jd_skills[i])

        skill_score = (weighted_match_score / total_weight) if total_weight else 0.0
        return matching, missing, round(skill_score, 3)
