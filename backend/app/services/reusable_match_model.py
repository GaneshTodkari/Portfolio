from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from app.services.jd_parser import JDParser
from app.services.resume_parser import ResumeParser
from app.services.skill_extractor import SkillExtractor


class EmbeddingService:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name)

    def generate_embedding(self, text: str):
        emb = self.model.encode([text or ""], convert_to_numpy=True)
        return emb[0]

    def generate_batch_embeddings(self, texts: list[str]):
        if not texts:
            return np.array([])
        return self.model.encode(texts, convert_to_numpy=True)


@dataclass
class MatchResult:
    final_score: float
    semantic_score: float
    skill_score: float
    experience_score: float
    matching_skills: list[str]
    missing_skills: list[str]
    resume_skills: list[str]
    jd_skills: list[str]


class ResumeMatchModel:
    """
    Full reusable pipeline:
    Resume parse -> JD parse -> skill unit build/match -> semantic score -> exp score -> final score
    """

    def __init__(
        self,
        embedding_model_name: str = "all-MiniLM-L6-v2",
        skill_similarity_threshold: float = 0.6,
        cluster_similarity_threshold: float = 0.75,
    ):
        self.embedding_model_name = embedding_model_name
        self.skill_similarity_threshold = skill_similarity_threshold
        self.cluster_similarity_threshold = cluster_similarity_threshold
        self._embedding_service = None
        self._resume_parser = None
        self._jd_parser = None
        self._matcher = None

    @property
    def embedding_service(self) -> EmbeddingService:
        if self._embedding_service is None:
            self._embedding_service = EmbeddingService(self.embedding_model_name)
        return self._embedding_service

    @property
    def resume_parser(self) -> ResumeParser:
        if self._resume_parser is None:
            self._resume_parser = ResumeParser()
        return self._resume_parser

    @property
    def jd_parser(self) -> JDParser:
        if self._jd_parser is None:
            self._jd_parser = JDParser()
        return self._jd_parser

    @property
    def matcher(self) -> SkillExtractor:
        if self._matcher is None:
            self._matcher = SkillExtractor(self.embedding_service)
        return self._matcher

    def score_resume(self, resume_path: str, jd_text: str) -> MatchResult:
        resume_data = self.resume_parser.parse(resume_path)
        jd_data = self.jd_parser.parse(jd_text)

        resume_skills = self._canonicalize_skills(resume_data.get("candidate_phrases", []))
        jd_skills = self._canonicalize_skills(jd_data.get("candidate_phrases", []))

        matching_skills, missing_skills, skill_score_ratio = self.matcher.semantic_skill_match(
            jd_text=jd_data.get("clean_text", ""),
            jd_skills=jd_skills,
            resume_skills=resume_skills,
        )
        skill_score = float(skill_score_ratio) * 100.0

        semantic_score = self._semantic_score(
            resume_data.get("clean_text", ""),
            jd_data.get("clean_text", ""),
        )

        required_exp = float(jd_data.get("required_experience", 0.0))
        resume_exp = float(resume_data.get("experience_years", 0.0))
        experience_score = self._experience_score(resume_exp, required_exp)

        final_score = round((0.55 * semantic_score) + (0.30 * skill_score) + (0.15 * experience_score), 2)

        return MatchResult(
            final_score=final_score,
            semantic_score=round(semantic_score, 2),
            skill_score=round(skill_score, 2),
            experience_score=round(experience_score, 2),
            matching_skills=matching_skills,
            missing_skills=missing_skills,
            resume_skills=resume_skills,
            jd_skills=jd_skills,
        )

    def _semantic_score(self, resume_text: str, jd_text: str) -> float:
        if not resume_text.strip() or not jd_text.strip():
            return 0.0
        r = self.embedding_service.generate_embedding(resume_text)
        j = self.embedding_service.generate_embedding(jd_text)
        sim = cosine_similarity([r], [j])[0][0]
        return float(np.clip(sim, 0.0, 1.0) * 100.0)

    def _experience_score(self, resume_exp: float, required_exp: float) -> float:
        if required_exp <= 0:
            return 100.0
        return float(min(resume_exp / required_exp, 1.0) * 100.0)

    def _canonicalize_skills(self, skills: list[str]) -> list[str]:
        canon = []
        for s in skills:
            t = " ".join(s.lower().strip().split())
            if t and t not in canon:
                canon.append(t)
        return canon
