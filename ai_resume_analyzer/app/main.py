"""REST API: POST /analyze, GET /history, GET /health."""
from typing import Literal
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from app import db
from app.matcher import match_score, skill_gap

app = FastAPI(title="AI Resume Analyzer API")
db.init_db()


class AnalyzeRequest(BaseModel):
    resume: str
    job_description: str
    backend: Literal["tfidf", "embedding"] = "embedding"


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/analyze")
def analyze(req: AnalyzeRequest):
    if not req.resume.strip() or not req.job_description.strip():
        raise HTTPException(status_code=422, detail="resume and job_description must be non-empty")
    score = round(match_score(req.resume, req.job_description, req.backend) * 100, 1)
    matched, missing = skill_gap(req.resume, req.job_description)
    analysis_id = db.save_analysis(req.backend, score, matched, missing)
    return {
        "id": analysis_id,
        "backend": req.backend,
        "match_score": score,
        "matched_skills": matched,
        "missing_skills": missing,
    }


@app.get("/history")
def history(limit: int = 20):
    return db.list_analyses(limit)
