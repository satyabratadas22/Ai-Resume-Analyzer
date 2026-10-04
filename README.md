# AI Resume Analyzer API

REST API that scores how well a resume fits a job description, using a classical
TF-IDF baseline and a semantic sentence-embedding model, with SQL storage of every
analysis and an evaluation script to compare the two on labeled data.

## Features
- `POST /analyze` - returns match score (0-100), matched and missing skills
- `GET /history` - recent analyses stored in SQLite
- Two backends: `tfidf` (baseline) and `embedding` (sentence-transformers, all-MiniLM-L6-v2)
- `evaluate.py` - ROC-AUC and best-threshold accuracy on labeled resume/JD pairs

## Run
    pip install -r requirements.txt
    uvicorn app.main:app --reload
    # open http://127.0.0.1:8000/docs

## Evaluate
Add your own labeled resume/JD pairs to `data/labeled_pairs.csv`
(columns: resume, job_description, label; 1 = good fit, 0 = poor fit), then:

    python evaluate.py --backend both

To just try the pipeline, use the small illustrative file:

    python evaluate.py --backend both --data data/sample_pairs.csv

> **Note:** `data/sample_pairs.csv` contains a few hand-written *sample* pairs to demonstrate the
> pipeline. Results on it are not a real benchmark. Real results will be reported here once
> `labeled_pairs.csv` is filled with genuine labeled data.

## Structure
    app/matcher.py   scoring backends and skill-gap detection
    app/db.py        SQLite persistence
    app/main.py      FastAPI endpoints
    evaluate.py      backend comparison on labeled data
    data/            labeled_pairs.csv (your data), sample_pairs.csv (demo only)
