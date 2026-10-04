"""Matching engine with two interchangeable backends:
  - "tfidf":     classical TF-IDF + cosine similarity (fast baseline)
  - "embedding": sentence-transformer embeddings + cosine similarity (semantic)
"""
import re
from functools import lru_cache
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

SKILL_PHRASES = [
    "machine learning", "deep learning", "natural language processing", "nlp",
    "large language models", "llm", "ai agents", "data structures", "algorithms",
    "rest api", "object oriented", "oop", "version control", "git", "sql", "python",
    "predictive modeling", "data science", "data engineering", "cloud",
    "mlops", "automation", "debugging", "code reviews", "technical documentation",
    "sdlc", "scikit-learn", "pandas", "numpy",
]


def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9+#.\- \n]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


@lru_cache(maxsize=1)
def _load_embedder(name: str = "all-MiniLM-L6-v2"):
    from sentence_transformers import SentenceTransformer  # imported lazily
    return SentenceTransformer(name)


def score_tfidf(resume: str, jd: str) -> float:
    vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    m = vec.fit_transform([clean_text(resume), clean_text(jd)])
    return float(cosine_similarity(m[0], m[1])[0][0])


def score_embedding(resume: str, jd: str) -> float:
    model = _load_embedder()
    a, b = model.encode([resume, jd], normalize_embeddings=True)
    return float(max(0.0, a @ b))


def match_score(resume: str, jd: str, backend: str = "tfidf") -> float:
    """Similarity in [0, 1]."""
    if backend == "embedding":
        return score_embedding(resume, jd)
    return score_tfidf(resume, jd)


def skill_gap(resume: str, jd: str):
    r, j = clean_text(resume), clean_text(jd)
    pat = lambda s: rf"\b{re.escape(s)}s?\b"
    in_jd = [s for s in SKILL_PHRASES if re.search(pat(s), j)]
    found = [s for s in in_jd if re.search(pat(s), r)]
    missing = [s for s in in_jd if s not in found]
    return found, missing
