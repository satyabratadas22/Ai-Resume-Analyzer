"""Evaluate matching backends on YOUR labeled resume/JD pairs.

data/labeled_pairs.csv columns: resume,job_description,label
  label = 1 if the resume is a good fit for the JD, 0 if it is a poor fit.

Usage: python evaluate.py [--backend tfidf|embedding|both]
Reports ROC-AUC and the accuracy at the best threshold for each backend.
"""
import argparse
import csv
from sklearn.metrics import roc_auc_score, accuracy_score
from app.matcher import match_score


def load(path="data/labeled_pairs.csv"):
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return [r["resume"] for r in rows], [r["job_description"] for r in rows], [int(r["label"]) for r in rows]


def evaluate(backend, resumes, jds, labels):
    scores = [match_score(r, j, backend) for r, j in zip(resumes, jds)]
    auc = roc_auc_score(labels, scores)
    best_acc, best_t = 0.0, 0.0
    for t in sorted(set(scores)):
        acc = accuracy_score(labels, [int(s >= t) for s in scores])
        if acc > best_acc:
            best_acc, best_t = acc, t
    return auc, best_acc, best_t


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--backend", default="both", choices=["tfidf", "embedding", "both"])
    p.add_argument("--data", default="data/labeled_pairs.csv")
    a = p.parse_args()
    resumes, jds, labels = load(a.data)
    if len(set(labels)) < 2:
        raise SystemExit(
            f"{a.data} needs rows with both labels (0 and 1). Add your own labeled pairs, "
            "or run with --data data/sample_pairs.csv to try the pipeline."
        )
    print(f"{len(labels)} labeled pairs from {a.data}")
    for b in (["tfidf", "embedding"] if a.backend == "both" else [a.backend]):
        auc, acc, t = evaluate(b, resumes, jds, labels)
        print(f"{b:10s} ROC-AUC={auc:.3f}  best-threshold accuracy={acc:.3f} (threshold={t:.3f})")
