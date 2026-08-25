"""
RAGLab evaluation harness.

Runs a set of questions (eval_questions.json) against the running RAG backend and
scores the system on the metrics that matter for RAG:

  - Keyword recall   : did the ANSWER contain the terms a correct answer should?
  - Retrieval sim    : average similarity of retrieved chunks (retrieval quality)
  - Answer relevance : semantic relevance the pipeline reports
  - Hallucination    : did the pipeline flag the answer as ungrounded?
  - Latency          : end-to-end query time

Usage:
    # 1. Make sure the backend is running (./run.sh) and the RIGHT document is indexed.
    # 2. Edit eval_questions.json so the questions/keywords match that document.
    venv/bin/python evaluate.py
    venv/bin/python evaluate.py --url http://127.0.0.1:5050 --questions eval_questions.json
"""

import argparse
import json
import time
import urllib.request


def ask(base_url, question, timeout=180):
    """POST one question to the backend's /query endpoint and return the JSON."""
    payload = json.dumps({"question": question}).encode()
    req = urllib.request.Request(
        f"{base_url}/query", data=payload,
        headers={"Content-Type": "application/json"}, method="POST",
    )
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read().decode())
    data["_elapsed"] = time.time() - t0
    return data


def keyword_recall(answer, keywords):
    """Fraction of expected keywords present in the answer (case-insensitive)."""
    if not keywords:
        return None
    a = answer.lower()
    hits = sum(1 for k in keywords if k.lower() in a)
    return hits / len(keywords)


def main():
    ap = argparse.ArgumentParser(description="Evaluate the RAGLab backend on a question set.")
    ap.add_argument("--url", default="http://127.0.0.1:5050", help="Backend base URL")
    ap.add_argument("--questions", default="eval_questions.json", help="Path to question set")
    args = ap.parse_args()

    with open(args.questions) as f:
        cases = json.load(f)["questions"]

    print(f"\nRAGLab evaluation — {len(cases)} questions against {args.url}\n")
    print(f"{'#':>2}  {'recall':>7}  {'relev':>6}  {'sim':>5}  {'halluc':>6}  {'time':>6}  question")
    print("-" * 92)

    recalls, relevances, sims, times, halluc_count = [], [], [], [], 0
    for i, case in enumerate(cases, 1):
        q = case["question"]
        try:
            d = ask(args.url, q)
        except Exception as e:
            print(f"{i:>2}  {'ERROR':>7}  {'':>6}  {'':>5}  {'':>6}  {'':>6}  {q[:40]} :: {e}")
            continue

        m = d.get("metrics", {})
        gen, ret = m.get("generation", {}), m.get("retrieval", {})
        rec = keyword_recall(d.get("response", ""), case.get("keywords"))
        relev = gen.get("relevance")
        sim = ret.get("average_similarity")
        halluc = gen.get("has_hallucination", False)
        t = d.get("_elapsed", m.get("total_time", 0))

        if rec is not None:
            recalls.append(rec)
        if relev is not None:
            relevances.append(relev)
        if sim is not None:
            sims.append(sim)
        times.append(t)
        halluc_count += 1 if halluc else 0

        rec_s = f"{rec*100:5.0f}%" if rec is not None else "   —"
        print(f"{i:>2}  {rec_s:>7}  {relev*100:5.0f}%  {sim*100:4.0f}%  "
              f"{'YES' if halluc else 'no':>6}  {t:5.1f}s  {q[:44]}")

    def avg(xs):
        return sum(xs) / len(xs) if xs else 0.0

    print("-" * 92)
    print("\nSUMMARY")
    print(f"  Answer keyword recall : {avg(recalls)*100:.0f}%   (did answers contain the expected facts)")
    print(f"  Answer relevance      : {avg(relevances)*100:.0f}%   (semantic match to the question)")
    print(f"  Retrieval similarity  : {avg(sims)*100:.0f}%   (quality of retrieved chunks)")
    print(f"  Hallucinations flagged: {halluc_count}/{len(cases)}")
    print(f"  Avg latency           : {avg(times):.1f}s\n")


if __name__ == "__main__":
    main()
