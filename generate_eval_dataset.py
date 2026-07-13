import csv
import json
from pathlib import Path
from retrieval_pipelin import query
from LLM import chat

QUERY_LOG_PATH = Path("logs/query_latency.csv")
OUTPUT_PATH = Path("eval_dataset.json")

GROUND_TRUTH_PROMPT = """You are creating a reference answer for evaluation purposes.
Based ONLY on the context below, write the single most accurate, complete answer
to the question. This will be used as the "correct answer" to grade other systems
against, so be precise and don't add anything not supported by the context."""


def load_unique_questions() -> list[str]:
    if not QUERY_LOG_PATH.exists():
        raise FileNotFoundError(f"{QUERY_LOG_PATH} not found — ask some questions via retrieval.py first.")
    seen = set()
    questions = []
    with open(QUERY_LOG_PATH, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            q = row["question"].strip()
            if q and q not in seen:
                seen.add(q)
                questions.append(q)
    return questions


def build_dataset(mode: str = "naive") -> list[dict]:
    questions = load_unique_questions()
    print(f"Found {len(questions)} unique questions in {QUERY_LOG_PATH}\n")
    dataset = []
    for i, q in enumerate(questions, 1):
        print(f"[{i}/{len(questions)}] {q}")
        result = query(q, mode=mode)
        answer = result["answer"]
        context = result["context"]
        ground_truth_prompt = f"Context:\n{context}\n\nQuestion: {q}"
        ground_truth = chat(GROUND_TRUTH_PROMPT, ground_truth_prompt)
        dataset.append({"question": q, "answer": answer, "contexts": [context], "ground_truth": ground_truth})
    return dataset


if __name__ == "__main__":
    dataset = build_dataset()
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)
    print(f"\nSaved {len(dataset)} entries to {OUTPUT_PATH}")
    print("Review/edit ground_truth fields before running evaluation if accuracy matters.")