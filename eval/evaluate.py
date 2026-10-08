import json
import sys
import time
from pathlib import Path
import csv

# Allow imports from the project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from search import search_documents
from llm import generate_answer
from auth import USERS, create_access_token, verify_access_token
from authorization import authorize_results
from acl_config import DOCUMENT_ACLS
from eval.judge import judge_answer


QUESTIONS_PATH = Path(__file__).parent / "multitenant_questions.json"


def load_questions(path=QUESTIONS_PATH):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def run_evaluation(questions_path=QUESTIONS_PATH):
    questions = load_questions(questions_path)
    # Preserve the original baseline dataset/results as historical artifacts.
    # Its unscoped sources cannot be evaluated against this smaller corpus.
    for item in questions:
        if item.get("username") not in USERS or (item["source"] and item["source"] not in DOCUMENT_ACLS):
            raise ValueError("Evaluation needs a current username and tenant-scoped source; use multitenant_questions.json")

    print(f"\nRunning evaluation on {len(questions)} questions...\n")

    results = []

    for i, item in enumerate(questions, start=1):
        question = item["question"]
        reference_answer = item["reference_answer"]
        expected_source = item["source"]
        username = item["username"]
        user = USERS[username]
        identity = verify_access_token(create_access_token(username, user["tenant_id"], user["role"]))
        if identity is None:
            raise ValueError("Evaluation identity is disabled or revoked")

        print(f"[{i}/{len(questions)}] {question}")

        # Start timer
        start_time = time.perf_counter()

        # Retrieve relevant chunks
        search_results = search_documents(
            query=question,
            tenant_id=identity["tenant_id"],
            role=identity["role"],
            limit=3,
        )
        authorize_results(search_results, identity["tenant_id"], identity["role"])

        # Generate answer
        generated_answer = generate_answer(
            question=question,
            search_results=search_results,
            tenant_id=identity["tenant_id"],
            role=identity["role"],
        )
        # RAG latency excludes the separate LLM correctness judge.
        latency = time.perf_counter() - start_time
        answer_correct = judge_answer(
            question=question,
            reference_answer=reference_answer,
            generated_answer=generated_answer,
        )

        # Check whether expected source was retrieved
        retrieved_sources = [
            result.payload["source"]
            for result in search_results
        ]

        retrieval_hit = expected_source in retrieved_sources if expected_source else None

        result = {
            "username": username,
            "question": question,
            "reference_answer": reference_answer,
            "generated_answer": generated_answer,
            "answer_correct": answer_correct,
            "expected_source": expected_source,
            "retrieved_sources": retrieved_sources,
            "retrieval_hit": retrieval_hit,
            "latency_seconds": round(latency, 3),
        }

        results.append(result)

        print(f"Answer: {generated_answer}")
        print(f"Retrieval hit: {retrieval_hit}")
        print(f"Latency: {latency:.2f}s")
        print("-" * 60)

    return results

def save_results(results, output_path=Path(__file__).parent / "multitenant_results.csv"):

    fieldnames = [
        "username",
        "question",
        "reference_answer",
        "generated_answer",
        "answer_correct",
        "expected_source",
        "retrieved_sources",
        "retrieval_hit",
        "latency_seconds",
    ]

    with open(output_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for result in results:
            row = result.copy()

            row["retrieved_sources"] = ", ".join(
                row["retrieved_sources"]
            )

            writer.writerow(row)

    print(f"\nResults saved to: {output_path}")
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Tenant-aware RAG evaluation (makes DeepSeek generation and judge calls)")
    parser.add_argument("--questions", type=Path, default=QUESTIONS_PATH)
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "multitenant_results.csv")
    args = parser.parse_args()
    results = run_evaluation(args.questions)
    save_results(results, args.output)

    total = len(results)
    correct_answers = sum(
        1 for result in results
        if result["answer_correct"]
    )

    retrieval_hits = sum(
        1 for result in results
        if result["retrieval_hit"]
    )
    retrieval_total = sum(result["retrieval_hit"] is not None for result in results)

    average_latency = sum(
        result["latency_seconds"]
        for result in results
    ) / total

    print("\n========== MULTI-TENANT RESULTS ==========")

    print(f"Questions: {total}")
    print(
        f"Answer accuracy: "
        f"{correct_answers}/{total} "
        f"({correct_answers / total * 100:.1f}%)"
    )

    print(
        f"Retrieval hits (answerable questions): {retrieval_hits}/{retrieval_total} "
        f"({retrieval_hits / retrieval_total * 100 if retrieval_total else 0:.1f}%)"
    )

    print(
        f"Average latency: "
        f"{average_latency:.2f} seconds"
    )

    print("======================================")
