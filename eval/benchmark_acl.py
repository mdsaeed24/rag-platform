"""Offline Qdrant ACL microbenchmark. Never constructs or sends LLM context.

Unfiltered searches are a timing control only, confined to this local script.
They are not available through the API or search_documents().
"""

import argparse
import json
import math
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qdrant_client import QdrantClient
from authorization import authorize_results
from search import COLLECTION_NAME, QDRANT_PATH, build_acl_filter, model


def summary(samples):
    values = sorted(samples)
    return {"median_ms": round(statistics.median(values), 6), "p95_ms": round(values[math.ceil(len(values) * .95) - 1], 6)}


def benchmark(iterations=200):
    if iterations < 1:
        raise ValueError("iterations must be positive")
    vector = model.encode("What does the CEO earn?").tolist()
    client = QdrantClient(path=str(QDRANT_PATH))
    cases = []
    try:
        point_count = client.get_collection(COLLECTION_NAME).points_count
        for tenant, role in (("acme", "employee"), ("acme", "hr"), ("globex", "hr")):
            filtered, unfiltered, validation, deltas = [], [], [], []
            acl_filter = build_acl_filter(tenant, role)
            for trial in range(iterations + 10):
                times = {}
                # Alternate execution order to reduce cache/order bias.
                order = (True, False) if trial % 2 else (False, True)
                for with_filter in order:
                    start = perf_counter()
                    result = client.query_points(COLLECTION_NAME, query=vector, query_filter=acl_filter if with_filter else None, limit=3).points
                    times[with_filter] = (perf_counter() - start) * 1000
                    if with_filter:
                        start = perf_counter()
                        authorize_results(result, tenant, role)
                        validation_ms = (perf_counter() - start) * 1000
                if trial >= 10:
                    filtered.append(times[True])
                    unfiltered.append(times[False])
                    validation.append(validation_ms)
                    deltas.append(times[True] - times[False])
            cases.append({"tenant_id": tenant, "role": role, "filtered_qdrant": summary(filtered), "unfiltered_timing_control": summary(unfiltered), "paired_filter_overhead": summary(deltas), "defense_validation": summary(validation)})
    finally:
        client.close()
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "collection": COLLECTION_NAME, "point_count": point_count,
        "iterations_per_identity": iterations, "warmup_pairs": 10,
        "scope": "Qdrant query and defense validation only; excludes model loading, embedding, storage opening, HTTP, and LLM. Tiny local corpus; not a production load benchmark.",
        "cases": cases,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "acl_benchmark.json")
    args = parser.parse_args()
    report = benchmark(args.iterations)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
