# Final offline validation

Generated: 2026-10-08T04:26:16.513166+00:00. Status: **offline_checks_passed**.

The four-stage roadmap has local implementations of baseline RAG, tenant/role authorization, semantic caching and routing, and bounded GraphRAG. This report verifies the listed offline cases; it does not certify production readiness or live GraphRAG answer correctness.

## Verification

| Check | Result |
| --- | --- |
| Regression tests | 198/198; 0 skipped |
| adversarial | 22/22 |
| routing | 17/17 |
| rag_cache | 15/15 |
| graph | 16/16 |
| graph_cache | 14/14 |
| deployment_smoke | 11/11 |
| Real provider calls in this run | 0 |
| Stored payload fingerprint unchanged | yes |

## Local pipeline timing

300 measured requests across 10 scenarios; 30 samples and 5 excluded warmups per scenario. Corpus: 4 points. Cases run in alternating forward/reverse round-robin order with independent caches.

| Scenario | Median ms | p95 ms | Mocked generations | Retrievals | Graph expansions |
| --- | ---: | ---: | ---: | ---: | ---: |
| rag_fresh | 5.369 | 5.647 | 30 | 30 | 0 |
| rag_exact | 5.362 | 5.611 | 0 | 30 | 0 |
| rag_semantic | 5.497 | 5.834 | 0 | 30 | 0 |
| graph_fresh | 7.879 | 8.787 | 30 | 30 | 30 |
| graph_exact | 7.841 | 8.438 | 0 | 30 | 30 |
| graph_semantic | 7.910 | 8.785 | 0 | 30 | 30 |
| employee_abstain | 5.355 | 6.334 | 0 | 30 | 0 |
| foreign_tenant_abstain | 5.444 | 6.277 | 0 | 30 | 0 |
| incomplete_graph_abstain | 7.695 | 8.163 | 0 | 30 | 30 |
| invalid_token | 0.322 | 0.369 | 0 | 0 | 0 |

Sequential in-process /ask including embedding, storage opening, authorization, routing, graph and cache work. Excludes login, model startup, HTTP transport, seeds and warmup. Completions and audit writes are mocked; not live latency or a load test.

Cache-hit scenarios still perform retrieval and, for GraphRAG, graph expansion. Zero mocked generations on these hits demonstrate call avoidance for these requests. They do not establish a live latency reduction, cost saving, traffic hit rate, or answer accuracy. p95 uses the nearest-rank definition.

## ACL microbenchmark

200 measured query pairs per identity, after 10 excluded pairs. Filtered/unfiltered execution order alternates. The unfiltered query exists only as an offline timing control.

| Identity | Filtered median ms | Unfiltered median ms | Paired overhead median ms | Validation median ms |
| --- | ---: | ---: | ---: | ---: |
| acme/employee | 0.1207 | 0.1138 | 0.0073 | 0.0013 |
| acme/hr | 0.1256 | 0.1138 | 0.0113 | 0.0020 |
| globex/hr | 0.1245 | 0.1105 | 0.0122 | 0.0020 |

Qdrant query and defense validation only; excludes model loading, embedding, storage opening, HTTP, and LLM. Tiny local corpus; not a production load benchmark.

## Reproduce

In the full development workspace, stop any RAG process sharing this local Qdrant directory, then run:

```sh
.venv/bin/python -m eval.final_validation --iterations 30 --acl-iterations 200
```

The runner uses dummy provider credentials, forces cached-model offline mode, blocks socket connections in the coordinator and test subprocess, and runs storage-owning phases sequentially. Detailed fresh reports and private logs are written to `dist/final-validation/`; the aggregate machine-readable result is `eval/final_results.json`. Historical baseline/live results are preserved. No ingestion runs. Evaluation tools, tests, and raw reports are excluded from the deployment archive; only this aggregate Markdown summary is packaged. Streamlit interface tests run separately in the UI environment; see ui/README.md.

Validation input SHA-256: `a0f83ef58928cceb24e0c0d5c77e51025225faa6ee683de4c5a55b4d74867468`.

## Remaining work

- Live GraphRAG fact/citation evaluation remains pending explicit approval for the specified demo evidence and DeepSeek destination. This run sends no evidence to a provider.
- This offline runner does not install dependencies or deploy containers. Separate native installation validation and its platform limits are recorded in the [deployment runbook](deploy/README.md). Container deployment remains untested.
- Production scaling needs shared persistent identity/policy storage, server-based vector storage, shared caching, and multi-instance controls. Current revocation, caching, metrics, and admission limits are process local.
- Broader GraphRAG needs a larger corpus, broader extraction, and retrieval/answer comparisons. The current graph handles limited English salary/policy sentence patterns; no graph retrieval-quality gain has been established.
- Public operation needs deployment-specific TLS, rate limits, slow-client controls, monitoring, backup/recovery drills, and replacement of public demo credentials. No public service is published.
