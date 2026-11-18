# Evidence audit (Part A1)

Audit date: 2026-10-28. Scope: repository files and all saved capsule `events.jsonl` members. The requested `docs/P1_HANDOFF.md` does not exist in this checkout (`rg --files -uu` inventory); `docs/DERIVATION.md` was read. This audit uses only evidence found in the repository. A claim without supporting evidence is FAIL.

| Claim | Result | Evidence and exact count/value |
|---|---|---|
| Prediction committed before any run | **PASS** | `prereg/formal-2026-10-18-t4/receipt.json`: prediction commit `7baff5121ed643f27ba06959065855a8913f8e8c`; receipt observed 2026-10-18 14:25:17 UTC and says `first_model_measurement_at_utc: null`. `git log` shows that commit at 2026-10-18 19:54:52 +05:30. The first acceptance evidence is dated later on 2026-10-18; later P1 cohort starts 2026-10-19. |
| 52 probes over 9 lengths | **PASS, with scope stated** | `data/processed/p1-2026-10-19/probes.csv`: 80 total workers, 52 `primary` records (27 default + 25 expandable), across exactly 9 unique lengths: 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 114688. These are not 52 workers for each allocator. |
| KV slope 28,672 B/token | **PASS** | `data/processed/p1-2026-10-19/analysis.json`: both allocator primary fits have `cache_bytes.slope_bytes_per_token` 28,672.000000000004, R²=1.0, 9 lengths. `docs/DERIVATION.md` derives 28,672 B/token from the pinned config. |
| Held-out validation within 0.01% | **FAIL (strict threshold)** | `data/processed/p1-2026-10-19/held_out_validation.csv`: default error = -0.010449242821379524% (absolute value exceeds 0.01%); expandable = 0%. Thus the claim for both allocators is false under a strict ≤0.01% rule, despite rounding to -0.01%. |
| A real CUDA OOM at 131,072 | **PASS** | Searched all 76 `events.jsonl` members inside saved capsule ZIPs and repository JSON/JSONL/Markdown evidence for `cuda_oom`. One event is present twice as `state/.../events.jsonl` and `local/.../events.jsonl` copies of the same journal in `artifacts/experiments/2026-10-21-boundary/checkpoint.zip`. It records `status: cuda_oom`, `failed_oom_upper_tokens: 131072`; the capsule verification records a forward `torch.cuda.OutOfMemoryError` requesting 32.00 GiB. This is one OOM event, not two independent attempts. |
| vLLM block sizes matched | **PARTIAL** | `configs/transfer/published_verification.json` compares predicted 917,504 bytes per 16-token Qwen2.5-7B block with rounded published `0.87 MB`. Its own caveat says decimal vs binary units are undefined: error is 5.46% if 0.87 means decimal MB, or 0.57% if MiB. It is an approximate KV-submodel comparison, not an exact match and not a measured GPU capacity. |

## Notes on boundary provenance

The saved OOM supports the default-allocator interval 114,688–131,071 under a monotonicity assumption, not an exact boundary. `docs/OOM_BOUNDARY.md` and `artifacts/experiments/2026-10-21-boundary/verification.json` state this. No expandable OOM is in the saved evidence. The request for a paired multi-OOM bisection is new work and has not been run.
