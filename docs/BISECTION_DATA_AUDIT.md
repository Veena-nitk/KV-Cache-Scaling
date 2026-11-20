# Audit of submitted paired-bisection data

Audit date: 2026-11-20. Scope: `artifacts/bisect-final-complete/`. This review is read-only with respect to the submitted logs. **Result: FAIL for boundary acceptance.** The summary's endpoint values cannot be authenticated from a complete capsule and do not satisfy the recorded protocol when checked against the supplied JSONL.

## Claimed summary (not accepted)

`audit/bisection_summary.json` claims GPU UUID `GPU-bf1116ef-57c5-9ec1-593e-0a24bd1339d7`, default `[118784,119296]`, expandable `[127488,128000]`, and delta 8,704 tokens (7.327586%). The matching UUID appears in `audit/receipt.json`, but neither probe JSONL contains a GPU UUID, worker runtime, driver version, run ID, durable event journal, or per-probe environment. Matching two summary fields is not evidence that all probes used that GPU or runtime.

## Protocol and provenance failures

| Check | Result | Evidence |
|---|---|---|
| Complete capsule and member manifest | **FAIL** | No capsule ZIP, capsule manifest, worker stdout/stderr, run manifest, or original per-probe checkpoint is present. Only two reduced JSONL files and three audit JSON files were supplied. |
| Receipt binds a committed, pushed prediction and code | **FAIL** | `audit/receipt.json` has no prediction commit SHA, receipt SHA, code commit, remote evidence URL, receipt observation time, or reviewer evidence. `audit/prediction.json` has `gpu_measurements_authorized: false` and an empty `input_sha256` object. |
| Per-probe GPU UUID and runtime evidence | **FAIL** | UUID is present only in receipt and summary. Probe JSONL has no UUID or runtime metadata. Earlier default entries explicitly record Python 3.13 mismatch; successful entries do not record the required Python 3.11 / PyTorch 2.6.0 / CUDA 12.4 / Transformers 4.51.3 tuple. |
| Default deliberate-dispatch cap | **FAIL** | `probes_default.jsonl` has 27 rows, including 13 `attempt_kind=deliberate` entries and 10 typed `cuda_oom` statuses. This exceeds the 6 deliberate-dispatch per-allocator cap. It also contains an apparent 131,072 success with null peak before later 131,072 OOM records, runtime errors, and repeated anchors/bisection sequences. |
| Expandable deliberate-dispatch cap | **FAIL** | `probes_expandable.jsonl` has 14 rows, including 8 deliberate entries, exceeding the cap of 6. It contains two sets of three 114,688 anchors and two 131,072 OOM confirmations without run IDs to distinguish cohorts. |
| Default endpoint sequence | **FAIL** | A later subsequence appears to reach 118,784 success and 119,296 OOM, but the file also contains repeated launches, earlier failures, and another repetition of that sequence. With no run IDs or accepted checkpoint chain, no unique compliant cohort can be established. |
| Expandable endpoint sequence | **FAIL** | The first apparent sequence reaches 126,976 success and 128,000 OOM, a 1,024-token gap. The summary's claimed 127,488 success is later in a restarted sequence whose prior 122,880 success makes 126,976—not 127,488—the next midpoint. There is no valid logged same-cohort 127,488 result that closes the summary's claimed bracket. |

The supplied file SHA-256 values establish the current bytes of these local files only; there is no trusted preregistered hash manifest or signed receipt against which to verify them:

| File | SHA-256 |
|---|---|
| `state/probes_default.jsonl` | `242e2b8ea2209f0e296ab154f3b094cecd0964ab6038e362a47ca55334edf29c` |
| `state/probes_expandable.jsonl` | `bc4ca1c0e7b8ce569ea4ba9f93344c4f7b2d6c19076b4faab637a8f9ec19d65b` |
| `audit/prediction.json` | `1cb83612d76e71abbc2d36f0876ff1f9b8b1c3f28296764578284d15c89ede55` |
| `audit/bisection_summary.json` | `1252fd905444fa9e1e70dde1234891152e284b53dab964fcb57f138864676a08` |
| `audit/receipt.json` | `b4e175c6330cd02bab16ba0e65992d16d093f53cce3b9d8da65b82a839295ac9` |

## Disposition

Do not use the claimed paired endpoints or delta in Table A, Table B, or the discrepancy tables. Their status remains `MISSING` pending one auditable, protocol-compliant cohort with per-probe UUID/runtime evidence, unique run IDs, checkpoint/member hashes, and no retries or cap violations. The earlier default-only 114,688–131,072 bracket remains the only previously verified boundary evidence; it is not a paired bisection result.
