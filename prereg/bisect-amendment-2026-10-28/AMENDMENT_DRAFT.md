# DRAFT — prospective amendment for paired allocator boundary bisection

**Status: draft only; not applied, committed, or authority to run.** This amendment supplements the frozen protocol in `prereg/final-2026-10-19`; it does not modify that preregistration or any receipt/capsule. TA approval and a separately frozen coordinator/receipt are prerequisites.

## OOM-budget amendment and new experiment identity

The earlier protocol's one-use deliberate-OOM allowance cannot execute this objective. If approved by the TA, this amendment authorizes a **separate new experiment ID** with at most six deliberate dispatches per allocator and twelve total. Each dispatch is consumed durably before launch; successful deliberate probes also consume a slot. Any runtime interruption, unresolved start, non-OOM worker error, anchor OOM, or exhausted cap stops the affected cohort without retry. The original global one-use ledger, receipts, and capsules remain preserved and are not reused as this cohort's budget. A new receipt must bind this exact amendment, code, GPU UUID, and context inventory before the first model probe.

## Question and workload

For the already specified Qwen2.5-1.5B cold single prefill (FP16, batch 1, DynamicCache, reduced logits, forced efficient SDPA, same pinned runtime), bracket the largest successful length and first typed `cuda_oom` independently under default allocator and `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`. Use one Colab session and the same physical GPU UUID for both allocator arms. No data are presumed for results.

## Sequence and stopping rule

For each allocator, in the same session/GPU UUID:

1. Run three independent fresh-worker repeats at L=114,688 as lower-anchor checks. All three must be successful. Any unexpected failure, unsupported expandable warning, runtime/identity drift, or non-idle return stops that arm; do not retry automatically.
2. Run one deliberate confirmation at L=131,072. It must return typed `cuda_oom` to establish the registered upper endpoint. A success means the stated bracket is not established; stop and amend before selecting a new upper endpoint. Generic errors, timeout, interruption, or missing output are not OOM.
3. Bisect the current interval with integer midpoint probes. For a successful midpoint replace the lower endpoint; for a typed `cuda_oom` replace the upper endpoint. Continue only until `first_cuda_oom - last_success <= 512` tokens. A non-OOM error, lost checkpoint, or identity change stops the arm without retries.
4. Complete the default arm and checkpoint it before the expandable arm, then verify the GPU UUID remains identical. Keep allocator-arm records separate; report the observed token delta only if both arms meet the endpoint rule.

Starting from 114,688 and 131,072, five midpoint outcomes suffice to reduce a 16,384-token interval to 512. Thus the planned maximum is **6 typed OOM outcomes per allocator** (one upper-endpoint confirmation plus at most five failed midpoint probes), **12 across both allocators**. This is an upper bound, not a target. Count any unexpected OOM during an anchor or other planned worker against that arm's cap; stop that arm immediately after an anchor OOM. Do not launch a probe that could exceed the cap. There are six planned deliberate OOM opportunities per allocator; total probe count includes successful anchors and successful midpoints and is not the OOM budget.

Every worker remains fresh-process, journaled and checkpointed before the next dispatch. The coordinator accepts only a complete pushed receipt and the notebook requires the user to download and acknowledge each capsule SHA before the next dispatch. Save request, result/error event, stdout/stderr, environment, allocator warning, NVML/CUDA identity, budget ledger, and capsule manifest/hash. Keep failed-forward partial counters separate from completed-forward peak results. Analyze with signed boundary error `(prediction - measured_boundary)/measured_boundary`; report exact endpoint definitions and the monotonicity assumption.

## Existing one-OOM harness and separate amendment runner

The original frozen protocol sets `deliberate_oom_attempt_budget: 1` (`configs/experiment.json:21`), and its project-wide `AttemptLedger` permits one deliberate attempt; `scripts/targeted_boundary.py` is hard-coded to default allocator and one 114,688 success followed by one 131,072 attempt. That original coordinator cannot implement this sequence. The separate draft runner in `scripts/bisect_boundary.py` and `src/kv_cache_scaling/bisect_worker.py` now enforces the bisection order, a new per-experiment six-dispatch-per-allocator budget, worker-side receipt authorization, and no automatic retries. Those files and the generated notebook must be frozen into a reviewed prediction and a later complete receipt before any GPU work. This amendment draft is not execution authority; do not bypass the old ledger or reuse the old receipt.
