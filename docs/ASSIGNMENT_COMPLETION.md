# Problem Statement 2 completion record

The submitted workload is one cold FP16 forward at batch size 1 on Qwen2.5-1.5B, using a fresh DynamicCache and forced efficient SDPA. The primary memory sweep and the separate same-allocation boundary confirmation use the original one-deliberate-OOM budget. The verifier accepts the preserved delivery with the measurement limitations below.

## Deliverables

| Deliverable | Evidence |
| --- | --- |
| Symbol-defined memory derivation | [Derivation](DERIVATION.md) and [pinned model config](../configs/models/qwen2.5-1.5b.json). At FP16, 12 query heads and two KV heads give 28,672 cache bytes per token at batch size 1. Using query heads would give six times that slope. |
| Timestamped advance prediction | [Original commit](https://github.com/DevanshSharma351/KV-Cache-Scaling/commit/7baff5121ed643f27ba06959065855a8913f8e8c), 18 October 2026 at 19:54:52 IST. The central prediction was 111,232 tokens. The [frozen prediction](../prereg/formal-2026-10-18-t4/prediction.md) includes the five required memory forecasts. |
| Checkpointing sweep harness | [Coordinator and recovery contract](COORDINATOR_CONTRACT.md), [measurement code](../src/kv_cache_scaling/probe.py), [core sweep capsule](../artifacts/experiments/2026-10-19-t4-final/gpu-checkpoint-080.zip) and [one-OOM confirmation](OOM_BOUNDARY.md). The core capsule contains 80 successful workers, including 52 primary probes. Across the saved allocations there are 84 successes and one deliberate forward CUDA OOM. |
| Memory curves with analytical overlays | [Allocated peak versus length](../artifacts/figures/project-results/01_peak_vs_length.pdf), [KV growth](../artifacts/figures/project-results/02_cache_scaling.pdf), [allocator slack](../artifacts/figures/project-results/03_allocator_slack.pdf) and [default boundary](../artifacts/figures/project-results/06_observed_oom_boundary.pdf). |
| Signed discrepancy in bytes and MiB | [Fixed-length component table](DISCREPANCY_114688.csv), [measurement interpretation](DISCREPANCY_114688.md) and [committed prediction error interval](PREDICTION_ERROR_AT_CURRENT_BRACKET.md). |
| Cross-model and cross-GPU estimates with a published comparison | [Four forecasts and published Qwen2.5 KV-block comparisons](../data/processed/project-results/transfer/transfer.md). The published comparisons validate the KV term. The transferred single-prefill context thresholds remain forecasts. |

## Memory components at 114,688 tokens

The post-load allocated snapshot is measured before the forward. Retained KV is inspected from tensors. The remainder is peak allocated minus post-load allocated minus retained KV. This residual combines transient peaks, outputs and other memory outside the two measured components. Component lifetimes can differ at the peak.

| Allocator | Post-load allocated (B) | Retained KV (B) | Forward peak allocated (B) | Remaining peak (B) | Share |
| --- | ---: | ---: | ---: | ---: | ---: |
| Default | 3,088,346,624 | 3,288,334,336 | 14,020,668,928 | 7,643,987,968 | 54.52% |
| Expandable | 3,087,429,120 | 3,288,334,336 | 14,019,751,424 | 7,643,987,968 | 54.52% |

The default signed fixed-length discrepancy was −1,339,293,184 bytes (−1,277.25 MiB). Expandable gave −1,340,210,688 bytes (−1,278.125 MiB). In both rows, the largest adjustment was −1,350,870,016 bytes (−1,288.29 MiB) relative to the assumed transient envelope. The physical split of that adjustment remains unaccounted. The controlled cuBLAS workspace inference contributes +8.125 MiB. The component sums close arithmetically. [Byte ledger](DISCREPANCY_114688.csv)

## Paired allocator comparison within the one-OOM budget

Each row gives the median of simultaneous reserved minus allocated snapshots immediately after the synchronized forward, in MiB. The default arm has three repeats per length. The expandable arm has three except at 114,688, where it has one. [Raw snapshots](../data/processed/project-results/snapshots.csv)

| Tokens | Default allocator slack (MiB) | Expandable allocator slack (MiB) |
| ---: | ---: | ---: |
| 512 | 208.30 | 51.18 |
| 1,024 | 264.29 | 73.17 |
| 2,048 | 356.28 | 155.15 |
| 4,096 | 456.25 | 301.12 |
| 8,192 | 866.18 | 567.06 |
| 16,384 | 1,530.06 | 1,142.93 |
| 32,768 | 2,569.81 | 2,254.68 |
| 65,536 | 4,953.31 | 4,500.18 |
| 114,688 | 8,518.56 | 7,875.43 |

At the shared 114,688-token success, expandable segments reduced simultaneous post-forward slack by 643.125 MiB and peak reserved memory by 644 MiB. The difference between the largest observed successes was **0 tokens**. The shared conservative admission bracket was 65,536–65,792 tokens per allocator, yielding a conservative-policy delta interval of **−256 to +256 tokens**. The fitted peak-allocation crossings differ by **+4 tokens** under the original budget. These three numbers describe observed successes, the safe admission rule and a model fit respectively. **The hardware-limit gain is unmeasured.** [Paired results](../data/processed/project-results/allocator_boundary_deltas.csv) · [Admission brackets](../data/processed/project-results/boundaries.csv) · [Fitted crossings](PREDICTION_ERROR_AT_CURRENT_BRACKET.md)

## Prediction error at the hardware boundary

The registered confirmation used the single deliberate OOM. Default fit at 114,688 and raised CUDA OOM at 131,072 on that allocation. Assuming monotone feasibility, the largest fitting integer is in **[114,688, 131,071]** and the committed prediction error is **−15.14% to −3.01%**. The requested 32.00 GiB allocation agrees with the installed Transformers source for a 34,360,000,512-byte mask at the upper endpoint. The failed forward ends before completed-forward measurements exist at that length. The causal shares of the threshold error across activation, fragmentation and framework overhead are therefore **unaccounted**. [Boundary capsule and source attribution](OOM_BOUNDARY.md)

The draft paired multi-OOM extension exceeds the assignment's deliberate-failure budget. Its submitted logs failed the evidence audit and do not support the claimed capacity gain. [Submitted log audit](BISECTION_DATA_AUDIT.md)
