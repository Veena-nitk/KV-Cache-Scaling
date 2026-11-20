# Prospective paired-allocator bisection amendment review

Codex reviewed this package on behalf of Person 1 under the user's explicit standing delegation. This is one contributor review, not independent human review. The allocation remains anonymous Person 1 55% / Person 2 45%.

The original numerical commitment (`7baff5121ed643f27ba06959065855a8913f8e8c`) is preserved: **111,232 tokens**, with **76,024–131,072** assumption range for **15,527,116,800 usable bytes**. Prior measurements (7 L=512 acceptance, 80-worker primary/diagnostic/held-out cohort, 2 replication workers, and 2026-10-21 single OOM confirmation) are disclosed and preserved. This amendment specifies a separate bisection cohort to address TA Request 1.

## Workload & Invariants

- Model: Qwen/Qwen2.5-1.5B (`8faed761d45a263340a0528343f099c05c9a4323`), FP16, batch size 1, DynamicCache, logits_to_keep=1, forced efficient SDPA.
- Runtime: Python 3.11, PyTorch 2.6.0+cu124, CUDA 12.4, Transformers 4.51.3 on Tesla T4.
- GPU Session Invariant: Both allocator arms (`default` and `expandable`) must execute within the **same Colab session on the exact same physical GPU UUID**.
- Isolation: Every probe runs in a fresh, isolated child process (`processes.launch_worker`), with synchronized peak counter reset before forward pass.
- Idle check: GPU memory must return to idle baseline after every probe before advancing.
- Checkpointing: Full per-probe checkpoint capsule exported, member-hashed, and fsynced outside Colab before the next dispatch. Drive remains disconnected.

## Protocol & Stopping Rule

For each allocator (`default`, then `expandable`):
1. **Anchor check**: 3 independent repeats at L=114,688 (`attempt_kind='safe'`). All 3 must succeed. Any error or OOM halts the arm immediately.
2. **Confirmation check**: 1 probe at L=131,072 (`attempt_kind='deliberate'`). Must return typed `cuda_oom` to confirm the registered upper boundary.
3. **Midpoint bisection**: Bisection with midpoints rounded to multiples of 512 tokens. Continue until `first_cuda_oom - last_success <= 512`.
4. **Result format**: `(last_success, first_cuda_oom)` where both are multiples of 512 tokens and gap <= 512 tokens.

## Deliberate OOM Cap

- **Maximum 6 deliberate OOM outcomes per allocator** (1 upper-bound confirmation + at most 5 bisection steps).
- **Maximum 12 deliberate OOM outcomes total across both allocators**.
- The coordinator halts if an arm exceeds 6 OOMs or if gap <= 512 is reached.
- Accidental OOM during an anchor or non-OOM process failure halts the arm immediately without retry.

## Reporting

- Report `(last_success, first_cuda_oom)` for `default`.
- Report `(last_success, first_cuda_oom)` for `expandable`.
- Report the expandable delta: tokens and percentage.
- Keep this cohort separate from the original fit.
