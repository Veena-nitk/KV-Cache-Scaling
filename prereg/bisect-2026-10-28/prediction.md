# Prospective bisection amendment: numerical commitment preserved

The original advance central estimate is **111,232 tokens**, with assumption range **76,024–131,072**, at B=1 and usable budget **15,527,116,800 bytes**. The equation, fixed-length predictions, and parameter derivation are unchanged from `7baff5121ed643f27ba06959065855a8913f8e8c`. This amendment is prospective for the paired allocator boundary bisection, not a replacement advance prediction. Prior observations are disclosed.

## Objective

Bisect the boundary between **114,688 tokens** (known success) and **131,072 tokens** (configuration cap) down to **≤ 512 tokens**, separately for:
1. `default` native PyTorch allocator
2. `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` allocator

Both allocator arms must be measured within the **same Colab session on the exact same physical GPU UUID**.

## Sequence per Allocator

1. **Anchor check**: 3 independent fresh-worker repeats at L=114,688 (`attempt_kind='safe'`). All 3 must succeed without error or non-idle GPU return.
2. **Confirmation check**: 1 probe at L=131,072 (`attempt_kind='deliberate'`). Must return typed `cuda_oom` to confirm the upper boundary.
3. **Midpoint bisection**: Binary search between `last_success` and `first_cuda_oom` using integer midpoints rounded to multiples of 512 tokens, until `first_cuda_oom - last_success <= 512`.
4. **Result per allocator**: `(last_success, first_cuda_oom)`, where both values are exact multiples of 512 tokens and the gap is ≤ 512 tokens.
5. **Expandable comparison**: Report the expandable delta in tokens and percentage:
   $$\Delta_{\text{tokens}} = \text{last\_success}_{\text{expandable}} - \text{last\_success}_{\text{default}}$$
   $$\Delta_{\%} = \frac{\Delta_{\text{tokens}}}{\text{last\_success}_{\text{default}}} \times 100$$

## Deliberate OOM Budget

Starting from interval [114,688, 131,072] (width 16,384 tokens), at most 5 midpoint probes are required to reduce the gap to ≤ 512 tokens. Including the initial upper-bound confirmation at 131,072, the planned maximum is:
- **Maximum 6 typed OOM outcomes for `default` allocator**
- **Maximum 6 typed OOM outcomes for `expandable` allocator**
- **Maximum 12 typed OOM outcomes total across both allocators**

Any unexpected OOM during safe anchor repeats counts against the budget and halts the arm immediately. No probe may be launched that exceeds this cap.

## Harness & Durability

- Every probe executes in a fresh child process with isolated environment and synchronized memory peak counters (`reset_peak_memory_stats(0)`).
- Peak allocated and reserved bytes recorded upon completion.
- GPU returns to idle baseline verified after every probe.
- Full per-probe checkpoint capsule exported, hashed, and fsynced outside Colab before next dispatch.
- Google Drive remains disconnected.
