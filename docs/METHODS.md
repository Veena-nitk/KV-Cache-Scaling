# Methods and measurement settings (Part A2)

This describes the repository's pinned primary probe, not a claim that every saved record came from an identical session. Line numbers refer to the committed source in this checkout.

## KV counter and cache

The primary cache is `transformers.DynamicCache`. `_import_runtime()` binds `cache_class=transformers.DynamicCache` and the efficient attention context in [`src/kv_cache_scaling/probe.py:192-215`](../src/kv_cache_scaling/probe.py#L192). The primary path instantiates that class at [`probe.py:344-350`](../src/kv_cache_scaling/probe.py#L344).

`inspect_cache()` checks 28 key and 28 value tensors, populated sequence length, expected shape `(1, num_key_value_heads, L, head_dim)`, FP16 dtype and `cuda:0`; it compares logical tensor bytes to `arch.kv_bytes(L)` at [`probe.py:147-166`](../src/kv_cache_scaling/probe.py#L147). Storage is separately deduplicated by `(device, data_ptr, storage bytes)` in [`src/kv_cache_scaling/telemetry.py:9-28`](../src/kv_cache_scaling/telemetry.py#L9). The result's unique and logical storage byte fields should not be conflated. For this model, the derivation is 2 × batch 1 × 28 layers × 2 KV heads × 128 head dimension × 2 FP16 bytes = **28,672 logical KV bytes/token** (`docs/DERIVATION.md`, config `configs/models/qwen2.5-1.5b.json`).

## Forward, logits, and attention

The pinned policy is one cold, batch-1 FP16 single prefill, empty DynamicCache, `logits_to_keep=1`, `output_attentions=False`, `output_hidden_states=False`, no warmup, `attn_implementation="sdpa"`, forced `SDPBackend.EFFICIENT_ATTENTION` (`probe.py:30-35`, `192-215`, `322-325`, `344-374`). The model is loaded with `torch_dtype=torch.float16` and evaluated at `probe.py:322-327`. Thus logits are reduced, not full sequence logits: output shape must equal `(1, 1, vocabulary)` and dtype FP16 at `probe.py:386-388`. No attention weights are returned. The kernel is required by the optional trace to include `_scaled_dot_product_efficient_attention` (`probe.py:378-383`); saved GPU acceptance evidence is in `artifacts/acceptance/2026-10-18-t4/verification.json`.

Peak counters reset after model load and before input, position, and cache allocation (`probe.py:339-342`). The forward peak is read after synchronize at `probe.py:372-377`; the window includes input allocation and first-forward transients. It is PyTorch tracked peak, not NVML device-wide peak. Snapshots and NVML sampling are defined in `telemetry.py:67-88` and `91-130`.

## Allocator and runtime

Default workers remove both allocator aliases; expandable workers set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` in [`src/kv_cache_scaling/processes.py:12-18`](../src/kv_cache_scaling/processes.py#L12). The child validates the environment before CUDA import and rejects conflicting aliases at [`probe.py:132-144`](../src/kv_cache_scaling/probe.py#L132). Only PyTorch's `native` backend is accepted (`probe.py:295-297`). An expandable unsupported warning invalidates that run (`probe.py:238-245`); backend `native` alone does not demonstrate that expandable segments took effect.

Expected runtime is Python 3.11, PyTorch 2.6.0, CUDA 12.4, Transformers 4.51.3 (`probe.py:27-35`, checks at `probe.py:192-200`). Runtime results record installed package inventory at `probe.py:281-288`; the GPU requirements pin Transformers 4.51.3 and Hugging Face Hub 0.30.2 through nvidia-ml-py 12.570.86 (`requirements-gpu.txt:3-11`). PyTorch 2.6.0 is installed separately from the CUDA 12.4 wheel in the Colab notebook; `probe.py:30,192-200` checks its version and CUDA build. These are repository pins/expected versions; per-run package inventory is the evidence for what a particular worker actually loaded.
