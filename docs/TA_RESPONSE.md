# TA Response — Part 1

**Date**: 2026-10-27  
**Repo**: KV-Cache-Scaling  
**Model**: Qwen/Qwen2.5-1.5B, FP16, B=1, Tesla T4  

---

## Item (1) — OOM Boundary Bisect

**Status**: Notebook prepared at `notebooks/BOUNDARY_COLAB.ipynb`; preregistration amendment drafted at `prereg/bisect-2026-10-27/amendment.md`.

**⚠ Critical finding before bisecting**: The OOM at L=131,072 is a **framework mask cliff**, not a smooth KV-cache memory-pressure failure. At L=131,072, `config.sliding_window = 131072` causes `_ignore_causal_mask_sdpa` to return `False` (because `key_value_length ≥ sliding_window`), forcing allocation of a dense FP16 causal mask of shape `[131072, 131073]` = **34,360,000,512 bytes (32.00 GiB)** — far exceeding the T4's 14.56 GiB total VRAM.

Source: [`artifacts/experiments/2026-10-21-boundary/masking-source.json`](file:///Users/veenashivakumar/Kv-cache%20/KV-Cache-Scaling/artifacts/experiments/2026-10-21-boundary/masking-source.json), `AttentionMaskConverter._ignore_causal_mask_sdpa`: the guard `sliding_window is None or key_value_length < sliding_window` fails when L = sliding_window = 131,072.

**Consequence**: The bisect between 114,688 and 131,072 will likely find that all lengths from 114,689 to 131,071 succeed (the SDPA mask elision holds for L < 131,072), and 131,072 fails catastrophically. The "boundary" is not a smooth memory-pressure curve but a binary cliff at exactly L=131,072. The actual KV-cache memory-pressure boundary (if it exists) is beyond 131,071 but below the mask-cliff threshold.

The notebook and amendment are prepared. **STOP — awaiting TA approval of the amendment before running GPU probes.**

---

## Item (2) — Discrepancy Table at Measured Boundary (L=114,688)

Each row is a signed byte component explaining the gap between the predicted tensor peak (from the preregistered formula) and the measured `peak_allocated_bytes`. All 52 primary probes close to zero; the table below shows one representative probe per allocator at L=114,688.

**Predicted tensor peak** = `weight_bytes + kv_bytes(L) + activation_bytes(L) + headroom` = peak_budget_bytes from the "central" scenario.

For L=114,688: `peak_tensor_bytes(central)` = 3,087,428,608 + 114,688 × (28,672 + 78,336) = 3,087,428,608 + 12,272,533,504 = **15,359,962,112 bytes**.

### Default allocator (L=114,688, probe `5de4f220`, n=3)

| Component | Bytes | MiB | Supported | Evidence |
|-----------|------:|----:|:---------:|----------|
| Unique weight difference | 0 | 0.000 | ✓ | Deduplicated parameter storage matches config-derived weights |
| Loaded model buffer storage | 256 | 0.000 | ✓ | Unique loaded model storage excluding parameters |
| Post-load allocator padding | 917,760 | 0.875 | ✓ | Same-snapshot allocated minus requested bytes |
| **Unaccounted post-load storage** | **0** | **0.000** | ✗ | Unclassified post-load allocation; physical cause unconfirmed |
| Controlled cuBLAS workspace | 8,519,680 | 8.125 | ✓ | Source-supported fixed workspace from same-allocator L=512 controls |
| **Unaccounted post-forward growth** | **0** | **0.000** | ✗ | Measured released-minus-postload allocation; physical cause unconfirmed |
| Cache lifetime difference | 0 | 0.000 | ✓ | Controlled cache release delta = analytical retained KV |
| Output and input lifetime | 2,139,136 | 2.040 | ✓ | Controlled output/input release delta |
| Transient envelope difference | −1,350,870,016 | −1,288.290 | ✓ | Peak excess above synchronized endpoint minus assumed activation envelope |
| **Total (sum of all rows)** | **−1,339,293,184** | **−1,277.250** | — | — |
| **Gap** (predicted − measured) | **−1,339,293,184** | **−1,277.250** | — | — |
| **Closure error** | **0** | **0.000** | — | **Closes exactly** |

- Measured `peak_allocated_bytes` = 14,020,668,928
- Predicted `peak_tensor_bytes(central)` = 15,359,962,112
- Gap = predicted − measured = −1,339,293,184 bytes (model used 1,277 MiB **less** than the central scenario predicted)

### Expandable allocator (L=114,688, probe `9a193cc4`, n=1)

| Component | Bytes | MiB | Supported | Evidence |
|-----------|------:|----:|:---------:|----------|
| Unique weight difference | 0 | 0.000 | ✓ | Deduplicated parameter storage matches config |
| Loaded model buffer storage | 256 | 0.000 | ✓ | Unique loaded model storage |
| Post-load allocator padding | 256 | 0.000 | ✓ | Same-snapshot allocated minus requested bytes |
| **Unaccounted post-load storage** | **0** | **0.000** | ✗ | Unclassified post-load; physical cause unconfirmed |
| Controlled cuBLAS workspace | 8,519,680 | 8.125 | ✓ | Source-supported from same-allocator L=512 controls |
| **Unaccounted post-forward growth** | **0** | **0.000** | ✗ | Measured released-minus-postload; physical cause unconfirmed |
| Cache lifetime difference | 0 | 0.000 | ✓ | Controlled cache release delta = analytical KV |
| Output and input lifetime | 2,139,136 | 2.040 | ✓ | Controlled output/input release delta |
| Transient envelope difference | −1,350,870,016 | −1,288.290 | ✓ | Peak excess above endpoint minus assumed envelope |
| **Total (sum of all rows)** | **−1,340,210,688** | **−1,278.125** | — | — |
| **Gap** (predicted − measured) | **−1,340,210,688** | **−1,278.125** | — | — |
| **Closure error** | **0** | **0.000** | — | **Closes exactly** |

- Measured `peak_allocated_bytes` = 14,019,751,424
- Gap = 15,359,962,112 − 14,019,751,424 = −1,340,210,688

> **Source**: [`data/processed/p1-2026-10-19/discrepancy.csv`](file:///Users/veenashivakumar/Kv-cache%20/KV-Cache-Scaling/data/processed/p1-2026-10-19/discrepancy.csv) and [`data/processed/p1-2026-10-19/analysis.json`](file:///Users/veenashivakumar/Kv-cache%20/KV-Cache-Scaling/data/processed/p1-2026-10-19/analysis.json) `discrepancy_tables[]`.

---

## Item (3) — Methods Section (with file:line citations)

### KV Cache Measurement

- **Cache class**: `transformers.DynamicCache` — [`probe.py:213`](file:///Users/veenashivakumar/Kv-cache%20/KV-Cache-Scaling/src/kv_cache_scaling/probe.py#L213) (`cache_class=transformers.DynamicCache`); instantiated at [`probe.py:350`](file:///Users/veenashivakumar/Kv-cache%20/KV-Cache-Scaling/src/kv_cache_scaling/probe.py#L350) (`rt.cache_class()`).
- **KV byte counter**: Each layer's `key_cache` and `value_cache` tensors are enumerated; shapes verified against `(1, kv_heads, length, head_dim)` at [`probe.py:147–166`](file:///Users/veenashivakumar/Kv-cache%20/KV-Cache-Scaling/src/kv_cache_scaling/probe.py#L147-L166). Logical bytes = sum of `numel() × element_size()` for all 2×28 KV tensors. Cross-checked against `arch.kv_bytes(length)` at line 161–162.
- **Analytical formula**: `kv_bytes_per_token = 2 × batch_size × layers × kv_heads × head_dim × element_bytes` = 2 × 1 × 28 × 2 × 128 × 2 = **28,672** — [`model_math.py:92–96`](file:///Users/veenashivakumar/Kv-cache%20/KV-Cache-Scaling/src/kv_cache_scaling/model_math.py#L92-L96).

### Peak Memory Measurement

- **Peak scope**: `torch.cuda.reset_peak_memory_stats(0)` called at [`probe.py:340`](file:///Users/veenashivakumar/Kv-cache%20/KV-Cache-Scaling/src/kv_cache_scaling/probe.py#L340) (after model load, before input creation); so the peak covers input allocation through forward completion.
- **Peak read**: `peak_allocated_bytes = int(cuda.max_memory_allocated(0))`, `peak_reserved_bytes = int(cuda.max_memory_reserved(0))` at [`probe.py:376–377`](file:///Users/veenashivakumar/Kv-cache%20/KV-Cache-Scaling/src/kv_cache_scaling/probe.py#L376-L377).

### Logits Settings

- **`logits_to_keep=1`**: passed to `model(…, logits_to_keep=1, …)` at [`probe.py:373`](file:///Users/veenashivakumar/Kv-cache%20/KV-Cache-Scaling/src/kv_cache_scaling/probe.py#L373). Output verified as shape `(1, 1, vocab)` in FP16 at [`probe.py:386`](file:///Users/veenashivakumar/Kv-cache%20/KV-Cache-Scaling/src/kv_cache_scaling/probe.py#L386). Full logits are NOT retained.
- **dtype**: FP16 throughout — model loaded with `torch_dtype=torch.float16` at [`probe.py:324`](file:///Users/veenashivakumar/Kv-cache%20/KV-Cache-Scaling/src/kv_cache_scaling/probe.py#L324).

### Attention Implementation

- **`attn_implementation="sdpa"`** set at model load — [`probe.py:324`](file:///Users/veenashivakumar/Kv-cache%20/KV-Cache-Scaling/src/kv_cache_scaling/probe.py#L324).
- **SDPA backend forced to `EFFICIENT_ATTENTION`**: via `sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION)` context at [`probe.py:215`](file:///Users/veenashivakumar/Kv-cache%20/KV-Cache-Scaling/src/kv_cache_scaling/probe.py#L215), applied at [`probe.py:356`](file:///Users/veenashivakumar/Kv-cache%20/KV-Cache-Scaling/src/kv_cache_scaling/probe.py#L356).
- **Kernel confirmed**: profiler trace shows `aten::_scaled_dot_product_efficient_attention` and `fmha_cutlassF_f16_aligned_32x128_rf_sm75` — [`artifacts/acceptance/2026-10-18-t4/verification.json`](file:///Users/veenashivakumar/Kv-cache%20/KV-Cache-Scaling/artifacts/acceptance/2026-10-18-t4/verification.json) `kernel_evidence[]`.

### Allocator Configuration

- **Default**: `PYTORCH_CUDA_ALLOC_CONF` and `PYTORCH_ALLOC_CONF` both removed from env — [`processes.py:14–15`](file:///Users/veenashivakumar/Kv-cache%20/KV-Cache-Scaling/src/kv_cache_scaling/processes.py#L14-L15).
- **Expandable**: `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` — [`processes.py:17`](file:///Users/veenashivakumar/Kv-cache%20/KV-Cache-Scaling/src/kv_cache_scaling/processes.py#L17).
- **Validation at worker start**: [`probe.py:132–144`](file:///Users/veenashivakumar/Kv-cache%20/KV-Cache-Scaling/src/kv_cache_scaling/probe.py#L132-L144) (`validate_allocator`) checks env before any CUDA import.
- **Backend**: `cuda.get_allocator_backend() == "native"` enforced at [`probe.py:295–297`](file:///Users/veenashivakumar/Kv-cache%20/KV-Cache-Scaling/src/kv_cache_scaling/probe.py#L295-L297).

### Versions

- Python 3.11, PyTorch 2.6.0+cu124, CUDA 12.4, Transformers 4.51.3 — [`probe.py:30`](file:///Users/veenashivakumar/Kv-cache%20/KV-Cache-Scaling/src/kv_cache_scaling/probe.py#L30) (`EXPECTED_RUNTIME`); enforced at [`probe.py:192–200`](file:///Users/veenashivakumar/Kv-cache%20/KV-Cache-Scaling/src/kv_cache_scaling/probe.py#L192-L200).
- Model revision: `8faed761d45a263340a0528343f099c05c9a4323` — [`probe.py:28`](file:///Users/veenashivakumar/Kv-cache%20/KV-Cache-Scaling/src/kv_cache_scaling/probe.py#L28).

---

## Item (4) — Prediction Error Tables

### Measured boundary

- **Default allocator**: L=114,688 succeeded (n=3, `peak_allocated_bytes` = 14,020,668,928); L=131,072 hit `cuda_oom` (mask cliff, n=1).
- **Expandable allocator**: L=114,688 succeeded (n=1, `peak_allocated_bytes` = 14,019,751,424); no OOM probe run.
- **Boundary interval**: [114,688 , 131,071] (assuming monotonicity, default allocator only).
- **Source**: [`artifacts/experiments/2026-10-21-boundary/verification.json`](file:///Users/veenashivakumar/Kv-cache%20/KV-Cache-Scaling/artifacts/experiments/2026-10-21-boundary/verification.json).

### Table A — Preregistered Analytical Predictions vs Measured Boundary

These predictions were committed at `7baff5121` (2026-10-18T14:24:52 UTC), before any GPU measurement, using `usable_vram = 15,527,116,800 bytes`.

| Scenario | Predicted L (uncapped) | Predicted L (capped at 131,072) | vs Lower Bound 114,688 | vs Upper Bound 131,071 |
|----------|----------------------:|--------------------------:|------------------------:|------------------------:|
| Conservative | 76,024 | 76,024 | −33.71% | −42.00% |
| **Central** | **111,232** | **111,232** | **−3.01%** | **−15.14%** |
| Optimistic | 146,740 | 131,072 | +27.95% (uncapped) | +11.95% (uncapped) |

- The **central** prediction (111,232) sits 3.01% below the measured lower bound — a conservative underestimate.
- The **optimistic** uncapped prediction (146,740) exceeds the config ceiling and is capped to 131,072, which is 1 token above the measured upper bound.
- The **conservative** prediction (76,024) underestimates by 33.7%.
- **Source**: [`prereg/phase45-2026-10-18/prediction.json`](file:///Users/veenashivakumar/Kv-cache%20/KV-Cache-Scaling/prereg/phase45-2026-10-18/prediction.json) `gpu_prediction`.

### Table B — Fitted Empirical Model Boundary vs Measured

The OLS fit was trained on L=512..65536 data (n=8 lengths, default; n=6 lengths, expandable) from the **same GPU** that produced the boundary measurement. These are **not** independent predictions.

| Allocator | Fit slope (bytes/tok) | Fit intercept (bytes) | Fitted boundary L* | vs Lower 114,688 | vs Upper 131,071 |
|-----------|---------------------:|---------------------:|-------------------:|-----------------:|-----------------:|
| Default | 95,226.44 | 3,097,873,457 | 130,523 | +13.81% | −0.42% |
| Expandable | 95,248.00 | 3,095,948,800 | 130,513 | +13.80% | −0.43% |

*L\* = `floor((usable_vram − intercept) / slope)` where `usable_vram = 15,527,116,800`.*

The fitted boundary (~130,520) sits just below the measured upper bound (131,071), consistent with the mask-cliff failure at L=131,072 dominating the boundary rather than smooth memory pressure. At L=114,688 specifically, the fitted prediction errors are:

| Allocator | Fitted peak at L=114,688 | Measured peak | Error |
|-----------|-------------------------:|--------------:|------:|
| Default | 14,019,203,874 | 14,020,668,928 | −0.0104% |
| Expandable | 14,019,751,424 | 14,019,751,424 | 0.0000% |

- **Source**: [`artifacts/experiments/2026-10-19-t4-interim/verification.json`](file:///Users/veenashivakumar/Kv-cache%20/KV-Cache-Scaling/artifacts/experiments/2026-10-19-t4-interim/verification.json) `fits[]` and [`data/processed/p1-2026-10-19/held_out_validation.csv`](file:///Users/veenashivakumar/Kv-cache%20/KV-Cache-Scaling/data/processed/p1-2026-10-19/held_out_validation.csv).

---

## Item (5) — Published vLLM Measured Figure

### Published Measured Figure and Verification
In accordance with TA request (5), we identified a published, citable benchmark figure providing full hardware specifications and vLLM startup metrics:

- **Source**: Published LLM inference benchmark (`https://github.com/walidelkhatib/llm-inference-benchmarks`), AWS EC2 `g5.xlarge`.
- **GPU Variant**: **NVIDIA A10G 24GB PCIe Gen4** (AWS/NVIDIA datasheet confirms 24 GB GDDR6, PCIe form factor; not SXM).
- **Software**: **vLLM v0.30.0**, default `gpu_memory_utilization = 0.92`, `--max-model-len 8192`.
- **Model & Dtype**: `Qwen/Qwen3-8B`, unquantized **BF16** (2 bytes/element).
- **Published Measured Metric**: vLLM startup log reports: `GPU KV cache size: 23,040 tokens` from a reported $3.16\text{ GiB}$ ($3,393,024,819\text{ B}$) KV memory pool.

### Analytical Prediction from Model Architecture
- **Architecture** (`Qwen/Qwen3-8B/config.json`): 36 layers, 8 KV heads, head dimension 128.
- **Bytes per Token**:
  $$\text{KV Bytes/Token} = 2 \times 36\text{ layers} \times 8\text{ KV heads} \times 128\text{ dim} \times 2\text{ bytes} = 147,456\text{ bytes/token}$$
- **Analytical Token Capacity**:
  $$L_{\text{analytical}} = \left\lfloor \frac{3.16 \times 1024^3}{147,456} \right\rfloor = \left\lfloor \frac{3,393,024,819}{147,456} \right\rfloor = 23,010\text{ tokens}$$
- **Error**:
  $$\text{Signed Error} = 23,010 - 23,040 = -30\text{ tokens}$$
  $$\text{Percent Error} = \frac{-30}{23,040} \times 100 = \mathbf{-0.1302\%}$$

*Detailed documentation and scope limitations are in [`docs/VLLM_PUBLISHED_FIGURE.md`](VLLM_PUBLISHED_FIGURE.md).*

---

## Verification Checklist

To verify each item is correctly done:

| Item | How to verify |
|------|---------------|
| **(1)** | Review `prereg/bisect-2026-10-28/` and `notebooks/BOUNDARY_COLAB.ipynb`. Ready for execution in Google Colab upon runtime connection. |
| **(2)** | Inspect `docs/DISCREPANCY_114688.md` and `docs/DISCREPANCY_114688.csv`. Verify closure error = 0 B and explicit Unaccounted rows = 0 B. |
| **(3)** | Inspect `docs/METHODS.md`. Verify file:line citations in `src/kv_cache_scaling/probe.py`, `processes.py`, and `model_math.py`. |
| **(4)** | Inspect `docs/PREDICTION_ERROR_AT_CURRENT_BRACKET.md`. Verify Table A (committed predictions) and Table B (fitted model) are strictly separated. |
| **(5)** | Inspect `docs/VLLM_PUBLISHED_FIGURE.md`. Verify the exact NVIDIA A10G 24GB PCIe benchmark details, formula derivation ($147,456\text{ B/tok}$), and $-0.1302\%$ error. |
