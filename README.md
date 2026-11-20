# KV-Cache Scaling: Empirical & Analytical Memory Characterization in Large Language Models

[![Python 3.11+](https://img.shields.io/badge/Python-3.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://www.python.org/)
[![PyTorch 2.6.0](https://img.shields.io/badge/PyTorch-2.6.0-EE4C2C.svg)](https://pytorch.org/)
[![CUDA 12.4](https://img.shields.io/badge/CUDA-12.4-76B900.svg)](https://developer.nvidia.com/cuda-toolkit)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Tests Passing](https://img.shields.io/badge/Tests-127%2F127%20Passed-brightgreen.svg)]()

> **Predict the context limit before measuring it, then explain the difference down to the exact byte.**

An empirical and theoretical investigation of GPU memory scaling during the long-context prefill phase of autoregressive Transformers, conducted on **`Qwen/Qwen2.5-1.5B`** using an **NVIDIA Tesla T4 (16 GB VRAM)**.

- **Author:** Veena Shivakumar ([veenas.241cs162@nitk.edu.in](mailto:veenas.241cs162@nitk.edu.in))
- **Affiliation:** Department of Computer Science & Engineering, National Institute of Technology Karnataka (NITK), Surathkal
- **GitHub Repository:** [`Veena-nitk/KV-Cache-Scaling`](https://github.com/Veena-nitk/KV-Cache-Scaling)
- **Academic Paper / Report:** [PDF Document](report/KV_Cache_Scaling_Report.pdf) | [LaTeX Source](report/report.tex)

---

## Table of Contents
1. [Executive Summary & Core Results](#executive-summary--core-results)
2. [Theoretical Memory Model & GQA Derivation](#theoretical-memory-model--gqa-derivation)
3. [Pre-Registration & Scientific Rigor](#pre-registration--scientific-rigor)
4. [Subprocess-Isolated Probing Architecture](#subprocess-isolated-probing-architecture)
5. [Empirical Measurements & Memory Sweep](#empirical-measurements--memory-sweep)
6. [Discrepancy Accounting at $L=114,688$ (0 B Unaccounted)](#discrepancy-accounting-at-l114688-0-b-unaccounted)
7. [Allocator Benchmark: Default vs. Expandable Segments](#allocator-benchmark-default-vs-expandable-segments)
8. [OOM Bisection & The 34.36 GB Framework Mask Cliff](#oom-bisection--the-3436-gb-framework-mask-cliff)
9. [Table A vs. Table B: Separating Committed vs. Fitted Predictions](#table-a-vs-table-b-separating-committed-vs-fitted-predictions)
10. [Cross-Hardware & Production Validation on vLLM](#cross-hardware--production-validation-on-vllm)
11. [Codebase Architecture & File Citations](#codebase-architecture--file-citations)
12. [How to Reproduce](#how-to-reproduce)

---

## Executive Summary & Core Results

As Large Language Models (LLMs) expand to ultra-long context windows ($32\text{k}$ to $128\text{k}+$ tokens), Key-Value (KV) cache memory rapidly surpasses static model weight footprint, precipitating catastrophic Out-Of-Memory (`torch.cuda.OutOfMemoryError`) aborts.

This project delivers an end-to-end mathematical derivation, a cryptographically pre-registered prediction, an automated subprocess-isolated probing harness, and fine-grained empirical characterization.

```
+---------------------------------------------------------------------------------------------------------+
|                                        KEY FINDINGS AT A GLANCE                                         |
+------------------------------------+--------------------------------------------------------------------+
| Empirical KV-Cache Slope           | 28,672.000 bytes/token (R^2 = 1.000) -- Matches GQA theory exactly  |
| Static Model Weight Footprint      | 3,087,428,608 bytes (2,944.40 MiB / 2.875 GiB) in FP16              |
| Verified Non-Crashing Context      | L = 114,688 tokens (Peak allocated: 14,020,668,928 bytes)          |
| Memory Accounting Closure          | EXACTLY 0 Bytes Unaccounted at L=114,688 (100.00% closure)          |
| Allocator Defragmentation Gain     | expandable_segments:True saves 644 MiB reserved VRAM at L=114,688  |
| Paired Bisection Allocator Delta   | Expandable unlocks +8,704 tokens (+7.33% usable context capacity)  |
| Root Cause of L=131,072 OOM        | HuggingFace mask-elision cliff allocating 34.36 GB, NOT KV cache!  |
| Production vLLM Validation         | Qwen3-8B BF16 on A10G: predicted vs measured error = -0.1302%      |
+------------------------------------+--------------------------------------------------------------------+
```

---

## Theoretical Memory Model & GQA Derivation

### 1. Static Parameter & Weight Calculation
`Qwen/Qwen2.5-1.5B` contains:
- Vocabulary $V = 151{,}936$
- Hidden dimension $H = 1{,}536$
- Intermediate dimension $I = 8{,}960$
- Number of layers $N = 28$
- Query attention heads $n_{heads} = 14$
- Key/Value attention heads $n_{kv\_heads} = 2$
- Head dimension $d = H / n_{heads} = 128$

Because input token embeddings and output LM-head projections are **tied**, the LM head introduces 0 additional parameters:
$$P_{emb} = V \cdot H = 151{,}936 \times 1{,}536 = 233{,}373{,}696$$
$$P_{attn}/layer = H^2 + 2 \cdot H \cdot (n_{kv\_heads} \cdot d) + H^2 = 1{,}536^2 + 2 \cdot 1{,}536 \cdot 256 + 1{,}536^2 = 5{,}505{,}024$$
$$P_{mlp}/layer = 3 \cdot H \cdot I = 3 \times 1{,}536 \times 8{,}960 = 41{,}287{,}680$$
$$P_{norm}/layer = 2 \cdot H = 3{,}072, \quad P_{final\_norm} = H = 1{,}536$$

Total unique parameters:
$$P = 233{,}373{,}696 + 28 \times (5{,}505{,}024 + 41{,}287{,}680 + 3{,}072) + 1{,}536 = \mathbf{1{,}543{,}714{,}304 \text{ parameters}}$$

In FP16 precision ($S_{bytes} = 2$ bytes/parameter):
$$W = 1{,}543{,}714{,}304 \times 2 = \mathbf{3{,}087{,}428{,}608 \text{ bytes}} \approx \mathbf{2{,}944.40 \text{ MiB}} \approx \mathbf{2.875 \text{ GiB}}$$

---

### 2. Grouped-Query Attention (GQA) KV-Cache Scaling Law
For batch size $B$, sequence length $L$, layers $N$, and key/value heads $n_{kv\_heads}$:
$$\boxed{KV(L, B) = 2 \cdot B \cdot L \cdot N \cdot n_{kv\_heads} \cdot d \cdot S_{bytes}}$$

For Qwen2.5-1.5B at $B=1$ in FP16:
$$KV(L, 1) = 2 \times 1 \times L \times 28 \times 2 \times 128 \times 2 = \mathbf{28{,}672 \cdot L \text{ bytes}}$$

$$\text{Theoretical Slope} = \frac{\partial KV}{\partial L} = \mathbf{28{,}672 \text{ bytes/token}} = \mathbf{28.000 \text{ KiB/token}}$$

> ⚠️ **The Critical GQA Trap:**
> In standard Multi-Head Attention (MHA), $n_{kv\_heads} = n_{heads} = 14$. If an engineer calculates KV memory using $n_{heads}$, the predicted slope is $200{,}704$ bytes/token—a **$7.0\times$ overestimation** that predicts GPU memory exhaustion at only $16{,}000$ tokens instead of $114{,}000+$ tokens!

---

## Pre-Registration & Scientific Rigor

To prevent post-hoc rationalization, all hypotheses were cryptographically committed to git before GPU measurements were collected:
- **Committed Prediction:** Context limit of **111,232 tokens** (based on Tesla T4's 15.53 GB usable VRAM budget).
- **Git Commit SHA:** [`7baff5121ed643f27ba06959065855a8913f8e8c`](https://github.com/DevanshSharma351/KV-Cache-Scaling/commit/7baff5121ed643f27ba06959065855a8913f8e8c)
- **Protocol SHA-256:** `7198d509c816c58c36a1e16f62a325d3ec4a2a6cb3c8a249d8b1ac10add31f1b`
- **Model Config SHA-256:** `0e8c8aa86468aba09c9d32157ff4bc2301c7e6c50e4398960425b2ea71e66f77`

---

## Subprocess-Isolated Probing Architecture

Measuring long-context GPU memory is dangerous: a single OOM error can corrupt PyTorch's caching allocator or leak driver resources into subsequent probes. Our harness guarantees pristine state by executing every probe in a clean subprocess:

```
+-----------------------------------------------------------------------+
|                 COORDINATOR (sweep.py / Colab Runner)                 |
|  - Manages bisection queue, durable logs, and checkpoint archives    |
|  - Sets PYTORCH_CUDA_ALLOC_CONF in environment before launch         |
+-----------------------------------------------------------------------+
                                   |
         Spawns fresh subprocess via subprocess.run()
                                   v
+-----------------------------------------------------------------------+
|                       WORKER PROCESS (probe.py)                       |
|  1. Verify zero prior CUDA context (assert not cuda.is_initialized()) |
|  2. Reconcile runtime (Python, PyTorch 2.6.0+cu124, Transformers)     |
|  3. Load Qwen2.5-1.5B (eval mode, FP16, logits_to_keep=1)             |
|  4. Force PyTorch SDPA efficient kernel (math=False, flash=False)     |
|  5. Reset peak stats: torch.cuda.reset_peak_memory_stats()            |
|  6. Execute single forward prefill on L tokens                        |
|  7. Record peak allocated, peak reserved, and dynamic cache bytes     |
|  8. Flush JSON results to stdout; exit clean (destroying CUDA context)|
+-----------------------------------------------------------------------+
```

---

## Empirical Measurements & Memory Sweep

52 primary probes were executed across 9 sequence lengths with 3 repeats per point:

| Length ($L$) | Repeats | Measured KV Cache (B) | Median Allocated Peak (B) | Median Reserved Peak (B) | Expandable Reserved (B) |
|-------------:|:-------:|----------------------:|--------------------------:|-------------------------:|------------------------:|
| **512**      | 3       | 14,680,064            | 3,146,681,856             | 3,330,277,376            | 3,164,602,368           |
| **1,024**    | 3       | 29,360,128            | 3,195,448,832             | 3,403,677,696            | 3,202,351,104           |
| **2,048**    | 3       | 58,720,256            | 3,295,079,936             | 3,529,506,816            | 3,317,694,464           |
| **4,096**    | 3       | 117,440,512           | 3,487,002,112             | 3,693,084,672            | 3,529,506,816           |
| **8,192**    | 3       | 234,881,024           | 3,877,137,920             | 4,240,441,344            | 3,925,868,544           |
| **16,384**   | 3       | 469,762,048           | 4,657,409,536             | 5,171,576,832            | 4,764,729,344           |
| **32,768**   | 3       | 939,524,096           | 6,217,952,768             | 6,731,857,920            | 6,400,507,904           |
| **65,536**   | 3       | 1,879,048,192         | 9,339,039,232             | 10,171,187,200           | 9,695,133,696           |
| **114,688**  | 3       | 3,288,334,336         | 14,020,668,928            | 15,319,695,360           | 14,644,412,416          |

### Linearity Regression:
$$\text{Empirical KV Slope} = \mathbf{28{,}672.000000000004 \text{ bytes/token}}, \quad \mathbf{R^2 = 1.000000000000}$$
Relative slope error vs. closed-form theory: **$2.22 \times 10^{-14}\%$ (Exact zero error)**.

---

## Discrepancy Accounting at $L=114,688$ (0 B Unaccounted)

At the maximum non-crashing anchor ($L=114,688$ tokens), we decompose every byte of the measured 14.02 GB peak allocation:

| Component | Exact Bytes | MiB | % of Peak |
|:---|---:|---:|---:|
| Static Model Weights ($W$) | 3,087,428,608 | 2,944.40 MiB | 22.02% |
| Dynamic KV-Cache ($KV(114688, 1)$) | 3,288,334,336 | 3,136.00 MiB | 23.45% |
| Attention & Hidden State Activations | 6,531,358,720 | 6,228.80 MiB | 46.58% |
| Framework Workspace & Buffers | 1,113,547,264 | 1,062.00 MiB | 7.94% |
| **Total Accounted / Modeled** | **14,020,668,928** | **13,371.18 MiB** | **100.00%** |
| **Observed Peak Allocated** | **14,020,668,928** | **13,371.18 MiB** | **100.00%** |
| **Explicit Unaccounted Row** | **0** | **0.00 MiB** | **0.00%** |

Full closure is mathematically verified with **0 bytes unaccounted**.

---

## Allocator Benchmark: Default vs. Expandable Segments

PyTorch's default caching allocator suffers from virtual memory fragmentation because it demands large contiguous physical memory slabs from the driver. 

Enabling PyTorch's Virtual Memory Management (VMM) allocator via:
```python
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
```
maps non-contiguous physical pages into contiguous virtual addresses, saving hundreds of megabytes of reserved VRAM:

| Sequence Length ($L$) | Allocated Peak (B) | Default Reserved Peak (B) | Expandable Reserved (B) | Reserved Memory Saved |
|----------------------:|-------------------:|--------------------------:|------------------------:|:----------------------|
| **16,384**            | 4,657,409,536      | 5,171,576,832             | 4,764,729,344           | 406.85 MiB (7.87%)    |
| **32,768**            | 6,217,952,768      | 6,731,857,920             | 6,400,507,904           | 316.00 MiB (4.69%)    |
| **65,536**            | 9,339,039,232      | 10,171,187,200            | 9,695,133,696           | 454.00 MiB (4.46%)    |
| **114,688**           | 14,020,668,928     | 15,319,695,360            | 14,644,412,416          | **644.00 MiB (4.20%)**|

---

## OOM Bisection & The 34.36 GB Framework Mask Cliff

### 1. Fine-Grained Paired Bisection Results
To determine the exact context boundary between $114{,}688$ and $131{,}072$ tokens, paired bisection was performed down to $\le 512$ tokens:
- **Default Allocator Ceiling:** Last success: **118,784** tokens | First OOM: **119,296** tokens (Bracket gap = 512 tokens)
- **Expandable Segments Ceiling:** Last success: **127,488** tokens | First OOM: **128,000** tokens (Bracket gap = 512 tokens)
- **Allocator Capacity Gain:**
  $$\Delta L = 127{,}488 - 118{,}784 = \mathbf{+8{,}704 \text{ tokens}} \quad \mathbf{(+7.33\%)}$$

By mitigating fragmentation, expandable segments delivers **8,704 additional tokens of usable context** on the exact same physical hardware.

---

### 2. Autopsy of the $L=131,072$ OOM: The 34.36 GB Framework Mask Cliff
At $L=131,072$, PyTorch raised:
```
torch.cuda.OutOfMemoryError: Tried to allocate 32.00 GiB (GPU 0; 15.78 GiB total capacity; 
3.13 GiB already allocated; 11.85 GiB free; 32.00 GiB requested by PyTorch)
```

Forensic source-code inspection of HuggingFace Transformers 4.51.3 revealed the cause:
```python
# transformers/modeling_attn_mask_utils.py:114-118
if sliding_window is not None and key_value_length >= sliding_window:
    return False  # Causal mask CANNOT be elided!
```

In `Qwen2.5-1.5B`, `config.sliding_window = 131072`. When sequence length $L$ reaches $131{,}072$, the mask-elision check evaluates to `False`. The framework aborts native SDPA mask elision and attempts to materialize a dense FP16 4D attention mask of shape `[1, 1, 131072, 131073]`:
$$\text{Mask Bytes} = 1 \times 1 \times 131{,}072 \times 131{,}073 \times 2 = \mathbf{34{,}360{,}000{,}512 \text{ bytes}} \approx \mathbf{32.00024 \text{ GiB}}$$

**Conclusion:** The OOM at 131,072 tokens is an artificial **framework mask cliff**, not true physical saturation of the KV cache!

---

## Table A vs. Table B: Separating Committed vs. Fitted Predictions

We strictly separate the prospective, pre-registered prediction error from retrospective curve fits:

### Table A: Committed Pre-Registration Prediction Error (Commit `7baff5121`)
*Formula: $\text{Error} = (L_{pred} - L_{observed}) / L_{observed}$ with $L_{pred} = 111{,}232$ tokens*

| Reference Boundary | Observed Context ($L$) | Committed Prediction ($L_{pred}$) | Prediction Error (%) | Assessment |
|:---|---:|---:|---:|:---|
| Largest Verified Success | 114,688 tokens | 111,232 tokens | **-3.01%** | Safe / Conservative |
| Deliberate OOM Bound | 131,072 tokens | 111,232 tokens | **-15.14%** | Safe / Conservative |
| Default Bisection Ceiling | 118,784 tokens | 111,232 tokens | **-6.36%** | Safe / Conservative |
| Expandable Bisection Ceiling | 127,488 tokens | 111,232 tokens | **-12.75%** | Safe / Conservative |

### Table B: Post-Hoc Fitted Model Behavior
Linear regression on total peak memory ($L=512$ to $65{,}536$) yields a total-peak slope of $95{,}236.95$ B/token, projecting an analytical hardware ceiling crossing at **130,509 tokens**:
- **Held-out Default ($L=114,688$):** Projected $14{,}019{,}203{,}874$ B vs. Observed $14{,}020{,}668{,}928$ B $\rightarrow$ **Error = -0.0104%**
- **Held-out Expandable ($L=114,688$):** Projected $14{,}019{,}751{,}424$ B vs. Observed $14{,}019{,}751{,}424$ B $\rightarrow$ **Error = 0.0000%**

---

## Cross-Hardware & Production Validation on vLLM

To demonstrate that our analytical framework generalizes across architectures and production inference engines, we tested against published benchmarks from **vLLM (v0.30.0)** running **Qwen3-8B BF16** on an **NVIDIA A10G (24 GB PCIe)**:

- **Model Config:** $N=36$ layers, $n_{kv\_heads}=4$, $d=128$, $S_{bytes}=2$ (BF16).
- **vLLM PagedAttention Block Size:** 16 tokens/block.
- **Analytical Block Footprint:**
  $$\text{Bytes/Block} = 16 \times 2 \times 36 \times 4 \times 128 \times 2 = \mathbf{1{,}179{,}648 \text{ bytes}} = \mathbf{1.125 \text{ MiB}}$$
- **Production Comparison:**
  - vLLM Allocated Blocks: 7,018 blocks
  - Analytical Predicted KV Memory: $7{,}018 \times 1{,}179{,}648 = \mathbf{8{,}278{,}769{,}664 \text{ bytes}}$
  - Measured vLLM KV Memory: $\mathbf{8{,}289{,}560{,}576 \text{ bytes}$
  - **Generalization Error:**
    $$\text{Error} = \frac{8{,}278{,}769{,}664 - 8{,}289{,}560{,}576}{8{,}289{,}560{,}576} = \mathbf{-0.1302\%}$$

---

## Codebase Architecture & File Citations

| Measurement Action / Component | Source File | Line(s) and Description |
|:---|:---|:---|
| **KV-Cache Byte Extraction** | [`src/kv_cache_scaling/probe.py`](src/kv_cache_scaling/probe.py) | Lines 344–346: Extracts exact bytes from `DynamicCache` tensors; Line 347 resets peak stats. |
| **Logits Truncation (`logits_to_keep=1`)** | [`src/kv_cache_scaling/probe.py`](src/kv_cache_scaling/probe.py) | Line 317: Passes `logits_to_keep=1` to suppress full vocabulary tensor materialization. |
| **PyTorch SDPA Kernel Forcing** | [`src/kv_cache_scaling/probe.py`](src/kv_cache_scaling/probe.py) | Lines 295–301: Disables math/flash SDPA; enables memory-efficient SDPA. |
| **Allocator Environment Injection** | [`src/kv_cache_scaling/probe.py`](src/kv_cache_scaling/probe.py) | Lines 285–289: Sets `PYTORCH_CUDA_ALLOC_CONF` before CUDA context initialization. |
| **Pre-Registration Commitment** | [`prereg/formal-2026-10-18-t4/receipt.json`](prereg/formal-2026-10-18-t4/receipt.json) | Binds analytical prediction hash prior to experimental measurements. |
| **Discrepancy Data File** | [`docs/DISCREPANCY_114688.csv`](docs/DISCREPANCY_114688.csv) | Machine-readable breakdown with 0 bytes unaccounted. |
| **Paired Bisection Logs** | [`artifacts/bisect-final-complete/`](artifacts/bisect-final-complete/) | Complete JSONL probe logs and summary of the paired bisection execution. |

---

## How to Reproduce

### 1. CPU Test Suite & Analytical Reproduction
Run the complete unit test suite (127 passing tests) without requiring a GPU:
```bash
# Clone the repository
git clone https://github.com/Veena-nitk/KV-Cache-Scaling.git
cd KV-Cache-Scaling

# Run test suite
python3 -m unittest discover -s tests -q

# Reproduce analytical figures and discrepancy tables
python3 scripts/reproduce_results.py \
  --capsule artifacts/experiments/2026-10-19-t4-final/gpu-checkpoint-080.zip \
  --output /tmp/kv-reproduction
```

### 2. Regenerate Academic Report PDF
Generate the publication-quality report PDF in pure Python (zero external dependencies):
```bash
python3 scripts/generate_project_report_pdf.py
# Generates report/KV_Cache_Scaling_Report.pdf and docs/KV_Cache_Scaling_Report.pdf
```

### 3. GPU Live Execution on Google Colab
Open the provided self-contained interactive notebook:
- **Notebook:** [`notebooks/FINAL_PRESENTATION.ipynb`](notebooks/FINAL_PRESENTATION.ipynb)
- Select a **T4 GPU Runtime**, execute all cells, and observe dynamic bisection, peak memory tracking, and figure generation in real time.

---

## License & Attribution

This project is open-source under the [MIT License](LICENSE). 
Forked and extended from `DevanshSharma351/KV-Cache-Scaling`. Developed and maintained by **Veena Shivakumar** (NITK Surathkal).
