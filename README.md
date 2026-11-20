# KV-Cache Scaling: Empirical & Analytical Memory Characterization in Large Language Models

[![Python 3.11–3.13](https://img.shields.io/badge/Python-3.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://www.python.org/)
[![PyTorch 2.6.0](https://img.shields.io/badge/PyTorch-2.6.0-EE4C2C.svg)](requirements-gpu.txt)
[![CUDA 12.4](https://img.shields.io/badge/CUDA-12.4-76B900.svg)](requirements-gpu.txt)

An empirical and analytical study of GPU memory during long-context prefill for **Qwen2.5-1.5B** on an **NVIDIA Tesla T4**. We derive the KV-cache footprint, commit a context estimate before collecting GPU data, and measure memory in checkpointed subprocesses.

- **Authors** · Devansh Sharma and Veena S
- **Affiliation** · Department of Computer Science and Engineering, National Institute of Technology Karnataka (NITK), Surathkal
- **Repository** · [DevanshSharma351/KV-Cache-Scaling](https://github.com/DevanshSharma351/KV-Cache-Scaling)
- **Academic report** · [Final report (PDF)](KV_Cache_Scaling_Final_Report.pdf)

---

## Table of Contents

1. [Executive Summary & Core Results](#executive-summary--core-results)
2. [Theoretical Memory Model & GQA Derivation](#theoretical-memory-model--gqa-derivation)
3. [Pre-Registration & Scientific Rigor](#pre-registration--scientific-rigor)
4. [Subprocess-Isolated Probing Architecture](#subprocess-isolated-probing-architecture)
5. [Empirical Measurements & Memory Sweep](#empirical-measurements--memory-sweep)
6. [Discrepancy Accounting at 114,688 Tokens](#discrepancy-accounting-at-114688-tokens)
7. [Allocator Benchmark: Default vs. Expandable Segments](#allocator-benchmark-default-vs-expandable-segments)
8. [OOM Boundary & the Framework Mask Cliff](#oom-boundary--the-framework-mask-cliff)
9. [Table A vs. Table B: Committed and Fitted Predictions](#table-a-vs-table-b-committed-and-fitted-predictions)
10. [Cross-Hardware Predictions & Published vLLM Comparison](#cross-hardware-predictions--published-vllm-comparison)
11. [Codebase Architecture & File Citations](#codebase-architecture--file-citations)
12. [How to Reproduce](#how-to-reproduce)

---

## Executive Summary & Core Results

| Result | Value |
|:---|:---|
| FP16 model weights | 3,087,428,608 bytes |
| KV-cache growth | 28,672 B/token for each allocator at batch size 1, with R² = 1.0 |
| Primary measurement coverage | 52 probes across nine lengths from 512 to 114,688 tokens |
| Targeted default-allocator boundary run | Prefill succeeded at 114,688 tokens and raised CUDA OOM at 131,072 on the same T4 |
| Expandable segments at 114,688 tokens | 644 MiB less peak reserved memory than default |
| Published Qwen3-8B cache-pool comparison | 23,010 estimated cache tokens vs. 23,040 in the benchmark log |

## Theoretical Memory Model & GQA Derivation

### 1. Static Parameter & Weight Calculation

The pinned configuration has 28 layers, a vocabulary of 151,936, a hidden width of 1,536, an intermediate width of 8,960, **12 query heads**, and **two KV heads**, with a head dimension of 128. Counting the tied embedding once and including projections, biases, gated MLP weights, and RMSNorm scales gives **1,543,714,304 unique parameters** or **3,087,428,608 bytes** in FP16. [Full derivation](docs/DERIVATION.md) · [Pinned configuration](configs/models/qwen2.5-1.5b.json)

### 2. Grouped-Query Attention (GQA) KV-Cache Scaling Law

For batch size $B$, sequence length $L$, $N$ layers, $K$ KV heads, head dimension $D$, and element width $s$ in bytes, the retained cache is

$$
\mathrm{KV}(L,B) = 2BLNKDs = 28{,}672BL\ \text{bytes}.
$$

Using the 12 query heads in place of the two KV heads overestimates retained cache storage by a factor of six.

## Pre-Registration & Scientific Rigor

The central estimate of **111,232 tokens** was committed before GPU measurement using a post-context-startup budget of **15,527,116,800 bytes**. The scenario range was **76,024–131,072 tokens** after the model configuration cap. The central peak tensor estimate uses the model weights, KV term, and an assumed transient term. The admission calculation adds a fixed headroom reserve. [Original commit](https://github.com/DevanshSharma351/KV-Cache-Scaling/commit/7baff5121ed643f27ba06959065855a8913f8e8c) · [Observation receipt](prereg/formal-2026-10-18-t4/receipt.json)

## Subprocess-Isolated Probing Architecture

Each probe runs in a fresh subprocess. The worker loads the pinned FP16 model, starts with an empty `DynamicCache`, and runs one cold prefill at batch size 1 using efficient SDPA and `logits_to_keep=1`. The measured runtime used Python 3.11, PyTorch 2.6.0+cu124, CUDA 12.4, and Transformers 4.51.3.

The worker synchronizes CUDA and resets peak counters after loading the model and before allocating inputs. It records peak allocated and reserved bytes and checks the cache tensors directly. The coordinator journals each result and verifies a checkpoint before dispatching the next probe. [Probe implementation](src/kv_cache_scaling/probe.py) · [Coordinator contract](docs/COORDINATOR_CONTRACT.md)

## Empirical Measurements & Memory Sweep

The primary measurements include **27 default** and **25 expandable** probes. Every allocator-length cell has three repeats except expandable at 114,688 tokens, which has one. The table shows per-cell medians. [Primary probe data](data/processed/project-results/probes.csv)

| Length (tokens) | Repeats (default / expandable) | KV cache (B) | Default peak allocated (B) | Default peak reserved (B) | Expandable peak reserved (B) |
|---:|:---:|---:|---:|---:|---:|
| 512 | 3 / 3 | 14,680,064 | 3,146,681,856 | 3,330,277,376 | 3,164,602,368 |
| 1,024 | 3 / 3 | 29,360,128 | 3,195,448,832 | 3,403,677,696 | 3,202,351,104 |
| 2,048 | 3 / 3 | 58,720,256 | 3,295,079,936 | 3,529,506,816 | 3,317,694,464 |
| 4,096 | 3 / 3 | 117,440,512 | 3,487,002,112 | 3,693,084,672 | 3,529,506,816 |
| 8,192 | 3 / 3 | 234,881,024 | 3,877,137,920 | 4,240,441,344 | 3,925,868,544 |
| 16,384 | 3 / 3 | 469,762,048 | 4,657,409,536 | 5,171,576,832 | 4,764,729,344 |
| 32,768 | 3 / 3 | 939,524,096 | 6,217,952,768 | 6,731,857,920 | 6,400,507,904 |
| 65,536 | 3 / 3 | 1,879,048,192 | 9,339,039,232 | 10,171,187,200 | 9,695,133,696 |
| 114,688 | 3 / 1 | 3,288,334,336 | 14,020,668,928 | 15,319,695,360 | 14,644,412,416 |

A linear fit to directly inspected KV bytes gave **28,672 B/token** and **R² = 1.0** for each allocator. [Analysis and fits](data/processed/project-results/analysis.json)

## Discrepancy Accounting at 114,688 Tokens

The central tensor estimate was **15,359,962,112 bytes**. The signed gap is measured peak minus predicted peak for one saved probe per allocator. The component sum closes with a zero arithmetic residual for both probes. The largest signed term compares measured transient excess with the assumed envelope. The cuBLAS workspace term is inferred from the pinned source and 512-token controls. [Signed component table](docs/DISCREPANCY_114688.csv)

| Allocator | Measured peak allocated (B) | Signed gap (B) | Arithmetic residual (B) |
|:---|---:|---:|---:|
| Default | 14,020,668,928 | -1,339,293,184 | 0 |
| Expandable | 14,019,751,424 | -1,340,210,688 | 0 |

## Allocator Benchmark: Default vs. Expandable Segments

The expandable arm sets `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` before CUDA initialization. This comparison uses peak reserved bytes at matched input lengths.

| Length (tokens) | Default peak reserved (B) | Expandable peak reserved (B) | Reduction |
|---:|---:|---:|---:|
| 16,384 | 5,171,576,832 | 4,764,729,344 | 388 MiB |
| 32,768 | 6,731,857,920 | 6,400,507,904 | 316 MiB |
| 65,536 | 10,171,187,200 | 9,695,133,696 | 454 MiB |
| 114,688 | 15,319,695,360 | 14,644,412,416 | **644 MiB (4.41%)** |

## OOM Boundary & the Framework Mask Cliff

### 1. Observed Default-Allocator Boundary

A targeted run on a separate T4 allocation completed a 114,688-token prefill and raised a typed forward CUDA OOM at 131,072 tokens. Under the monotone fit/failure assumption, these observations bracket the largest fitting integer context at **114,688–131,071 tokens**. [Boundary verification](artifacts/experiments/2026-10-21-boundary/verification.json)

The submission branch implements a paired-bisection coordinator and Colab notebook. The coordinator checks the registered GPU UUID, schedules anchor probes and 512-token midpoint steps, and checkpoints each probe. [Coordinator](scripts/bisect_boundary.py) · [Boundary notebook](notebooks/BOUNDARY_COLAB.ipynb)

### 2. Framework Mask Allocation at 131,072 Tokens

At 131,072 tokens, the mask-elision condition in Transformers 4.51.3 fails at the configured window. The `DynamicCache` path requests a dense FP16 causal mask of shape `[1, 1, 131072, 131073]` before the efficient SDPA kernel. Its **34,360,000,512 bytes (32.00024 GiB)** match the recorded error’s rounded **32.00 GiB** request. [Captured framework source](artifacts/experiments/2026-10-21-boundary/masking-source.json)

## Table A vs. Table B: Committed and Fitted Predictions

### Table A. Committed Context Prediction

The central estimate’s signed error is 100 × (prediction − limit) / limit, evaluated at each endpoint of the verified fitting interval.

| Fitting interval endpoint (tokens) | Committed prediction (tokens) | Signed error |
|---:|---:|---:|
| 114,688 | 111,232 | -3.01% |
| 131,071 | 111,232 | -15.14% |

### Table B. Fitted Peak-Memory Models

The in-sample peak-memory fits over all nine lengths give candidate crossings of **130,509 tokens** for default and **130,513 tokens** for expandable at the original usable-memory budget. A separate frozen post-sweep model gives these held-out results at 114,688 tokens. [Held-out data](data/processed/project-results/held_out_validation.csv)

| Allocator | Held-out repeats | Frozen peak estimate (B, rounded) | Observed median peak (B) | Signed error |
|:---|---:|---:|---:|---:|
| Default | 3 | 14,019,203,874 | 14,020,668,928 | -0.010449% |
| Expandable | 1 | 14,019,751,424 | 14,019,751,424 | 0% |

## Cross-Hardware Predictions & Published vLLM Comparison

The configuration-based transfer model produces the following central forecasts after the model caps. The usable VRAM and transient memory terms are scenario assumptions. [Transfer results](data/processed/project-results/transfer/transfer.json)

| Model | A100 40 GB | RTX 4090 24 GB |
|:---|---:|---:|
| Qwen2.5-3B | 32,768 | 32,768 |
| Qwen2.5-7B | 109,629 | 39,097 |

A published vLLM 0.30.0 benchmark for **Qwen3-8B BF16 on an NVIDIA A10G** reports a rounded **3.16 GiB** cache pool and **23,040 cache tokens**. With 36 layers, eight KV heads and a head dimension of 128, the formula gives 147,456 B/token. The reported pool implies **23,010 tokens**, about **0.13% below** the logged capacity. This comparison checks the cache formula using the benchmark’s reported pool size. [Published benchmark](https://github.com/walidelkhatib/llm-inference-benchmarks) · [Calculation](docs/VLLM_PUBLISHED_FIGURE.md)

## Codebase Architecture & File Citations

| Component | Source |
|:---|:---|
| CUDA probe, cache inspection and peak counters | [probe.py](src/kv_cache_scaling/probe.py) |
| Allocator environment and subprocess launch | [processes.py](src/kv_cache_scaling/processes.py) |
| Checkpointed measurement sweep | [sweep.py](src/kv_cache_scaling/sweep.py) · [coordinator store](src/kv_cache_scaling/coordinator_store.py) |
| Paired-bisection tooling | [Coordinator](scripts/bisect_boundary.py) · [Unit tests](tests/test_bisect_boundary.py) |
| Signed discrepancy data | [114,688-token comparison](docs/DISCREPANCY_114688.csv) · [Primary discrepancy data](data/processed/project-results/discrepancy.csv) |
| Figures and analysis | [Scientific figures](artifacts/figures/project-results/) · [Processed results](data/processed/project-results/) |

## How to Reproduce

### 1. CPU Test Suite & Analytical Reproduction

From the Git repository root, use Python 3.12, as in the report CI job. Choose a new output directory for each replay.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e . -r requirements-analysis.txt
python -m unittest discover -s tests -q
python scripts/reproduce_results.py \
  --capsule artifacts/experiments/2026-10-19-t4-final/gpu-checkpoint-080.zip \
  --output /tmp/kv-reproduction
```

### 2. Academic Report PDF

The CPU replay writes tables, figures, transfer outputs, and a generated evidence report under the output directory. The [final academic report](KV_Cache_Scaling_Final_Report.pdf) presents the findings and submission-branch analysis in paper format. [Reproduction guide](docs/REPRODUCTION.md)

### 3. Presentation on Google Colab

Open [FINAL_PRESENTATION.ipynb](notebooks/FINAL_PRESENTATION.ipynb) to replay the saved analysis on CPU. The [published Colab board](https://colab.research.google.com/github/DevanshSharma351/KV-Cache-Scaling/blob/e0b545c/notebooks/FINAL_PRESENTATION.ipynb#scrollTo=a9d9306e) displays the figures and endpoint table without a runtime.

---

## Attribution

Research and report by **Devansh Sharma** and **Veena S**, Department of Computer Science and Engineering, NITK Surathkal.
