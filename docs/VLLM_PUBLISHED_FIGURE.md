# Published measured vLLM KV-cache capacity (Part A4)

## Measured figure

The published benchmark reports **23,040 GPU KV-cache tokens** for `Qwen/Qwen3-8B` (BF16, unquantized) with **vLLM 0.30.0**, on one AWS EC2 `g5.xlarge` carrying **one NVIDIA A10G with 24 GB**. The run used the default `gpu_memory_utilization` setting (0.92, consistent with the reported 92% GPU memory reservation) except `--max-model-len 8192`. The A10G is the PCIe Gen4 variant (the AWS/NVIDIA data sheet gives 24 GB GDDR6 and PCIe Gen4; it is not an SXM GPU). Sources: [published benchmark and startup figure](https://github.com/walidelkhatib/llm-inference-benchmarks), [vLLM memory configuration documentation](https://docs.vllm.ai/en/latest/configuration/optimization/), [Qwen/Qwen3-8B config.json](https://huggingface.co/Qwen/Qwen3-8B/blob/e8bbd8252970581ea5b08b6a5b3e668adaf3161a/config.json), [AWS/NVIDIA A10G data sheet](https://d1.awsstatic.com/product-marketing/ec2/NVIDIA_AWS_A10G_DataSheet_FINAL_02_17_2022.pdf).

The benchmark reports a 3.16 GiB KV pool and gives the startup line `GPU KV cache size: 23,040 tokens`. The model config specifies 36 layers, 8 KV heads, and head dimension 128. For BF16 (2 bytes/element), the repository's analytical KV formula gives:

```text
bytes/token = 2 (K and V) × 36 layers × 8 KV heads × 128 dimensions × 2 bytes
            = 147,456 bytes/token
analytical tokens = floor(3.16 × 2^30 / 147,456) = 23,010 tokens
signed error = analytical − measured = 23,010 − 23,040 = −30 tokens
percent error = −30 / 23,040 × 100 = −0.1302%
```

**Interpretation:** this comparison is conditional on the benchmark's reported, rounded 3.16 GiB pool. It checks our config-derived per-token arithmetic against the published measured pool capacity. It is not an independent forecast of the A10G's free KV memory: weights, activation peak, allocator overhead, and the memory pool were measured by that benchmark's vLLM startup. Since 3.16 GiB is rounded, the calculated −0.1302% is not more precise than the source figure.

## Scope and limitations

This is a measured total KV-cache capacity, not the maximum context of one request and not a vLLM block-size comparison. Model, dtype, utilization, vLLM version, GPU name, memory size, and PCIe form are identified above. The publisher reports one run per configuration. It is a different model and GPU from this project's Qwen2.5-1.5B FP16 T4 experiment, so it validates only the architecture-based bytes-per-token calculation under the published benchmark's reported pool size. It does not validate the project's T4 memory boundary or fitted peak-memory model.
