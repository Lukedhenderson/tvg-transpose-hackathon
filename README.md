# tvg-transpose-hackathon

An ML model **energy optimization** pipeline that takes any model, applies internal optimizations, and reports **original vs optimized energy cost** with **token and energy savings**, using **real energy-per-instruction** estimates for CPU and GPU.

## Goals

- **Input:** Any ML model (e.g., LLM, vision, small models).
- **Internal optimization:** Apply model- and hardware-aware optimizations (e.g., pruning, quantization, operator fusion, scheduling) to reduce compute and memory.
- **Output:**
  - **Original energy cost** vs **optimized energy cost** (e.g., J or kWh per run/batch).
  - **Token savings** (e.g., tokens processed per unit energy, or tokens saved per run).
  - **Energy savings** (absolute and relative) for CPU and GPU.

## Energy Model

Energy estimates are based on **real energy per instruction** (or per operation) for:

- **CPU:** Measured or published energy per instruction / per operation for the target ISA and microarchitecture (e.g., from RAPL, power meters, or literature).
- **GPU:** Measured or published energy per FLOP / per instruction for the target GPU (e.g., from nvidia-smi, NVML, or vendor/published data).

The pipeline will:

1. Profile or estimate **instruction/op counts** (and types) for the original and optimized model on CPU and GPU.
2. Multiply by **energy-per-instruction (or per-FLOP) coefficients** derived from real hardware measurements or trusted sources.
3. Report **total energy** (J or kWh), **tokens processed**, and **savings** (tokens and energy).

## Metrics

| Metric | Description |
|--------|-------------|
| **Original energy cost** | Total energy (J/kWh) to run the unoptimized model (CPU + GPU). |
| **Optimized energy cost** | Total energy (J/kWh) to run the optimized model. |
| **Energy savings** | Absolute (J/kWh) and relative (%) reduction. |
| **Token savings** | Tokens processed per unit energy (e.g., tokens/J) or tokens “saved” per run by the optimized model. |

## Roadmap

- [ ] Define energy-per-instruction / per-FLOP lookup (CPU + GPU).
- [ ] Integrate model loader and baseline profiler (ops + tokens).
- [ ] Implement internal optimization passes (e.g., quantization, pruning).
- [ ] Compute original vs optimized energy and token/energy savings.
- [ ] Output report: original energy, optimized energy, token savings, energy savings.
