# Chronos

**A lightweight temporal-tile scheduler for spatiotemporal neural networks (STNNs).**

STNNs — spiking neural networks, ConvLSTM/ConvGRU models, and other stateful architectures — interleave two kinds of operators: *spatial* operators (conv, linear) that are independent across time steps and dominate compute, and *temporal* operators (membrane updates, gates) that carry state and serialize execution. Existing schedulers and compilers fuse only within a single time step, leaving cross-time parallelism and cross-time batching untouched — or pay hours of autotuning to recover it.

Chronos closes this gap with a scheduling abstraction, not a kernel library: it reorganizes the *execution* of an STNN graph while keeping every spatial operator on its fastest existing implementation.

## Highlights

- **Mean speedups on A100** of roughly 2.0× over SpikingJelly and TorchScript, 2.0× over BladeDISC, and 1.25× over TensorRT across 11 STNN workloads; **1.58× over Welder on V100**; **1.42× average on CPU** over TorchScript/SpikingJelly.
- **Scheduling overhead of seconds**, versus hours (TVM) to thousands of seconds (search-based compilers) — no per-model kernel autotuning.
- Portable by construction: multi-stream execution on NVIDIA GPUs, fusion-and-codegen backend on CPUs; the temporal restructuring alone yields gains even without multi-stream hardware.

## The tTile abstraction

A **tTile** is a schedulable execution block capturing a bounded region of dataflow *and stateflow*: (1) spatial operator instances drawn from consecutive time steps that share no temporal dependence, and (2) the temporal operator instances whose dependences fit entirely inside the same window. It is loop tiling applied jointly to the spatial and temporal dimensions of the operator stack — tTiles along the anti-diagonal are dependence-free and run concurrently.

Three design decisions follow:

1. **Adaptive spatial batching across time.** Spatial instances at different t are independent given their inputs, so Chronos batches them into single vendor-library calls — with a *configurable* batch size selected by a cost model, not fixed at 1 or T.
2. **Batching-guided temporal fusion.** Temporal operators remain strictly sequential in t; Chronos fuses them only where the chosen spatial batch makes it valid, never mutating state prematurely.
3. **Deliberately conservative spatial fusion.** Chronos avoids fusing spatial operators where fusion would demand expensive autotuning or forfeit vendor-library kernels; it fuses only always-profitable patterns (e.g., elementwise chains). This is what keeps scheduling cost at seconds.

## Scheduling and runtime

- **Dual-level holistic placement** assigns every operator a 4D coordinate (tTile-level and intra-tTile 2D positions), computed by two topological traversals — the static interface between graph analysis and the runtime.
- **Track-based SPSP execution**: a hierarchical track abstraction (trackgroups → tracks) dynamically binds coordinates to execution slots, alternating sequential and parallel phases. On GPUs, tracks map to CUDA streams; tTile-internal paths overlap with batched vendor kernels across streams.
- **Cost-model-driven size selection** picks the temporal batch size per region; the model's coefficients are robust — top-1-or-2 configuration hit rate stays at 100% under ±20% coefficient perturbation across hardware/time-step settings.

## Evaluated workloads

11 STNNs spanning three families: spiking CNNs (ResNet18/34, AlexNet, ZFNet, VGG11, MobileNet-V1/V2), conv-recurrent networks (ConvLSTM, ConvGRU), and spiking transformers (Spikformer-style, SpikeBERT-style), on A100, V100, and Xeon CPU.

## Getting started

```bash
git clone https://github.com/pengjh0111/Chronos
cd Chronos
# TODO: pip install -r requirements.txt  (PyTorch version, SpikingJelly for baselines)

# TODO: single-model entry point, e.g.
# python run.py --model resnet18_snn --timesteps 16 --device cuda
```

<!-- TODO: fill in actual entry points, dataset/checkpoint preparation, and baseline installation
     (SpikingJelly / TorchScript / BladeDISC / TensorRT / Welder) for reproducing the paper tables. -->

## Repository structure

```
# TODO: adjust to the actual tree
scheduler/    tTile construction, dual-level placement, cost model
runtime/      track-based SPSP execution (CUDA streams backend, CPU codegen backend)
models/       STNN workload definitions
benchmarks/   end-to-end evaluation scripts and baseline harnesses
```

## Relationship to triton-snn-fusion

Chronos treats every operator as an opaque kernel and optimizes *when and where* it runs. Its successor, [triton-snn-fusion](https://github.com/pengjh0111/triton-snn-fusion), opens the kernel: it compiles a bounded spatio-temporal region into a single fused Triton kernel in which weights stay resident on-chip across a temporal window and the state recurrence runs as an in-register epilogue — removing the memory traffic that no cross-kernel scheduler can reach. The two are complementary layers of the same stack.

## Citation

The Chronos paper is currently under submission. <!-- TODO: add BibTeX after publication -->

## License

<!-- TODO: choose and add a license file -->
