---
name: benchmark-performance
description: Benchmark or optimize software runtime, throughput, or memory use. Use for software performance measurements and benchmark design, not instruction editing alone.
---

# Benchmark Performance

For design-only requests, provide the benchmark plan without changing code or running workloads. Before implementation, establish a comparable baseline. Choose workloads that represent the behavior the user cares about, including realistic input sizes and relevant slow or memory-intensive cases. Record the environment, command, inputs, and configuration needed to reproduce the measurement.

Measure the metric that matches the goal. Report wall time and peak memory when they are relevant; add throughput, latency distribution, allocation counts, or another metric only when it helps evaluate the requested change. Use warmups and repeated runs when startup cost or noise could change the conclusion.

Make the smallest change that addresses the measured bottleneck. Run the same benchmark under comparable conditions afterward, and directly verify that outputs and externally visible behavior remain equivalent. Do not present an improvement when the runs are too noisy or the workloads are not comparable; explain the uncertainty and what additional measurement would resolve it.

Report the before-and-after results, relative change, benchmark commands, workload, environment, and any material tradeoff. Keep durable benchmark fixtures or results in the repository only when they will support future regression checks or review.
