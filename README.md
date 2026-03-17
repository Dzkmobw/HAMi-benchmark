# Benchmark Plan

This directory defines the benchmark content for the external scheduler.

Target metrics:

- average job completion time
- GPU utilization
- GPU memory utilization

## Why this benchmark

For a single RTX 4070 SUPER, the external scheduler is most likely to beat
default HAMi behavior under mixed workloads, not under one exclusive job.

So the benchmark uses a mixed queue with three workload classes:

- short-inference
- medium-inference
- background-training

This stresses:

- queue ordering
- admission timing
- memory-aware co-location
- interference control

## Plans

Two plans are defined:

- `benchmark-plan-mini.yaml`
  - quick validation round
  - expected duration: 3 to 6 minutes
- `benchmark-plan.yaml`
  - normal comparison round
  - expected duration: 10 to 20 minutes

`benchmark-plan.yaml` is the current full-round plan.

## Selected workload mix

The benchmark mix for one round is:

- 7 short-inference jobs
- 2 medium-inference jobs
- 1 background-training job

This is a 70 / 20 / 10 mix.

## Workload classes

### 1. short-inference

Purpose:

- improve average completion time
- keep GPU busy with short jobs

Recommended real task:

- sentence embedding inference
- or OCR single-image inference

Recommended scheduler metadata:

- class: latency-sensitive
- runtime-seconds: 20 to 40
- gpu-mem-mib: 800 to 1200

Recommended HAMi resources:

- nvidia.com/gpu: 1
- nvidia.com/gpumem: 1000
- nvidia.com/gpucores: 20

### 2. medium-inference

Purpose:

- improve GPU utilization and memory utilization
- create contention that the scheduler can manage

Recommended real task:

- YOLOv8n image inference
- or small batched vision inference

Recommended scheduler metadata:

- class: throughput
- runtime-seconds: 60 to 120
- gpu-mem-mib: 1800 to 2600

Recommended HAMi resources:

- nvidia.com/gpu: 1
- nvidia.com/gpumem: 2200
- nvidia.com/gpucores: 35

### 3. background-training

Purpose:

- create long-running interference
- test whether short jobs can still complete faster

Recommended real task:

- CIFAR10 ResNet18 training
- or a small LoRA / fine-tuning loop

Recommended scheduler metadata:

- class: background
- runtime-seconds: 300 to 900
- gpu-mem-mib: 3500 to 5000

Recommended HAMi resources:

- nvidia.com/gpu: 1
- nvidia.com/gpumem: 4000
- nvidia.com/gpucores: 45

## Round definition

One benchmark round should follow this submission pattern:

1. submit 1 background-training job
2. submit 2 medium-inference jobs
3. submit 7 short-inference jobs

The jobs should be submitted within a short arrival window, for example
10 to 30 seconds, so that queueing decisions matter.

## Comparison modes

Run exactly the same job mix in two modes:

1. HAMi only
2. HAMi + external scheduler

## Success criteria

The external scheduler is considered better if it achieves:

- lower average job completion time
- higher average GPU utilization
- higher average GPU memory utilization

while keeping:

- no extra OOM
- no deadlocked queued jobs

## Notes

- Use real deep learning tasks, not sleep-only containers.
- Keep image, dataset, batch size, and submission order identical between runs.
- Use the same measurement window for both modes.
