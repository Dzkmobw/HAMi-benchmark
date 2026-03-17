# HAMi Mini Benchmark

This directory turns the mini benchmark plan into runnable workloads.

## What it runs

The runnable mini benchmark uses three real GPU compute workloads implemented in PyTorch:

- `short-inference`
  - embedding-style inference
  - 3 jobs
- `medium-inference`
  - vision-style inference
  - 1 job
- `background-training`
  - short CNN training loop
  - 1 job

These workloads are designed to be runnable with minimal setup:

- no extra dataset download is required
- no extra model download is required
- all jobs use the same base image
- the workload code is mounted through a ConfigMap

## Important note

These are runnable benchmark workloads, not production models.

They are intentionally implemented as:

- real PyTorch inference/training loops
- synthetic inputs
- optional GPU memory reservation buffers

This keeps them easy to run on your current setup while still exercising:

- GPU compute
- GPU memory usage
- scheduler queueing behavior

## Layout

- `workloads/mini_workload.py`
  - one Python entrypoint for all mini tasks
- `manifests/hami-only`
  - jobs for the plain HAMi baseline
- `manifests/external`
  - jobs for the external-scheduler mode
- `run-mini-hami-only`
  - run the mini round in HAMi-only mode and write logs
- `run-mini-external`
  - run the mini round in external-scheduler mode and write logs
- `run-mini-common`
  - shared runner used by both modes
- `cleanup-mini`
  - remove mini benchmark jobs and mounted workload ConfigMap
- `tools/benchmark_logger.py`
  - sample GPU metrics and write a run summary

## Prerequisites

- HAMi is running
- for external mode, the external controller is running
- the benchmark namespace can be created in the cluster
- the image `pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime` can be pulled

## Run

HAMi only:

```bash
/home/ttk/project/HAMi-benchmark/mini/run-mini-hami-only
```

External scheduler mode:

```bash
/home/ttk/project/HAMi-benchmark/mini/run-mini-external
```

Cleanup:

```bash
/home/ttk/project/HAMi-benchmark/mini/cleanup-mini
```

## Logs

Each run creates a timestamped log directory under:

```bash
/home/ttk/project/HAMi-benchmark/mini/logs
```

Each run directory contains:

- `run.log`
- `metrics_samples.csv`
- `pods.json`
- `jobs.json`
- `summary.json`
- `summary.txt`

The automatic summary includes the main comparison metrics:

- average completion time
- average GPU utilization
- average GPU memory utilization

It also records:

- average runtime
- average queue delay
- max GPU utilization
- max GPU memory utilization
