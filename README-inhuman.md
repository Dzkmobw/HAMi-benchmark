# HAMi Benchmark: README-inhuman

This document is for coding agents, not end users.

Its purpose is to make this repository understandable without prior chat context.

## 1. Repository Purpose

This repository exists to benchmark and compare:

- plain HAMi behavior
- HAMi plus the external scheduler in the sibling repository
- and, in some cases, a host-direct baseline outside Kubernetes

Primary comparison metrics:

- average completion time
- GPU utilization
- GPU memory utilization

The benchmark is designed around a **single-GPU mixed workload** setup, specifically a 4070 SUPER-class environment.

## 2. Relationship To Other Repositories

This repository assumes:

- sibling HAMi source exists at:
  - `/home/ttk/project/HAMi`
- sibling external scheduler repo exists at:
  - `/home/ttk/project/HAMi-external-scheduler`

This repo does not contain HAMi itself.
It contains benchmark plans, runnable benchmark workloads, orchestration scripts, and logs.

## 3. High-Level Structure

Top-level important files:

- `README.md`
  - user-oriented benchmark overview
- `benchmark-plan.yaml`
  - full-plan conceptual benchmark definition
- `benchmark-plan-mini.yaml`
  - quick-plan conceptual definition

Runnable benchmark implementations:

- `mini/`
  - Kubernetes-based benchmark workloads for:
    - HAMi-only
    - external-scheduler mode
- `host-mini/`
  - host-direct baseline outside Kubernetes/HAMi

Historical outputs:

- `mini/logs/`
- `host-mini/logs/`

These logs are useful for analysis but should not be treated as source of truth for repository behavior.

## 4. Conceptual Plans vs Runnable Reality

There are two layers in this repository:

### 4.1 Conceptual benchmark plans

Files:

- `benchmark-plan.yaml`
- `benchmark-plan-mini.yaml`

These define:

- intended workload mix
- intended objective
- suggested real-world task analogs
- comparison modes

They are design documents, not the direct execution engine.

### 4.2 Runnable benchmark implementation

Files and directories:

- `mini/workloads/mini_workload.py`
- `mini/manifests/hami-only/`
- `mini/manifests/external/`
- `mini/run-mini-common`
- `mini/run-mini-hami-only`
- `mini/run-mini-external`
- `mini/cleanup-mini`
- `mini/tools/benchmark_logger.py`
- `host-mini/tools/host_benchmark_runner.py`
- `host-mini/run-host-mini`

This runnable layer is the real operational benchmark system.

Important note:

The current runnable mini workload mix is **not identical** to the original conceptual plan in `benchmark-plan.yaml`.
The conceptual plan still describes a 7/2/1-style full round, while the runnable mini has been evolved several times for this project.

Current runnable mini source of truth is:

- manifests in `mini/manifests/`
- `TASKS` in `host-mini/tools/host_benchmark_runner.py`

## 5. Current Runnable Mini Mix

Current runnable mini benchmark consists of:

- 1 background-training job
- 3 medium-inference jobs
- 6 short-inference jobs

Current runnable timings in practice:

- background-training: 150s
- medium-inference: 80s each
- short-inference: 30s each

Current host-direct source of truth:

- `host-mini/tools/host_benchmark_runner.py`

Current Kubernetes source of truth:

- per-job manifests under:
  - `mini/manifests/hami-only/`
  - `mini/manifests/external/`

This mix was intentionally increased beyond the original tiny mini so that total wall time would not be dominated almost entirely by the single background task.

## 6. Mini Benchmark Workloads

The mini benchmark uses a single Python entrypoint:

- `mini/workloads/mini_workload.py`

This file implements three synthetic-but-real GPU workloads in PyTorch.

### 6.1 `short-inference`

Implementation:

- small embedding-style inference model
- random token inputs
- no backward pass

Purpose:

- simulate short latency-sensitive inference
- create many short jobs so queueing order matters

### 6.2 `medium-inference`

Implementation:

- small vision-style convolutional inference model
- random image inputs
- no backward pass

Purpose:

- create heavier inference pressure than short tasks
- raise GPU and memory utilization
- create realistic interference against shorter tasks

### 6.3 `background-training`

Implementation:

- small CNN training loop
- random images and labels
- forward + backward + optimizer step

Purpose:

- create a long-running background job
- simulate interference from training or low-priority continuous work

### 6.4 Memory reservation

The workload implementation includes an optional GPU memory reservation buffer.

This is important:

- the benchmark is not only trying to create compute pressure
- it is also trying to create controllable memory pressure

This is why each job includes:

- runtime
- memory reservation
- batch size

The benchmark therefore produces more realistic GPU contention than a simple sleep container.

## 7. `mini/` Directory

This is the main Kubernetes benchmark path.

### 7.1 `mini/manifests/hami-only/`

Contains Job manifests for plain HAMi mode.

These jobs:

- do not use external scheduler admission gating
- still request HAMi-managed GPU resources
- run the same `mini_workload.py`

### 7.2 `mini/manifests/external/`

Contains Job manifests for external-scheduler mode.

These jobs differ by including:

- `external-scheduler.hami.io/enabled: "true"`
- class annotation
- runtime annotation
- estimated memory annotation
- and external scheduling gates

These manifests are how the benchmark drives the external scheduler.

### 7.3 `mini/run-mini-common`

This is the shared execution wrapper used by both benchmark modes.

Responsibilities:

- ensure namespace exists
- refresh ConfigMap containing `mini_workload.py`
- delete prior mini jobs
- start metrics sampling
- submit manifests from the mode-specific manifest directory
- wait for jobs with the proper labels to finish
- on exit, summarize results

Important current behavior:

- it auto-discovers manifests with `find ... | sort`
- it does not hardcode the number of tasks anymore
- this was necessary once the mini mix expanded beyond the earlier 5-job version

### 7.4 `mini/run-mini-hami-only`

Thin wrapper around `run-mini-common hami-only`.

### 7.5 `mini/run-mini-external`

Thin wrapper around `run-mini-common external`.

### 7.6 `mini/cleanup-mini`

Removes:

- mini benchmark jobs
- mini benchmark pods
- workload ConfigMap

This is routinely used before a fresh benchmark round.

### 7.7 `mini/tools/benchmark_logger.py`

This file samples metrics and writes benchmark summaries.

Key outputs:

- `run.log`
- `metrics_samples.csv`
- `pods.json`
- `jobs.json`
- `summary.json`
- `summary.txt`

Important current summary fields include:

- `total_benchmark_wall_time_seconds`
- `max_task_completion_time_seconds`
- `average_completion_time_seconds`
- `average_runtime_seconds`
- `average_queue_delay_seconds`
- `average_gpu_utilization_percent`
- `average_gpu_memory_utilization_percent`
- `average_gpu_memory_used_mib`
- `max_gpu_utilization_percent`
- `max_gpu_memory_utilization_percent`

Current sampling model:

- periodic polling every 2 seconds
- values averaged over recorded samples
- not a per-second reaggregation system

This matters if you later compare it to other telemetry.

## 8. `host-mini/` Directory

This is the host-direct baseline.

Purpose:

- compare against a run that does not use Kubernetes scheduling, HAMi, or the external scheduler
- still run the same conceptual task mix

### 8.1 `host-mini/tools/host_benchmark_runner.py`

This file is the source of truth for host baseline tasks.

Important structure:

- `TaskSpec`
  - name
  - workload kind
  - runtime
  - reserve memory
  - batch size
- `TaskResult`
  - execution result fields
- `TASKS`
  - current host-direct benchmark task list

Execution model:

- sequential `docker run --gpus all`
- mounts `mini_workload.py`
- passes environment variables into the container
- samples GPU metrics with `nvidia-smi`

This means host-direct is not a Kubernetes baseline.
It is a host-only baseline with the same synthetic workload semantics.

### 8.2 `host-mini/run-host-mini`

Small wrapper that runs the host baseline.

### 8.3 Important semantic difference

Host-direct usually behaves more serially than HAMi-based modes.
That is expected.

Do not interpret host-direct as a scheduler baseline in the same sense as HAMi-only.
It is more of a substrate baseline:

- no HAMi
- no external scheduler
- no K8s-level admission control

## 9. Logs and Historical Runs

This repository already contains many historical benchmark logs.

Important directories:

- `mini/logs/`
- `host-mini/logs/`

These are useful for:

- regression comparison
- seeing what metrics fields look like
- postmortem analysis

But they also create noise.

Important rule for a new agent:

- use code and manifests as source of truth
- use logs as evidence, not as architecture documentation

## 10. Known Historical Findings

These findings matter because they explain why the benchmark and scripts evolved.

### 10.1 HAMi node lock timeout mattered a lot

Earlier in this project, effective behavior still reflected a 5-minute node lock path.
After fixing script behavior so HAMi really launched with:

- `--node-lock-timeout=30s`

HAMi-only benchmark behavior improved dramatically.

This means:

- benchmark results before and after that fix are not directly comparable
- any future regression investigation must first confirm actual live lock timeout behavior

### 10.2 External scheduler was functional before it was better

There were benchmark rounds where the external scheduler was clearly active and working correctly, but still underperformed HAMi-only on key metrics.

This is important:

- admission control correctness is not the same thing as policy quality
- a run can prove the architecture works while also proving the current policy is not yet superior

### 10.3 Small mini mixes were too background-dominated

The runnable mini mix was expanded because older mini rounds were too heavily determined by the single background task.

That is why current runnable mini now uses:

- more medium tasks
- more short tasks

to make total round behavior depend more on queueing and coexistence, not only on one long tail task.

## 11. Important Current Mismatches / Caveats

### 11.1 Conceptual plan vs runnable mini

The design documents still describe one workload mix.
The actual runnable mini currently uses a different mix.

For implementation work, prefer:

- manifests
- `TASKS` in host runner

over high-level conceptual prose.

### 11.2 `mini/README.md` may lag behind current counts

Historically, the README described an older smaller mix.
Always verify current runnable counts from code/manifests.

### 11.3 Logs are intentionally ignored by git

This is correct.
They are outputs, not source files.

## 12. Safe Modification Zones

If you are changing this repo, changes usually belong in one of these zones:

### 12.1 Workload semantics

- `mini/workloads/mini_workload.py`

Use this when you want to change:

- what short/medium/background actually do
- how compute pressure is created
- how memory reservation works

### 12.2 Workload mix

- `mini/manifests/...`
- `host-mini/tools/host_benchmark_runner.py`

Use this when you want to change:

- task counts
- runtime mix
- resource requests
- memory annotations

Keep host-direct and Kubernetes paths aligned unless you intentionally want them to diverge.

### 12.3 Benchmark orchestration

- `mini/run-mini-common`
- `mini/cleanup-mini`
- `mini/tools/benchmark_logger.py`

Use this when you want to change:

- run lifecycle
- metrics capture
- summary calculations
- log outputs

## 13. Recommended Reading Order For A New Agent

If you are new to this repository, read in this order:

1. `README.md`
2. `benchmark-plan.yaml`
3. `mini/workloads/mini_workload.py`
4. `mini/run-mini-common`
5. `mini/tools/benchmark_logger.py`
6. `host-mini/tools/host_benchmark_runner.py`
7. manifests under `mini/manifests/hami-only/` and `mini/manifests/external/`

That order gives the fastest path from concept to runnable behavior.

## 14. Recommended Operational Workflow

If you are validating end-to-end behavior:

1. bring up HAMi and, if needed, external scheduler from the sibling repo
2. run:
   - `mini/cleanup-mini`
3. run one of:
   - `mini/run-mini-hami-only`
   - `mini/run-mini-external`
   - `host-mini/run-host-mini`
4. inspect the timestamped run directory

If comparing modes, keep constant:

- workload mix
- resource requests
- arrival order
- image
- measurement window

## 15. Summary

This repository is not a generic ML benchmark suite.

It is a **scheduler evaluation harness** built around:

- synthetic but real PyTorch GPU workloads
- repeatable mixed-load rounds
- explicit comparison between:
  - HAMi-only
  - HAMi + external scheduler
  - host-direct baseline

Everything in this repository should be interpreted through that lens.
