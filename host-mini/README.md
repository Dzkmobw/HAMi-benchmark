# Host Mini Benchmark

This baseline runs the same `mini` workloads directly from the host with Docker and `nvidia-smi`.

It does not use:

- Kubernetes
- HAMi
- the external scheduler

It uses the same workload entrypoint as the Kubernetes mini benchmark:

- [`mini_workload.py`](/home/ttk/project/HAMi-benchmark/mini/workloads/mini_workload.py)

## What It Measures

The host baseline writes a comparable set of outputs under:

- [`logs`](/home/ttk/project/HAMi-benchmark/host-mini/logs)

Each run writes:

- `run.log`
- `metrics_samples.csv`
- `tasks.json`
- `summary.json`
- `summary.txt`

## Run

```bash
/home/ttk/project/HAMi-benchmark/host-mini/run-host-mini
```

## Notes

- Tasks are submitted as one batch with a `3s` logical stagger, then executed sequentially on the host.
- GPU metrics are sampled from `nvidia-smi`.
- Containers are run with `docker run --rm --gpus all`.
