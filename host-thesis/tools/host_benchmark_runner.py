from __future__ import annotations

import csv
import json
import os
import signal
import statistics
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path


STOP = False
IMAGE = "pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime"
WORKLOAD_PATH = Path("/home/ttk/project/HAMi-benchmark/thesis/workloads/thesis_workload.py")
TRAINING_ACCURACY_THRESHOLD = float(os.environ.get("TRAINING_ACCURACY_THRESHOLD", "0.5"))
WORKLOAD_DATA_ROOT_HOST = Path(os.environ.get("WORKLOAD_DATA_ROOT_HOST", "/mnt/data/benchmark-data"))


@dataclass
class TaskSpec:
    name: str
    workload_kind: str
    runtime_seconds: int
    reserve_mib: int
    batch_size: int
    priority_tier: str
    submit_offset_seconds: int


@dataclass
class TaskResult:
    name: str
    workload_kind: str
    priority_tier: str
    submitted_at: str
    start_at: str | None
    finish_at: str | None
    return_code: int | None
    queue_delay_seconds: float | None
    runtime_seconds: float | None
    completion_time_seconds: float | None
    result_payload: dict | None
    training_epoch_metrics: list[dict] | None
    training_time_to_accuracy_threshold_seconds: float | None


TASKS = [
    TaskSpec("thesis-background-cnn-training", "background-cnn-training", 120, 3200, 96, "opportunistic", 0),
    TaskSpec("thesis-text-embedding-inference-1", "text-embedding-inference", 25, 1000, 128, "production", 15),
    TaskSpec("thesis-vision-inference-1", "vision-inference", 45, 1800, 48, "production", 30),
    TaskSpec("thesis-text-embedding-inference-2", "text-embedding-inference", 25, 1000, 128, "production", 45),
]


def handle_signal(_signum: int, _frame: object) -> None:
    global STOP
    STOP = True


signal.signal(signal.SIGINT, handle_signal)
signal.signal(signal.SIGTERM, handle_signal)


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def format_dt(value: datetime | None) -> str | None:
    return value.isoformat() if value else None


def require_command(name: str) -> None:
    subprocess.run([name, "--version"], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def require_nvidia_smi() -> None:
    subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def query_gpu_total_mib() -> float:
    output = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"], text=True)
    return sum(float(line.strip()) for line in output.splitlines() if line.strip())


def sample_gpu() -> tuple[float, float]:
    output = subprocess.check_output(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits"], text=True)
    gpu_utils = []
    mem_used = []
    for line in output.splitlines():
        if not line.strip():
            continue
        util, mem = [part.strip() for part in line.split(",")]
        gpu_utils.append(float(util))
        mem_used.append(float(mem))
    return max(gpu_utils) if gpu_utils else 0.0, sum(mem_used)


def sampler_loop(csv_path: Path, gpu_total_mib: float, interval_seconds: float, stop_event: threading.Event) -> None:
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["timestamp_utc", "gpu_util_percent", "gpu_mem_used_mib", "gpu_mem_util_percent"])
        fh.flush()
        while not stop_event.is_set():
            try:
                gpu_util, mem_used_mib = sample_gpu()
                mem_util = (mem_used_mib / gpu_total_mib * 100.0) if gpu_total_mib > 0 else 0.0
                writer.writerow([utc_now().isoformat(), gpu_util, mem_used_mib, mem_util])
            except Exception as exc:  # noqa: BLE001
                writer.writerow([utc_now().isoformat(), f"error:{exc}", "", ""])
            fh.flush()
            stop_event.wait(interval_seconds)


def docker_command(task: TaskSpec) -> list[str]:
    workload_dir = str(WORKLOAD_PATH.parent)
    return [
        "docker",
        "run",
        "--rm",
        "--gpus",
        "all",
        "--shm-size",
        "2g",
        "-e",
        f"WORKLOAD_TASK_NAME={task.name}",
        "-e",
        f"WORKLOAD_PRIORITY_TIER={task.priority_tier}",
        "-e",
        f"WORKLOAD_KIND={task.workload_kind}",
        "-e",
        f"WORKLOAD_RUNTIME_SECONDS={task.runtime_seconds}",
        "-e",
        f"WORKLOAD_RESERVE_MIB={task.reserve_mib}",
        "-e",
        f"WORKLOAD_BATCH_SIZE={task.batch_size}",
        "-e",
        "WORKLOAD_DATA_ROOT=/data",
        "-e",
        "TORCH_HOME=/data/torch-cache",
        "-e",
        "WORKLOAD_USE_PRETRAINED_VISION=1",
        "-v",
        f"{workload_dir}:/workloads:ro",
        "-v",
        f"{WORKLOAD_DATA_ROOT_HOST}:/data",
        IMAGE,
        "python",
        "/workloads/thesis_workload.py",
    ]


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    index = (len(ordered) - 1) * q
    lo = int(index)
    hi = min(lo + 1, len(ordered) - 1)
    frac = index - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


def extract_result_payload(output: str) -> dict | None:
    for line in reversed(output.splitlines()):
        if line.startswith("RESULT_JSON:"):
            try:
                return json.loads(line.split("RESULT_JSON:", 1)[1])
            except json.JSONDecodeError:
                return None
    return None


def extract_training_epoch_metrics(output: str) -> list[dict]:
    metrics: list[dict] = []
    for line in output.splitlines():
        if not line.startswith("TRAIN_EPOCH_JSON:"):
            continue
        try:
            metrics.append(json.loads(line.split("TRAIN_EPOCH_JSON:", 1)[1]))
        except json.JSONDecodeError:
            continue
    return metrics


def time_to_accuracy_threshold(epoch_metrics: list[dict], threshold: float) -> float | None:
    for metric in epoch_metrics:
        accuracy = metric.get("epoch_average_accuracy")
        elapsed = metric.get("elapsed_seconds")
        if accuracy is None or elapsed is None:
            continue
        if float(accuracy) >= threshold:
            return float(elapsed)
    return None


def run_benchmark() -> int:
    require_command("docker")
    require_nvidia_smi()
    WORKLOAD_DATA_ROOT_HOST.mkdir(parents=True, exist_ok=True)
    (WORKLOAD_DATA_ROOT_HOST / "torch-cache").mkdir(parents=True, exist_ok=True)

    root_dir = Path(__file__).resolve().parents[1]
    log_dir = root_dir / "logs" / f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-host-direct"
    log_dir.mkdir(parents=True, exist_ok=True)

    gpu_total_mib = query_gpu_total_mib()
    run_log_path = log_dir / "run.log"
    tasks_json_path = log_dir / "tasks.json"
    metrics_csv_path = log_dir / "metrics_samples.csv"

    stop_event = threading.Event()
    sampler = threading.Thread(target=sampler_loop, args=(metrics_csv_path, gpu_total_mib, 2.0, stop_event), daemon=True)
    sampler.start()

    results: list[TaskResult] = []
    run_start = utc_now()

    with run_log_path.open("w", encoding="utf-8") as run_log:
        def log(message: str) -> None:
            print(message)
            run_log.write(message + "\n")
            run_log.flush()

        log("==> Host thesis benchmark mode: static-exclusive-host")
        log(f"==> Image: {IMAGE}")
        log(f"==> Workload path: {WORKLOAD_PATH}")
        log(f"==> GPU total MiB: {gpu_total_mib}")
        log(f"==> Log dir: {log_dir}")

        try:
            for task in TASKS:
                submitted_at = run_start + timedelta(seconds=task.submit_offset_seconds)
                start_at = utc_now()
                queue_delay = max(0.0, (start_at - submitted_at).total_seconds())

                command = docker_command(task)
                log(f"==> Starting task: {task.name}")
                log("    " + " ".join(command))

                process = subprocess.run(command, text=True, capture_output=True)
                if process.stdout:
                  run_log.write(process.stdout)
                if process.stderr:
                  run_log.write(process.stderr)
                run_log.flush()

                finish_at = utc_now()
                runtime_seconds = (finish_at - start_at).total_seconds()
                completion_seconds = (finish_at - submitted_at).total_seconds()
                result_payload = extract_result_payload(process.stdout or "")
                epoch_metrics = extract_training_epoch_metrics(process.stdout or "")
                threshold_time = time_to_accuracy_threshold(epoch_metrics, TRAINING_ACCURACY_THRESHOLD)

                results.append(
                    TaskResult(
                        name=task.name,
                        workload_kind=task.workload_kind,
                        priority_tier=task.priority_tier,
                        submitted_at=format_dt(submitted_at),
                        start_at=format_dt(start_at),
                        finish_at=format_dt(finish_at),
                        return_code=process.returncode,
                        queue_delay_seconds=queue_delay,
                        runtime_seconds=runtime_seconds,
                        completion_time_seconds=completion_seconds,
                        result_payload=result_payload,
                        training_epoch_metrics=epoch_metrics,
                        training_time_to_accuracy_threshold_seconds=threshold_time,
                    )
                )

                log(
                    f"==> Finished task: {task.name} rc={process.returncode} "
                    f"queue={queue_delay:.1f}s runtime={runtime_seconds:.1f}s completion={completion_seconds:.1f}s"
                )

                if process.returncode != 0:
                    raise RuntimeError(f"task failed: {task.name}")
        finally:
            stop_event.set()
            sampler.join(timeout=5)

    tasks_json_path.write_text(json.dumps([asdict(result) for result in results], indent=2), encoding="utf-8")
    summarize(log_dir, gpu_total_mib, results)
    return 0


def summarize(log_dir: Path, gpu_total_mib: float, results: list[TaskResult]) -> None:
    metrics_csv = log_dir / "metrics_samples.csv"
    gpu_util_samples: list[float] = []
    gpu_mem_used_samples: list[float] = []
    gpu_mem_util_samples: list[float] = []
    with metrics_csv.open("r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            try:
                gpu_util_samples.append(float(row["gpu_util_percent"]))
                gpu_mem_used_samples.append(float(row["gpu_mem_used_mib"]))
                gpu_mem_util_samples.append(float(row["gpu_mem_util_percent"]))
            except (ValueError, KeyError):
                continue

    completion = [r.completion_time_seconds for r in results if r.completion_time_seconds is not None]
    runtime = [r.runtime_seconds for r in results if r.runtime_seconds is not None]
    queue = [r.queue_delay_seconds for r in results if r.queue_delay_seconds is not None]
    production_completion = [r.completion_time_seconds for r in results if r.priority_tier == "production" and r.completion_time_seconds is not None]
    production_queue = [r.queue_delay_seconds for r in results if r.priority_tier == "production" and r.queue_delay_seconds is not None]
    opportunistic_runtime = [r.runtime_seconds for r in results if r.priority_tier == "opportunistic" and r.runtime_seconds is not None]
    training_losses = [float(r.result_payload["final_loss"]) for r in results if r.result_payload and r.result_payload.get("final_loss") is not None]
    training_accuracies = [float(r.result_payload["final_accuracy"]) for r in results if r.result_payload and r.result_payload.get("final_accuracy") is not None]
    training_curves = [
        {
            "name": r.name,
            "workload": r.workload_kind,
            "tier": r.priority_tier,
            "epoch_metrics": r.training_epoch_metrics or [],
            "time_to_accuracy_threshold_seconds": r.training_time_to_accuracy_threshold_seconds,
        }
        for r in results
        if r.training_epoch_metrics
    ]
    training_time_to_threshold = [
        float(r.training_time_to_accuracy_threshold_seconds)
        for r in results
        if r.training_time_to_accuracy_threshold_seconds is not None
    ]
    total_processed_samples = sum(int(r.result_payload.get("processed_samples", 0)) for r in results if r.result_payload)

    benchmark_start = min(datetime.fromisoformat(r.submitted_at) for r in results if r.submitted_at)
    benchmark_finish = max(datetime.fromisoformat(r.finish_at) for r in results if r.finish_at)
    wall_time = (benchmark_finish - benchmark_start).total_seconds()

    summary = {
        "mode": "static-exclusive-host",
        "gpu_total_mib": gpu_total_mib,
        "job_count": len(results),
        "completed_pod_count": len(results),
        "failed_pod_count": len([r for r in results if r.return_code not in (0, None)]),
        "oom_count": 0,
        "total_benchmark_wall_time_seconds": wall_time,
        "max_task_completion_time_seconds": max(completion) if completion else None,
        "average_completion_time_seconds": statistics.mean(completion) if completion else None,
        "p95_completion_time_seconds": percentile(completion, 0.95),
        "p99_completion_time_seconds": percentile(completion, 0.99),
        "average_runtime_seconds": statistics.mean(runtime) if runtime else None,
        "average_queue_delay_seconds": statistics.mean(queue) if queue else None,
        "throughput_tasks_per_minute": (len(results) / wall_time * 60.0) if wall_time > 0 else None,
        "total_processed_samples": total_processed_samples,
        "aggregate_sample_throughput_per_second": (total_processed_samples / wall_time) if wall_time > 0 else None,
        "high_priority_average_completion_time_seconds": statistics.mean(production_completion) if production_completion else None,
        "high_priority_p95_completion_time_seconds": percentile(production_completion, 0.95),
        "high_priority_average_queue_delay_seconds": statistics.mean(production_queue) if production_queue else None,
        "opportunistic_average_runtime_seconds": statistics.mean(opportunistic_runtime) if opportunistic_runtime else None,
        "average_training_final_loss": statistics.mean(training_losses) if training_losses else None,
        "average_training_final_accuracy": statistics.mean(training_accuracies) if training_accuracies else None,
        "training_accuracy_threshold": TRAINING_ACCURACY_THRESHOLD,
        "training_curve_task_count": len(training_curves),
        "training_accuracy_threshold_hit_count": len(training_time_to_threshold),
        "average_training_time_to_accuracy_threshold_seconds": statistics.mean(training_time_to_threshold) if training_time_to_threshold else None,
        "p95_training_time_to_accuracy_threshold_seconds": percentile(training_time_to_threshold, 0.95),
        "average_gpu_utilization_percent": statistics.mean(gpu_util_samples) if gpu_util_samples else None,
        "average_gpu_memory_utilization_percent": statistics.mean(gpu_mem_util_samples) if gpu_mem_util_samples else None,
        "average_gpu_memory_used_mib": statistics.mean(gpu_mem_used_samples) if gpu_mem_used_samples else None,
        "max_gpu_utilization_percent": max(gpu_util_samples) if gpu_util_samples else None,
        "max_gpu_memory_utilization_percent": max(gpu_mem_util_samples) if gpu_mem_util_samples else None,
        "tasks": [asdict(result) for result in results],
    }

    (log_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (log_dir / "training_curves.json").write_text(json.dumps(training_curves, indent=2), encoding="utf-8")
    lines = [
        f"mode: {summary['mode']}",
        f"job_count: {summary['job_count']}",
        f"completed_pod_count: {summary['completed_pod_count']}",
        f"total_benchmark_wall_time_seconds: {summary['total_benchmark_wall_time_seconds']}",
        f"average_completion_time_seconds: {summary['average_completion_time_seconds']}",
        f"p95_completion_time_seconds: {summary['p95_completion_time_seconds']}",
        f"average_queue_delay_seconds: {summary['average_queue_delay_seconds']}",
        f"throughput_tasks_per_minute: {summary['throughput_tasks_per_minute']}",
        f"aggregate_sample_throughput_per_second: {summary['aggregate_sample_throughput_per_second']}",
        f"high_priority_average_completion_time_seconds: {summary['high_priority_average_completion_time_seconds']}",
        f"high_priority_p95_completion_time_seconds: {summary['high_priority_p95_completion_time_seconds']}",
        f"average_training_final_loss: {summary['average_training_final_loss']}",
        f"average_training_final_accuracy: {summary['average_training_final_accuracy']}",
        f"training_accuracy_threshold: {summary['training_accuracy_threshold']}",
        f"training_curve_task_count: {summary['training_curve_task_count']}",
        f"training_accuracy_threshold_hit_count: {summary['training_accuracy_threshold_hit_count']}",
        f"average_training_time_to_accuracy_threshold_seconds: {summary['average_training_time_to_accuracy_threshold_seconds']}",
        f"p95_training_time_to_accuracy_threshold_seconds: {summary['p95_training_time_to_accuracy_threshold_seconds']}",
        f"average_gpu_utilization_percent: {summary['average_gpu_utilization_percent']}",
        f"average_gpu_memory_utilization_percent: {summary['average_gpu_memory_utilization_percent']}",
    ]
    (log_dir / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    sys.exit(run_benchmark())
