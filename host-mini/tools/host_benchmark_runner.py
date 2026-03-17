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
WORKLOAD_PATH = Path("/home/ttk/project/HAMi-benchmark/mini/workloads/mini_workload.py")


@dataclass
class TaskSpec:
    name: str
    workload_kind: str
    runtime_seconds: int
    reserve_mib: int
    batch_size: int


@dataclass
class TaskResult:
    name: str
    workload_kind: str
    submitted_at: str
    start_at: str | None
    finish_at: str | None
    return_code: int | None
    queue_delay_seconds: float | None
    runtime_seconds: float | None
    completion_time_seconds: float | None


TASKS = [
    TaskSpec("mini-background-training", "background-training", 180, 2200, 64),
    TaskSpec("mini-medium-inference-1", "medium-inference", 60, 1200, 8),
    TaskSpec("mini-short-inference-1", "short-inference", 20, 400, 256),
    TaskSpec("mini-short-inference-2", "short-inference", 20, 400, 256),
    TaskSpec("mini-short-inference-3", "short-inference", 20, 400, 256),
]


def handle_signal(_signum: int, _frame: object) -> None:
    global STOP
    STOP = True


signal.signal(signal.SIGINT, handle_signal)
signal.signal(signal.SIGTERM, handle_signal)


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def utc_now_iso() -> str:
    return utc_now().isoformat()


def format_dt(value: datetime | None) -> str | None:
    return value.isoformat() if value else None


def require_command(name: str) -> None:
    subprocess.run([name, "--version"], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def require_nvidia_smi() -> None:
    subprocess.run(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def query_gpu_total_mib() -> float:
    output = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
        text=True,
    )
    values = [float(line.strip()) for line in output.splitlines() if line.strip()]
    return sum(values)


def sample_gpu() -> tuple[float, float, float]:
    output = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits"],
        text=True,
    )
    gpu_utils = []
    mem_used = []
    for line in output.splitlines():
        if not line.strip():
            continue
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 2:
            continue
        gpu_utils.append(float(parts[0]))
        mem_used.append(float(parts[1]))
    return (max(gpu_utils) if gpu_utils else 0.0, sum(mem_used), 0.0)


def sampler_loop(csv_path: Path, gpu_total_mib: float, interval_seconds: float, stop_event: threading.Event) -> None:
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            [
                "timestamp_utc",
                "gpu_util_percent",
                "gpu_mem_used_mib",
                "gpu_mem_util_percent",
            ]
        )
        fh.flush()

        while not stop_event.is_set():
            try:
                gpu_util, mem_used_mib, _ = sample_gpu()
                mem_util = (mem_used_mib / gpu_total_mib * 100.0) if gpu_total_mib > 0 else 0.0
                writer.writerow([utc_now_iso(), gpu_util, mem_used_mib, mem_util])
            except Exception as exc:  # noqa: BLE001
                writer.writerow([utc_now_iso(), f"error:{exc}", "", ""])
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
        "-e",
        f"WORKLOAD_KIND={task.workload_kind}",
        "-e",
        f"WORKLOAD_RUNTIME_SECONDS={task.runtime_seconds}",
        "-e",
        f"WORKLOAD_RESERVE_MIB={task.reserve_mib}",
        "-e",
        f"WORKLOAD_BATCH_SIZE={task.batch_size}",
        "-v",
        f"{workload_dir}:/workloads:ro",
        IMAGE,
        "python",
        "/workloads/mini_workload.py",
    ]


def run_benchmark() -> int:
    require_command("docker")
    require_nvidia_smi()

    root_dir = Path(__file__).resolve().parents[1]
    log_dir = root_dir / "logs" / f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-host-direct"
    log_dir.mkdir(parents=True, exist_ok=True)

    gpu_total_mib = query_gpu_total_mib()
    run_log_path = log_dir / "run.log"
    tasks_json_path = log_dir / "tasks.json"
    metrics_csv_path = log_dir / "metrics_samples.csv"

    stop_event = threading.Event()
    sampler = threading.Thread(
        target=sampler_loop,
        args=(metrics_csv_path, gpu_total_mib, 2.0, stop_event),
        daemon=True,
    )
    sampler.start()

    results: list[TaskResult] = []
    run_start = utc_now()
    submit_stagger_seconds = 3

    with run_log_path.open("w", encoding="utf-8") as run_log:
        def log(message: str) -> None:
            print(message)
            run_log.write(message + "\n")
            run_log.flush()

        log("==> Host mini benchmark mode: host-direct")
        log(f"==> Image: {IMAGE}")
        log(f"==> Workload path: {WORKLOAD_PATH}")
        log(f"==> GPU total MiB: {gpu_total_mib}")
        log(f"==> Log dir: {log_dir}")

        try:
            for index, task in enumerate(TASKS):
                submitted_at = run_start + timedelta(seconds=index * submit_stagger_seconds)
                start_at = utc_now()
                queue_delay = max(0.0, (start_at - submitted_at).total_seconds())

                command = docker_command(task)
                log(f"==> Starting task: {task.name}")
                log("    " + " ".join(command))

                process = subprocess.run(command, stdout=run_log, stderr=run_log, text=True)
                finish_at = utc_now()
                runtime_seconds = (finish_at - start_at).total_seconds()
                completion_seconds = (finish_at - submitted_at).total_seconds()

                results.append(
                    TaskResult(
                        name=task.name,
                        workload_kind=task.workload_kind,
                        submitted_at=format_dt(submitted_at),
                        start_at=format_dt(start_at),
                        finish_at=format_dt(finish_at),
                        return_code=process.returncode,
                        queue_delay_seconds=queue_delay,
                        runtime_seconds=runtime_seconds,
                        completion_time_seconds=completion_seconds,
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
    summarize(log_dir=log_dir, gpu_total_mib=gpu_total_mib, results=results)
    return 0


def summarize(log_dir: Path, gpu_total_mib: float, results: list[TaskResult]) -> None:
    metrics_csv = log_dir / "metrics_samples.csv"
    gpu_util_samples = []
    gpu_mem_used_samples = []
    gpu_mem_util_samples = []

    with metrics_csv.open("r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            try:
                gpu_util_samples.append(float(row["gpu_util_percent"]))
                gpu_mem_used_samples.append(float(row["gpu_mem_used_mib"]))
                gpu_mem_util_samples.append(float(row["gpu_mem_util_percent"]))
            except (ValueError, KeyError):
                continue

    completed = [result for result in results if result.return_code == 0 and result.finish_at]
    completion_times = [result.completion_time_seconds for result in completed if result.completion_time_seconds is not None]
    runtimes = [result.runtime_seconds for result in completed if result.runtime_seconds is not None]
    queue_delays = [result.queue_delay_seconds for result in completed if result.queue_delay_seconds is not None]

    summary = {
        "mode": "host-direct",
        "metrics_source": "nvidia-smi",
        "gpu_total_mib": gpu_total_mib,
        "task_count": len(results),
        "completed_task_count": len(completed),
        "average_completion_time_seconds": statistics.mean(completion_times) if completion_times else None,
        "average_runtime_seconds": statistics.mean(runtimes) if runtimes else None,
        "average_queue_delay_seconds": statistics.mean(queue_delays) if queue_delays else None,
        "average_gpu_utilization_percent": statistics.mean(gpu_util_samples) if gpu_util_samples else None,
        "average_gpu_memory_utilization_percent": statistics.mean(gpu_mem_util_samples) if gpu_mem_util_samples else None,
        "average_gpu_memory_used_mib": statistics.mean(gpu_mem_used_samples) if gpu_mem_used_samples else None,
        "max_gpu_utilization_percent": max(gpu_util_samples) if gpu_util_samples else None,
        "max_gpu_memory_utilization_percent": max(gpu_mem_util_samples) if gpu_mem_util_samples else None,
        "tasks": [asdict(result) for result in results],
    }

    (log_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = [
        "mode: host-direct",
        "metrics_source: nvidia-smi",
        f"gpu_total_mib: {gpu_total_mib}",
        f"task_count: {summary['task_count']}",
        f"completed_task_count: {summary['completed_task_count']}",
        f"average_completion_time_seconds: {summary['average_completion_time_seconds']}",
        f"average_runtime_seconds: {summary['average_runtime_seconds']}",
        f"average_queue_delay_seconds: {summary['average_queue_delay_seconds']}",
        f"average_gpu_utilization_percent: {summary['average_gpu_utilization_percent']}",
        f"average_gpu_memory_utilization_percent: {summary['average_gpu_memory_utilization_percent']}",
        f"average_gpu_memory_used_mib: {summary['average_gpu_memory_used_mib']}",
        f"max_gpu_utilization_percent: {summary['max_gpu_utilization_percent']}",
        f"max_gpu_memory_utilization_percent: {summary['max_gpu_memory_utilization_percent']}",
    ]
    (log_dir / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def main() -> int:
    if len(sys.argv) < 2 or sys.argv[1] != "run":
        print("Usage: host_benchmark_runner.py run", file=sys.stderr)
        return 1
    return run_benchmark()


if __name__ == "__main__":
    sys.exit(main())
