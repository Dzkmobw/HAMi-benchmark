from __future__ import annotations

import argparse
import csv
import json
import os
import signal
import statistics
import subprocess
import sys
import time
import urllib.request
from datetime import datetime, timezone


STOP = False


def handle_signal(_signum: int, _frame: object) -> None:
    global STOP
    STOP = True


signal.signal(signal.SIGINT, handle_signal)
signal.signal(signal.SIGTERM, handle_signal)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def run_kubectl_json(namespace: str, resource: str, selector: str) -> dict:
    cmd = [
        "kubectl",
        "get",
        resource,
        "-n",
        namespace,
        "-l",
        selector,
        "-o",
        "json",
    ]
    output = subprocess.check_output(cmd, text=True)
    return json.loads(output)


def parse_metrics_text(text: str, gpu_total_mib: float) -> tuple[float, float, float]:
    gpu_util_values = []
    gpu_mem_used_mib = 0.0

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("HostCoreUtilization"):
            try:
                gpu_util_values.append(float(line.split()[-1]))
            except ValueError:
                continue
        elif line.startswith("HostGPUMemoryUsage"):
            try:
                gpu_mem_used_mib += float(line.split()[-1]) / 1024 / 1024
            except ValueError:
                continue

    gpu_util = max(gpu_util_values) if gpu_util_values else 0.0
    gpu_mem_util = (gpu_mem_used_mib / gpu_total_mib * 100.0) if gpu_total_mib > 0 else 0.0
    return gpu_util, gpu_mem_used_mib, gpu_mem_util


def sample_loop(namespace: str, selector: str, metrics_endpoint: str, gpu_total_mib: float, log_dir: str, interval: float) -> int:
    os.makedirs(log_dir, exist_ok=True)
    csv_path = os.path.join(log_dir, "metrics_samples.csv")

    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
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

        while not STOP:
            try:
                with urllib.request.urlopen(metrics_endpoint, timeout=3) as response:
                    text = response.read().decode("utf-8")
                gpu_util, gpu_mem_used_mib, gpu_mem_util = parse_metrics_text(text, gpu_total_mib)
                writer.writerow([utc_now_iso(), gpu_util, gpu_mem_used_mib, gpu_mem_util])
                fh.flush()
            except Exception as exc:  # noqa: BLE001
                writer.writerow([utc_now_iso(), f"error:{exc}", "", ""])
                fh.flush()
            time.sleep(interval)

    return 0


def parse_time(raw: str | None) -> datetime | None:
    if not raw:
        return None
    return datetime.fromisoformat(raw.replace("Z", "+00:00"))


def summarize_runs(namespace: str, selector: str, metrics_endpoint: str, gpu_total_mib: float, log_dir: str, mode: str) -> int:
    os.makedirs(log_dir, exist_ok=True)
    pods_json = run_kubectl_json(namespace, "pods", selector)
    jobs_json = run_kubectl_json(namespace, "jobs", selector)

    with open(os.path.join(log_dir, "pods.json"), "w", encoding="utf-8") as fh:
        json.dump(pods_json, fh, indent=2)
    with open(os.path.join(log_dir, "jobs.json"), "w", encoding="utf-8") as fh:
        json.dump(jobs_json, fh, indent=2)

    metrics_csv = os.path.join(log_dir, "metrics_samples.csv")
    gpu_util_samples = []
    gpu_mem_util_samples = []
    gpu_mem_used_samples = []

    if os.path.exists(metrics_csv):
        with open(metrics_csv, "r", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                try:
                    gpu_util_samples.append(float(row["gpu_util_percent"]))
                    gpu_mem_used_samples.append(float(row["gpu_mem_used_mib"]))
                    gpu_mem_util_samples.append(float(row["gpu_mem_util_percent"]))
                except (ValueError, KeyError):
                    continue

    completed_elapsed = []
    completed_runtime = []
    queue_delay = []
    pod_summaries = []
    benchmark_start = None
    benchmark_finish = None

    for item in pods_json.get("items", []):
        meta = item.get("metadata", {})
        status = item.get("status", {})
        phase = status.get("phase", "Unknown")
        creation = parse_time(meta.get("creationTimestamp"))
        start = parse_time(status.get("startTime"))
        finished = None

        for container_status in status.get("containerStatuses", []) or []:
            terminated = ((container_status.get("state") or {}).get("terminated")) or {}
            if terminated.get("finishedAt"):
                finished = parse_time(terminated.get("finishedAt"))
                break

        elapsed_seconds = None
        runtime_seconds = None
        queue_seconds = None

        if creation and finished:
            elapsed_seconds = (finished - creation).total_seconds()
            completed_elapsed.append(elapsed_seconds)
            if benchmark_start is None or creation < benchmark_start:
                benchmark_start = creation
            if benchmark_finish is None or finished > benchmark_finish:
                benchmark_finish = finished
        if start and finished:
            runtime_seconds = (finished - start).total_seconds()
            completed_runtime.append(runtime_seconds)
        if creation and start:
            queue_seconds = (start - creation).total_seconds()
            queue_delay.append(queue_seconds)

        pod_summaries.append(
            {
                "name": meta.get("name"),
                "phase": phase,
                "creation_timestamp": meta.get("creationTimestamp"),
                "start_time": status.get("startTime"),
                "finish_time": finished.isoformat() if finished else None,
                "elapsed_submission_to_completion_seconds": elapsed_seconds,
                "runtime_seconds": runtime_seconds,
                "queue_delay_seconds": queue_seconds,
            }
        )

    summary = {
        "mode": mode,
        "namespace": namespace,
        "selector": selector,
        "metrics_endpoint": metrics_endpoint,
        "gpu_total_mib": gpu_total_mib,
        "job_count": len(jobs_json.get("items", [])),
        "pod_count": len(pods_json.get("items", [])),
        "completed_pod_count": len(completed_elapsed),
        "total_benchmark_wall_time_seconds": (
            (benchmark_finish - benchmark_start).total_seconds()
            if benchmark_start and benchmark_finish
            else None
        ),
        "max_task_completion_time_seconds": max(completed_elapsed) if completed_elapsed else None,
        "average_completion_time_seconds": statistics.mean(completed_elapsed) if completed_elapsed else None,
        "average_runtime_seconds": statistics.mean(completed_runtime) if completed_runtime else None,
        "average_queue_delay_seconds": statistics.mean(queue_delay) if queue_delay else None,
        "average_gpu_utilization_percent": statistics.mean(gpu_util_samples) if gpu_util_samples else None,
        "average_gpu_memory_utilization_percent": statistics.mean(gpu_mem_util_samples) if gpu_mem_util_samples else None,
        "average_gpu_memory_used_mib": statistics.mean(gpu_mem_used_samples) if gpu_mem_used_samples else None,
        "max_gpu_utilization_percent": max(gpu_util_samples) if gpu_util_samples else None,
        "max_gpu_memory_utilization_percent": max(gpu_mem_util_samples) if gpu_mem_util_samples else None,
        "pods": pod_summaries,
    }

    with open(os.path.join(log_dir, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    lines = [
        f"mode: {mode}",
        f"namespace: {namespace}",
        f"selector: {selector}",
        f"metrics_endpoint: {metrics_endpoint}",
        f"job_count: {summary['job_count']}",
        f"pod_count: {summary['pod_count']}",
        f"completed_pod_count: {summary['completed_pod_count']}",
        f"total_benchmark_wall_time_seconds: {summary['total_benchmark_wall_time_seconds']}",
        f"max_task_completion_time_seconds: {summary['max_task_completion_time_seconds']}",
        f"average_completion_time_seconds: {summary['average_completion_time_seconds']}",
        f"average_runtime_seconds: {summary['average_runtime_seconds']}",
        f"average_queue_delay_seconds: {summary['average_queue_delay_seconds']}",
        f"average_gpu_utilization_percent: {summary['average_gpu_utilization_percent']}",
        f"average_gpu_memory_utilization_percent: {summary['average_gpu_memory_utilization_percent']}",
        f"average_gpu_memory_used_mib: {summary['average_gpu_memory_used_mib']}",
        f"max_gpu_utilization_percent: {summary['max_gpu_utilization_percent']}",
        f"max_gpu_memory_utilization_percent: {summary['max_gpu_memory_utilization_percent']}",
    ]
    with open(os.path.join(log_dir, "summary.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")

    print("\n".join(lines))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    sample = subparsers.add_parser("sample")
    sample.add_argument("--namespace", required=True)
    sample.add_argument("--selector", required=True)
    sample.add_argument("--metrics-endpoint", required=True)
    sample.add_argument("--gpu-total-mib", type=float, default=12288.0)
    sample.add_argument("--log-dir", required=True)
    sample.add_argument("--interval-seconds", type=float, default=2.0)

    summarize = subparsers.add_parser("summarize")
    summarize.add_argument("--namespace", required=True)
    summarize.add_argument("--selector", required=True)
    summarize.add_argument("--metrics-endpoint", required=True)
    summarize.add_argument("--gpu-total-mib", type=float, default=12288.0)
    summarize.add_argument("--log-dir", required=True)
    summarize.add_argument("--mode", required=True)

    args = parser.parse_args()

    if args.command == "sample":
        return sample_loop(
            namespace=args.namespace,
            selector=args.selector,
            metrics_endpoint=args.metrics_endpoint,
            gpu_total_mib=args.gpu_total_mib,
            log_dir=args.log_dir,
            interval=args.interval_seconds,
        )

    if args.command == "summarize":
        return summarize_runs(
            namespace=args.namespace,
            selector=args.selector,
            metrics_endpoint=args.metrics_endpoint,
            gpu_total_mib=args.gpu_total_mib,
            log_dir=args.log_dir,
            mode=args.mode,
        )

    return 1


if __name__ == "__main__":
    sys.exit(main())
