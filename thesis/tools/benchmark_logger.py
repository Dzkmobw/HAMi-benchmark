from __future__ import annotations

import argparse
import csv
import json
import os
import re
import signal
import statistics
import subprocess
import time
import urllib.request
from datetime import datetime, timezone


STOP = False
DEFAULT_TRAINING_ACCURACY_THRESHOLD = float(os.environ.get("TRAINING_ACCURACY_THRESHOLD", "0.5"))

SUMMARY_KEY_ORDER = [
    "mode",
    "namespace",
    "selector",
    "metrics_endpoint",
    "gpu_total_mib",
    "job_count",
    "pod_count",
    "completed_pod_count",
    "failed_pod_count",
    "oom_count",
    "total_benchmark_wall_time_seconds",
    "max_task_completion_time_seconds",
    "average_completion_time_seconds",
    "p95_completion_time_seconds",
    "p99_completion_time_seconds",
    "average_runtime_seconds",
    "average_queue_delay_seconds",
    "throughput_tasks_per_minute",
    "total_processed_samples",
    "aggregate_sample_throughput_per_second",
    "intercept_log_pod_count",
    "intercept_event_count",
    "pause_event_pod_count",
    "pause_event_count",
    "pause_total_waited_ms",
    "high_priority_average_completion_time_seconds",
    "high_priority_p95_completion_time_seconds",
    "high_priority_p99_completion_time_seconds",
    "high_priority_average_queue_delay_seconds",
    "high_priority_average_batch_latency_ms",
    "opportunistic_average_runtime_seconds",
    "average_training_final_loss",
    "average_training_final_accuracy",
    "training_best_accuracy",
    "training_accuracy_threshold",
    "training_curve_task_count",
    "training_accuracy_threshold_hit_count",
    "average_training_time_to_accuracy_threshold_seconds",
    "p95_training_time_to_accuracy_threshold_seconds",
    "average_gpu_utilization_percent",
    "average_gpu_memory_utilization_percent",
    "average_gpu_memory_used_mib",
    "max_gpu_utilization_percent",
    "max_gpu_memory_utilization_percent",
]

TASK_KEY_ORDER = [
    "name",
    "phase",
    "tier",
    "workload",
    "creation_timestamp",
    "start_time",
    "finish_time",
    "elapsed_submission_to_completion_seconds",
    "runtime_seconds",
    "queue_delay_seconds",
    "terminated_reason",
    "result_payload",
    "training_epoch_metrics",
    "training_epoch_count",
    "training_time_to_accuracy_threshold_seconds",
    "intercept_log_detected",
    "intercept_event_count",
    "pause_event_detected",
    "pause_event_count",
    "pause_events",
    "pause_total_waited_ms",
]

TRAINING_CURVE_KEY_ORDER = [
    "name",
    "workload",
    "tier",
    "epoch_metrics",
    "best_accuracy",
    "time_to_accuracy_threshold_seconds",
]


def handle_signal(_signum: int, _frame: object) -> None:
    global STOP
    STOP = True


signal.signal(signal.SIGINT, handle_signal)
signal.signal(signal.SIGTERM, handle_signal)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def prune_empty(value: object) -> object | None:
    if value is None:
        return None
    if isinstance(value, dict):
        cleaned: dict[str, object] = {}
        for key, item in value.items():
            pruned = prune_empty(item)
            if pruned is None:
                continue
            if isinstance(pruned, (dict, list)) and not pruned:
                continue
            cleaned[key] = pruned
        return cleaned
    if isinstance(value, list):
        cleaned_list = []
        for item in value:
            pruned = prune_empty(item)
            if pruned is None:
                continue
            if isinstance(pruned, (dict, list)) and not pruned:
                continue
            cleaned_list.append(pruned)
        return cleaned_list
    if isinstance(value, str) and value == "":
        return None
    return value


def order_dict(data: dict, key_order: list[str]) -> dict:
    ordered: dict[str, object] = {}
    for key in key_order:
        if key in data:
            ordered[key] = data[key]
    for key, value in data.items():
        if key not in ordered:
            ordered[key] = value
    return ordered


def run_kubectl_json(namespace: str, resource: str, selector: str) -> dict:
    output = subprocess.check_output(
        ["kubectl", "get", resource, "-n", namespace, "-l", selector, "-o", "json"],
        text=True,
    )
    return json.loads(output)


def run_kubectl_logs(namespace: str, pod_name: str) -> str:
    try:
        return subprocess.check_output(
            ["kubectl", "logs", pod_name, "-n", namespace],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except subprocess.CalledProcessError:
        return ""


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
    del namespace, selector
    os.makedirs(log_dir, exist_ok=True)
    csv_path = os.path.join(log_dir, "metrics_samples.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["timestamp_utc", "gpu_util_percent", "gpu_mem_used_mib", "gpu_mem_util_percent"])
        fh.flush()
        while not STOP:
            try:
                with urllib.request.urlopen(metrics_endpoint, timeout=3) as response:
                    text = response.read().decode("utf-8")
                gpu_util, gpu_mem_used_mib, gpu_mem_util = parse_metrics_text(text, gpu_total_mib)
                writer.writerow([utc_now_iso(), gpu_util, gpu_mem_used_mib, gpu_mem_util])
            except Exception as exc:  # noqa: BLE001
                writer.writerow([utc_now_iso(), f"error:{exc}", "", ""])
            fh.flush()
            time.sleep(interval)
    return 0


def parse_time(raw: str | None) -> datetime | None:
    if not raw:
        return None
    return datetime.fromisoformat(raw.replace("Z", "+00:00"))


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


def extract_result_payload(log_text: str) -> dict | None:
    for line in reversed(log_text.splitlines()):
        if line.startswith("RESULT_JSON:"):
            try:
                return json.loads(line.split("RESULT_JSON:", 1)[1])
            except json.JSONDecodeError:
                return None
    return None


def extract_intercept_log(log_text: str) -> list[str]:
    lines = log_text.splitlines()
    inside = False
    collected: list[str] = []
    for line in lines:
        if line.strip() == "CUDA_INTERCEPT_LOG_BEGIN":
            inside = True
            continue
        if line.strip() == "CUDA_INTERCEPT_LOG_END":
            break
        if inside:
            collected.append(line)
    if collected:
        return collected
    return [line for line in lines if line.startswith("[cuda-intercept]")]


def extract_training_epoch_metrics(log_text: str) -> list[dict]:
    metrics: list[dict] = []
    for line in log_text.splitlines():
        if not line.startswith("TRAIN_EPOCH_JSON:"):
            continue
        try:
            metrics.append(json.loads(line.split("TRAIN_EPOCH_JSON:", 1)[1]))
        except json.JSONDecodeError:
            continue
    return metrics


def extract_pause_events(intercept_log_lines: list[str]) -> list[str]:
    return [
        line
        for line in intercept_log_lines
        if "pause entered" in line
        or "pause released" in line
        or "pause_file detected" in line
        or "pause_file cleared" in line
    ]


def extract_pause_waited_ms(pause_events: list[str]) -> list[float]:
    waited_values: list[float] = []
    for line in pause_events:
        match = re.search(r"waited_ms=([0-9]+(?:\.[0-9]+)?)", line)
        if match is None:
            continue
        try:
            waited_values.append(float(match.group(1)))
        except ValueError:
            continue
    return waited_values


def time_to_accuracy_threshold(epoch_metrics: list[dict], threshold: float) -> float | None:
    for metric in epoch_metrics:
        accuracy = metric.get("epoch_average_accuracy")
        elapsed = metric.get("elapsed_seconds")
        if accuracy is None or elapsed is None:
            continue
        if float(accuracy) >= threshold:
            return float(elapsed)
    return None


def summarize_runs(namespace: str, selector: str, metrics_endpoint: str, gpu_total_mib: float, log_dir: str, mode: str) -> int:
    os.makedirs(log_dir, exist_ok=True)
    pods_json = run_kubectl_json(namespace, "pods", selector)
    jobs_json = run_kubectl_json(namespace, "jobs", selector)

    with open(os.path.join(log_dir, "pods.json"), "w", encoding="utf-8") as fh:
        json.dump(pods_json, fh, indent=2)
    with open(os.path.join(log_dir, "jobs.json"), "w", encoding="utf-8") as fh:
        json.dump(jobs_json, fh, indent=2)

    metrics_csv = os.path.join(log_dir, "metrics_samples.csv")
    gpu_util_samples: list[float] = []
    gpu_mem_util_samples: list[float] = []
    gpu_mem_used_samples: list[float] = []
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

    completed_elapsed: list[float] = []
    completed_runtime: list[float] = []
    queue_delay: list[float] = []
    production_completion: list[float] = []
    production_queue_delay: list[float] = []
    opportunistic_runtime: list[float] = []
    production_batch_latency_ms: list[float] = []
    task_results: list[dict] = []
    training_losses: list[float] = []
    training_accuracies: list[float] = []
    training_best_accuracies: list[float] = []
    training_time_to_threshold: list[float] = []
    training_curves: list[dict] = []
    total_processed_samples = 0
    intercept_log_pod_count = 0
    intercept_event_count = 0
    pause_event_pod_count = 0
    pause_event_count = 0
    benchmark_start = None
    benchmark_finish = None
    failed_pod_count = 0
    oom_count = 0

    for item in pods_json.get("items", []):
        meta = item.get("metadata", {})
        status = item.get("status", {})
        labels = meta.get("labels", {}) or {}
        phase = status.get("phase", "Unknown")
        creation = parse_time(meta.get("creationTimestamp"))
        start = parse_time(status.get("startTime"))
        finished = None
        terminated_reason = None

        for container_status in status.get("containerStatuses", []) or []:
            terminated = ((container_status.get("state") or {}).get("terminated")) or {}
            if terminated.get("finishedAt"):
                finished = parse_time(terminated.get("finishedAt"))
                terminated_reason = terminated.get("reason")
                break

        elapsed_seconds = None
        runtime_seconds = None
        queue_seconds = None
        if creation and finished:
            elapsed_seconds = (finished - creation).total_seconds()
            completed_elapsed.append(elapsed_seconds)
            benchmark_start = creation if benchmark_start is None or creation < benchmark_start else benchmark_start
            benchmark_finish = finished if benchmark_finish is None or finished > benchmark_finish else benchmark_finish
        if start and finished:
            runtime_seconds = (finished - start).total_seconds()
            completed_runtime.append(runtime_seconds)
        if creation and start:
            queue_seconds = (start - creation).total_seconds()
            queue_delay.append(queue_seconds)

        tier = labels.get("benchmark.hami.io/tier", "unknown")
        workload = labels.get("benchmark.hami.io/workload", "unknown")
        pod_log = run_kubectl_logs(namespace, meta.get("name", ""))
        result_payload = extract_result_payload(pod_log)
        intercept_log_lines = extract_intercept_log(pod_log)
        pause_events = extract_pause_events(intercept_log_lines)
        pause_waited_ms = extract_pause_waited_ms(pause_events)
        epoch_metrics = extract_training_epoch_metrics(pod_log)
        threshold_time = time_to_accuracy_threshold(epoch_metrics, DEFAULT_TRAINING_ACCURACY_THRESHOLD)

        if tier == "production" and elapsed_seconds is not None:
            production_completion.append(elapsed_seconds)
        if tier == "production" and queue_seconds is not None:
            production_queue_delay.append(queue_seconds)
        if tier == "opportunistic" and runtime_seconds is not None:
            opportunistic_runtime.append(runtime_seconds)
        if tier == "production" and result_payload and result_payload.get("avg_batch_latency_ms") is not None:
            production_batch_latency_ms.append(float(result_payload["avg_batch_latency_ms"]))

        if result_payload:
            total_processed_samples += int(result_payload.get("processed_samples", 0))
            if result_payload.get("final_loss") is not None:
                training_losses.append(float(result_payload["final_loss"]))
            if result_payload.get("final_accuracy") is not None:
                training_accuracies.append(float(result_payload["final_accuracy"]))
        if threshold_time is not None:
            training_time_to_threshold.append(threshold_time)
        if epoch_metrics:
            epoch_best_accuracy = max(
                float(metric.get("epoch_average_accuracy", 0.0))
                for metric in epoch_metrics
                if metric.get("epoch_average_accuracy") is not None
            )
            training_best_accuracies.append(epoch_best_accuracy)
            training_curves.append(
                order_dict(
                    prune_empty(
                        {
                            "name": meta.get("name"),
                            "workload": workload,
                            "tier": tier,
                            "epoch_metrics": epoch_metrics,
                            "best_accuracy": epoch_best_accuracy,
                            "time_to_accuracy_threshold_seconds": threshold_time,
                        }
                    )
                    or {},
                    TRAINING_CURVE_KEY_ORDER,
                )
            )
        if intercept_log_lines:
            intercept_log_pod_count += 1
            intercept_event_count += len(intercept_log_lines)
        if pause_events:
            pause_event_pod_count += 1
            pause_event_count += len(pause_events)

        if phase == "Failed":
            failed_pod_count += 1
        if terminated_reason == "OOMKilled" or "OutOfMemory" in pod_log or "CUDA out of memory" in pod_log:
            oom_count += 1

        task_results.append(
            order_dict(
                prune_empty(
                    {
                        "name": meta.get("name"),
                        "phase": phase,
                        "tier": tier,
                        "workload": workload,
                        "creation_timestamp": meta.get("creationTimestamp"),
                        "start_time": status.get("startTime"),
                        "finish_time": finished.isoformat() if finished else None,
                        "elapsed_submission_to_completion_seconds": elapsed_seconds,
                        "runtime_seconds": runtime_seconds,
                        "queue_delay_seconds": queue_seconds,
                        "terminated_reason": terminated_reason,
                        "result_payload": result_payload,
                        "training_epoch_metrics": epoch_metrics,
                        "training_epoch_count": len(epoch_metrics) if epoch_metrics else None,
                        "training_time_to_accuracy_threshold_seconds": threshold_time,
                        "intercept_log_detected": True if intercept_log_lines else None,
                        "intercept_event_count": len(intercept_log_lines) if intercept_log_lines else None,
                        "pause_event_detected": True if pause_events else None,
                        "pause_event_count": len(pause_events) if pause_events else None,
                        "pause_events": pause_events,
                        "pause_total_waited_ms": sum(pause_waited_ms) if pause_waited_ms else None,
                    }
                )
                or {},
                TASK_KEY_ORDER,
            )
        )

    wall_time = (benchmark_finish - benchmark_start).total_seconds() if benchmark_start and benchmark_finish else None
    aggregate_sample_throughput = (total_processed_samples / wall_time) if wall_time and wall_time > 0 else None

    summary = {
        "mode": mode,
        "namespace": namespace,
        "selector": selector,
        "metrics_endpoint": metrics_endpoint,
        "gpu_total_mib": gpu_total_mib,
        "job_count": len(jobs_json.get("items", [])),
        "pod_count": len(pods_json.get("items", [])),
        "completed_pod_count": len(completed_elapsed),
        "failed_pod_count": failed_pod_count,
        "oom_count": oom_count,
        "total_benchmark_wall_time_seconds": wall_time,
        "max_task_completion_time_seconds": max(completed_elapsed) if completed_elapsed else None,
        "average_completion_time_seconds": statistics.mean(completed_elapsed) if completed_elapsed else None,
        "p95_completion_time_seconds": percentile(completed_elapsed, 0.95),
        "p99_completion_time_seconds": percentile(completed_elapsed, 0.99),
        "average_runtime_seconds": statistics.mean(completed_runtime) if completed_runtime else None,
        "average_queue_delay_seconds": statistics.mean(queue_delay) if queue_delay else None,
        "throughput_tasks_per_minute": ((len(completed_elapsed) / wall_time) * 60.0) if wall_time and wall_time > 0 else None,
        "total_processed_samples": total_processed_samples,
        "aggregate_sample_throughput_per_second": aggregate_sample_throughput,
        "intercept_log_pod_count": intercept_log_pod_count,
        "intercept_event_count": intercept_event_count,
        "pause_event_pod_count": pause_event_pod_count,
        "pause_event_count": pause_event_count,
        "pause_total_waited_ms": sum(
            float(task["pause_total_waited_ms"]) for task in task_results if task.get("pause_total_waited_ms") is not None
        ),
        "high_priority_average_completion_time_seconds": statistics.mean(production_completion) if production_completion else None,
        "high_priority_p95_completion_time_seconds": percentile(production_completion, 0.95),
        "high_priority_p99_completion_time_seconds": percentile(production_completion, 0.99),
        "high_priority_average_queue_delay_seconds": statistics.mean(production_queue_delay) if production_queue_delay else None,
        "high_priority_average_batch_latency_ms": statistics.mean(production_batch_latency_ms) if production_batch_latency_ms else None,
        "opportunistic_average_runtime_seconds": statistics.mean(opportunistic_runtime) if opportunistic_runtime else None,
        "average_training_final_loss": statistics.mean(training_losses) if training_losses else None,
        "average_training_final_accuracy": statistics.mean(training_accuracies) if training_accuracies else None,
        "training_best_accuracy": max(training_best_accuracies) if training_best_accuracies else None,
        "training_accuracy_threshold": DEFAULT_TRAINING_ACCURACY_THRESHOLD,
        "training_curve_task_count": len(training_curves),
        "training_accuracy_threshold_hit_count": len(training_time_to_threshold),
        "average_training_time_to_accuracy_threshold_seconds": statistics.mean(training_time_to_threshold) if training_time_to_threshold else None,
        "p95_training_time_to_accuracy_threshold_seconds": percentile(training_time_to_threshold, 0.95),
        "average_gpu_utilization_percent": statistics.mean(gpu_util_samples) if gpu_util_samples else None,
        "average_gpu_memory_utilization_percent": statistics.mean(gpu_mem_util_samples) if gpu_mem_util_samples else None,
        "average_gpu_memory_used_mib": statistics.mean(gpu_mem_used_samples) if gpu_mem_used_samples else None,
        "max_gpu_utilization_percent": max(gpu_util_samples) if gpu_util_samples else None,
        "max_gpu_memory_utilization_percent": max(gpu_mem_util_samples) if gpu_mem_util_samples else None,
        "tasks": task_results,
    }

    summary = order_dict(prune_empty(summary) or {}, SUMMARY_KEY_ORDER)

    with open(os.path.join(log_dir, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    with open(os.path.join(log_dir, "task_results.json"), "w", encoding="utf-8") as fh:
        json.dump(task_results, fh, indent=2)
    with open(os.path.join(log_dir, "training_curves.json"), "w", encoding="utf-8") as fh:
        json.dump(training_curves, fh, indent=2)

    lines = [f"{key}: {summary[key]}" for key in SUMMARY_KEY_ORDER if key in summary]
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
    sample.add_argument("--interval-seconds", type=float, default=1.0)

    summarize = subparsers.add_parser("summarize")
    summarize.add_argument("--namespace", required=True)
    summarize.add_argument("--selector", required=True)
    summarize.add_argument("--metrics-endpoint", required=True)
    summarize.add_argument("--gpu-total-mib", type=float, default=12288.0)
    summarize.add_argument("--log-dir", required=True)
    summarize.add_argument("--mode", required=True)

    args = parser.parse_args()
    if args.command == "sample":
        return sample_loop(args.namespace, args.selector, args.metrics_endpoint, args.gpu_total_mib, args.log_dir, args.interval_seconds)
    return summarize_runs(args.namespace, args.selector, args.metrics_endpoint, args.gpu_total_mib, args.log_dir, args.mode)


if __name__ == "__main__":
    raise SystemExit(main())
