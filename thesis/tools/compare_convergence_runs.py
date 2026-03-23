from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def load_json(path: Path) -> dict | list:
    return json.loads(path.read_text(encoding="utf-8"))


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


def mean_or_none(values: list[float]) -> float | None:
    return statistics.mean(values) if values else None


def stdev_or_none(values: list[float]) -> float | None:
    return statistics.stdev(values) if len(values) >= 2 else None


def summarize_curve_points(curves: list[dict]) -> list[dict]:
    per_epoch: dict[int, dict[str, list[float]]] = {}
    for curve in curves:
        for metric in curve.get("epoch_metrics", []):
            epoch = int(metric.get("epoch", 0))
            slot = per_epoch.setdefault(
                epoch,
                {
                    "elapsed_seconds": [],
                    "epoch_average_loss": [],
                    "epoch_average_accuracy": [],
                    "smoothed_loss": [],
                    "smoothed_accuracy": [],
                },
            )
            for key in slot:
                value = metric.get(key)
                if value is not None:
                    slot[key].append(float(value))
    summary_points: list[dict] = []
    for epoch in sorted(per_epoch):
        slot = per_epoch[epoch]
        summary_points.append(
            {
                "epoch": epoch,
                "mean_elapsed_seconds": mean_or_none(slot["elapsed_seconds"]),
                "mean_epoch_average_loss": mean_or_none(slot["epoch_average_loss"]),
                "mean_epoch_average_accuracy": mean_or_none(slot["epoch_average_accuracy"]),
                "mean_smoothed_loss": mean_or_none(slot["smoothed_loss"]),
                "mean_smoothed_accuracy": mean_or_none(slot["smoothed_accuracy"]),
            }
        )
    return summary_points


def summarize_group(label: str, run_dirs: list[Path]) -> dict:
    final_losses: list[float] = []
    final_accuracies: list[float] = []
    threshold_times: list[float] = []
    all_curves: list[dict] = []
    threshold = None

    runs = []
    for run_dir in run_dirs:
        summary = load_json(run_dir / "summary.json")
        curves = load_json(run_dir / "training_curves.json")
        if isinstance(curves, list):
            all_curves.extend(curves)
        if summary.get("average_training_final_loss") is not None:
            final_losses.append(float(summary["average_training_final_loss"]))
        if summary.get("average_training_final_accuracy") is not None:
            final_accuracies.append(float(summary["average_training_final_accuracy"]))
        if summary.get("average_training_time_to_accuracy_threshold_seconds") is not None:
            threshold_times.append(float(summary["average_training_time_to_accuracy_threshold_seconds"]))
        if summary.get("training_accuracy_threshold") is not None:
            threshold = float(summary["training_accuracy_threshold"])
        runs.append(
            {
                "run_dir": str(run_dir),
                "final_loss": summary.get("average_training_final_loss"),
                "final_accuracy": summary.get("average_training_final_accuracy"),
                "time_to_accuracy_threshold_seconds": summary.get("average_training_time_to_accuracy_threshold_seconds"),
            }
        )

    return {
        "label": label,
        "run_count": len(run_dirs),
        "training_accuracy_threshold": threshold,
        "mean_final_loss": mean_or_none(final_losses),
        "stdev_final_loss": stdev_or_none(final_losses),
        "mean_final_accuracy": mean_or_none(final_accuracies),
        "stdev_final_accuracy": stdev_or_none(final_accuracies),
        "mean_time_to_accuracy_threshold_seconds": mean_or_none(threshold_times),
        "stdev_time_to_accuracy_threshold_seconds": stdev_or_none(threshold_times),
        "p95_time_to_accuracy_threshold_seconds": percentile(threshold_times, 0.95),
        "runs": runs,
        "mean_training_curve": summarize_curve_points(all_curves),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-label", default="baseline")
    parser.add_argument("--baseline-run", action="append", dest="baseline_runs", required=True)
    parser.add_argument("--method-label", default="method")
    parser.add_argument("--method-run", action="append", dest="method_runs", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    baseline = summarize_group(args.baseline_label, [Path(path) for path in args.baseline_runs])
    method = summarize_group(args.method_label, [Path(path) for path in args.method_runs])

    comparison = {
        "baseline": baseline,
        "method": method,
        "delta_final_loss": None
        if baseline["mean_final_loss"] is None or method["mean_final_loss"] is None
        else method["mean_final_loss"] - baseline["mean_final_loss"],
        "delta_final_accuracy": None
        if baseline["mean_final_accuracy"] is None or method["mean_final_accuracy"] is None
        else method["mean_final_accuracy"] - baseline["mean_final_accuracy"],
        "delta_time_to_accuracy_threshold_seconds": None
        if baseline["mean_time_to_accuracy_threshold_seconds"] is None or method["mean_time_to_accuracy_threshold_seconds"] is None
        else method["mean_time_to_accuracy_threshold_seconds"] - baseline["mean_time_to_accuracy_threshold_seconds"],
    }

    (output_dir / "convergence_comparison.json").write_text(json.dumps(comparison, indent=2), encoding="utf-8")

    lines = [
        f"baseline_label: {baseline['label']}",
        f"baseline_run_count: {baseline['run_count']}",
        f"baseline_mean_final_loss: {baseline['mean_final_loss']}",
        f"baseline_mean_final_accuracy: {baseline['mean_final_accuracy']}",
        f"baseline_mean_time_to_accuracy_threshold_seconds: {baseline['mean_time_to_accuracy_threshold_seconds']}",
        f"method_label: {method['label']}",
        f"method_run_count: {method['run_count']}",
        f"method_mean_final_loss: {method['mean_final_loss']}",
        f"method_mean_final_accuracy: {method['mean_final_accuracy']}",
        f"method_mean_time_to_accuracy_threshold_seconds: {method['mean_time_to_accuracy_threshold_seconds']}",
        f"delta_final_loss: {comparison['delta_final_loss']}",
        f"delta_final_accuracy: {comparison['delta_final_accuracy']}",
        f"delta_time_to_accuracy_threshold_seconds: {comparison['delta_time_to_accuracy_threshold_seconds']}",
    ]
    (output_dir / "convergence_comparison.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
