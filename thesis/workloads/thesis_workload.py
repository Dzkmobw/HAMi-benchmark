from __future__ import annotations

import json
import math
import os
import time
import traceback

import torch
import torch.nn as nn
import torch.nn.functional as F


def get_env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    return int(raw) if raw else default


def get_env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    return float(raw) if raw else default


def get_env_str(name: str, default: str) -> str:
    return os.environ.get(name, default)


def maybe_reserve_memory(device: torch.device, reserve_mib: int) -> torch.Tensor | None:
    if reserve_mib <= 0 or device.type != "cuda":
        return None
    elements = reserve_mib * 1024 * 1024 // 2
    return torch.empty(int(elements), device=device, dtype=torch.float16)


def emit_intercept_log() -> None:
    log_path = os.environ.get("CUDA_INTERCEPT_LOG_PATH", "").strip()
    if not log_path or not os.path.exists(log_path):
        return
    try:
        print("CUDA_INTERCEPT_LOG_BEGIN", flush=True)
        with open(log_path, "r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                print(line.rstrip("\n"), flush=True)
        print("CUDA_INTERCEPT_LOG_END", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"CUDA_INTERCEPT_LOG_ERROR:{exc}", flush=True)


def emit_training_epoch_metric(
    task_name: str,
    priority_tier: str,
    epoch_index: int,
    elapsed_seconds: float,
    epoch_batches: int,
    epoch_processed_samples: int,
    total_processed_samples: int,
    epoch_average_loss: float,
    epoch_average_accuracy: float,
    smoothed_loss: float,
    smoothed_accuracy: float,
) -> None:
    print(
        "TRAIN_EPOCH_JSON:"
        + json.dumps(
            {
                "task_name": task_name,
                "workload_kind": "background-cnn-training",
                "priority_tier": priority_tier,
                "epoch": epoch_index,
                "elapsed_seconds": elapsed_seconds,
                "epoch_batches": epoch_batches,
                "epoch_processed_samples": epoch_processed_samples,
                "total_processed_samples": total_processed_samples,
                "epoch_average_loss": epoch_average_loss,
                "epoch_average_accuracy": epoch_average_accuracy,
                "smoothed_loss": smoothed_loss,
                "smoothed_accuracy": smoothed_accuracy,
            },
            sort_keys=True,
        ),
        flush=True,
    )


def build_synthetic_class_templates(device: torch.device) -> torch.Tensor:
    templates = torch.zeros(10, 3, 64, 64, device=device)
    patch_size = 12
    accent_size = 8
    row_positions = [6, 24, 42]
    col_positions = [4, 16, 28, 40, 52]
    for class_id in range(10):
        row = row_positions[class_id // 5]
        col = col_positions[class_id % 5]
        primary_channel = class_id % 3
        accent_channel = (class_id + 1) % 3
        templates[class_id, primary_channel, row : row + patch_size, col - 4 : col + 8] = 1.0
        templates[class_id, accent_channel, row // 2 : row // 2 + accent_size, col - 2 : col + 6] = 0.45
        templates[class_id, :, row + patch_size : row + patch_size + 2, :] = class_id / 20.0
    return templates


def synthetic_training_batch(
    batch_size: int,
    device: torch.device,
    class_templates: torch.Tensor,
    noise_scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    labels = torch.randint(0, class_templates.shape[0], (batch_size,), device=device)
    images = class_templates.index_select(0, labels).clone()
    images.add_(torch.randn(batch_size, 3, 64, 64, device=device) * noise_scale)
    return images, labels


class TextEmbeddingModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(32000, 256)
        self.proj = nn.Sequential(
            nn.Linear(256, 512),
            nn.GELU(),
            nn.Linear(512, 384),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        pooled = self.embedding(tokens).mean(dim=1)
        return self.proj(pooled)


class VisionInferenceModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.head = nn.Linear(128, 1000)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        x = self.net(images).flatten(1)
        return self.head(x)


class TrainModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.head = nn.Linear(128, 10)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        x = self.net(images).flatten(1)
        return self.head(x)


def build_result(
    workload_kind: str,
    task_name: str,
    priority_tier: str,
    runtime_seconds: int,
    batches: int,
    processed_samples: int,
    started_at: float,
    final_loss: float | None = None,
    final_accuracy: float | None = None,
) -> dict:
    elapsed = max(time.monotonic() - started_at, 1e-6)
    return {
        "task_name": task_name,
        "workload_kind": workload_kind,
        "priority_tier": priority_tier,
        "configured_runtime_seconds": runtime_seconds,
        "elapsed_seconds": elapsed,
        "batches": batches,
        "processed_samples": processed_samples,
        "throughput_samples_per_second": processed_samples / elapsed,
        "avg_batch_latency_ms": elapsed / max(batches, 1) * 1000.0,
        "final_loss": final_loss,
        "final_accuracy": final_accuracy,
    }


def text_embedding_inference(device: torch.device, runtime_seconds: int, reserve_mib: int, batch_size: int, task_name: str, priority_tier: str) -> dict:
    model = TextEmbeddingModel().to(device).eval()
    _buffer = maybe_reserve_memory(device, reserve_mib)
    started = time.monotonic()
    batches = 0
    processed = 0
    with torch.inference_mode():
        while time.monotonic() - started < runtime_seconds:
            tokens = torch.randint(0, 32000, (batch_size, 64), device=device)
            _ = model(tokens)
            if device.type == "cuda":
                torch.cuda.synchronize()
            batches += 1
            processed += batch_size
    return build_result("text-embedding-inference", task_name, priority_tier, runtime_seconds, batches, processed, started)


def vision_inference(device: torch.device, runtime_seconds: int, reserve_mib: int, batch_size: int, task_name: str, priority_tier: str) -> dict:
    model = VisionInferenceModel().to(device).eval()
    _buffer = maybe_reserve_memory(device, reserve_mib)
    started = time.monotonic()
    batches = 0
    processed = 0
    with torch.inference_mode():
        while time.monotonic() - started < runtime_seconds:
            images = torch.randn(batch_size, 3, 224, 224, device=device)
            _ = model(images)
            if device.type == "cuda":
                torch.cuda.synchronize()
            batches += 1
            processed += batch_size
    return build_result("vision-inference", task_name, priority_tier, runtime_seconds, batches, processed, started)


def background_cnn_training(device: torch.device, runtime_seconds: int, reserve_mib: int, batch_size: int, task_name: str, priority_tier: str) -> dict:
    model = TrainModel().to(device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)
    _buffer = maybe_reserve_memory(device, reserve_mib)
    class_templates = build_synthetic_class_templates(device)
    noise_scale = get_env_float("WORKLOAD_TRAIN_NOISE_SCALE", 0.08)
    samples_per_epoch = get_env_int("WORKLOAD_TRAIN_SAMPLES_PER_EPOCH", 2048)
    batches_per_epoch = max(1, math.ceil(samples_per_epoch / max(batch_size, 1)))
    started = time.monotonic()
    batches = 0
    processed = 0
    loss_ema = 0.0
    acc_ema = 0.0
    epoch_index = 0
    epoch_batches = 0
    epoch_processed = 0
    epoch_loss_total = 0.0
    epoch_accuracy_total = 0.0
    while time.monotonic() - started < runtime_seconds:
        images, labels = synthetic_training_batch(batch_size, device, class_templates, noise_scale)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = F.cross_entropy(logits, labels)
        loss.backward()
        optimizer.step()
        if device.type == "cuda":
            torch.cuda.synchronize()
        with torch.no_grad():
            accuracy = (logits.argmax(dim=1) == labels).float().mean().item()
        batches += 1
        processed += batch_size
        epoch_batches += 1
        epoch_processed += batch_size
        epoch_loss_total += loss.item()
        epoch_accuracy_total += accuracy
        if batches == 1:
            loss_ema = loss.item()
            acc_ema = accuracy
        else:
            loss_ema = 0.9 * loss_ema + 0.1 * loss.item()
            acc_ema = 0.9 * acc_ema + 0.1 * accuracy
        if epoch_batches >= batches_per_epoch or time.monotonic() - started >= runtime_seconds:
            epoch_index += 1
            emit_training_epoch_metric(
                task_name=task_name,
                priority_tier=priority_tier,
                epoch_index=epoch_index,
                elapsed_seconds=time.monotonic() - started,
                epoch_batches=epoch_batches,
                epoch_processed_samples=epoch_processed,
                total_processed_samples=processed,
                epoch_average_loss=epoch_loss_total / max(epoch_batches, 1),
                epoch_average_accuracy=epoch_accuracy_total / max(epoch_batches, 1),
                smoothed_loss=loss_ema,
                smoothed_accuracy=acc_ema,
            )
            epoch_batches = 0
            epoch_processed = 0
            epoch_loss_total = 0.0
            epoch_accuracy_total = 0.0
    return build_result(
        "background-cnn-training",
        task_name,
        priority_tier,
        runtime_seconds,
        batches,
        processed,
        started,
        final_loss=loss_ema,
        final_accuracy=acc_ema,
    )


def main() -> int:
    workload_kind = get_env_str("WORKLOAD_KIND", "text-embedding-inference")
    runtime_seconds = get_env_int("WORKLOAD_RUNTIME_SECONDS", 30)
    reserve_mib = get_env_int("WORKLOAD_RESERVE_MIB", 0)
    batch_size = get_env_int("WORKLOAD_BATCH_SIZE", 32)
    task_name = get_env_str("WORKLOAD_TASK_NAME", workload_kind)
    priority_tier = get_env_str("WORKLOAD_PRIORITY_TIER", "production")
    seed = get_env_int("WORKLOAD_SEED", 20260323)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        f"[start] workload={workload_kind} runtime_seconds={runtime_seconds} "
        f"reserve_mib={reserve_mib} batch_size={batch_size} priority_tier={priority_tier} "
        f"seed={seed} device={device}",
        flush=True,
    )
    if device.type == "cuda":
        print(f"[gpu] name={torch.cuda.get_device_name(0)}", flush=True)

    try:
        if workload_kind == "text-embedding-inference":
            result = text_embedding_inference(device, runtime_seconds, reserve_mib, batch_size, task_name, priority_tier)
        elif workload_kind == "vision-inference":
            result = vision_inference(device, runtime_seconds, reserve_mib, batch_size, task_name, priority_tier)
        elif workload_kind == "background-cnn-training":
            result = background_cnn_training(device, runtime_seconds, reserve_mib, batch_size, task_name, priority_tier)
        else:
            raise ValueError(f"unsupported workload kind: {workload_kind}")

        print("RESULT_JSON:" + json.dumps(result, sort_keys=True), flush=True)
        return 0
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        raise
    finally:
        emit_intercept_log()


if __name__ == "__main__":
    raise SystemExit(main())
