from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
import time
import traceback
import urllib.request

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms


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


def get_data_root() -> str:
    data_root = get_env_str("WORKLOAD_DATA_ROOT", "/tmp/hami-benchmark-data").strip() or "/tmp/hami-benchmark-data"
    os.makedirs(data_root, exist_ok=True)
    return data_root


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


def compute_series_slope(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    n = float(len(values))
    mean_x = (n - 1.0) / 2.0
    mean_y = sum(values) / n
    numerator = 0.0
    denominator = 0.0
    for index, value in enumerate(values):
        x = float(index)
        numerator += (x - mean_x) * (value - mean_y)
        denominator += (x - mean_x) ** 2
    if denominator <= 0.0:
        return 0.0
    return numerator / denominator


def download_file(url: str, destination: str) -> None:
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    if os.path.exists(destination):
        return
    urllib.request.urlretrieve(url, destination)


def stable_token_id(token: str, vocab_size: int = 50000) -> int:
    digest = hashlib.blake2b(token.encode("utf-8"), digest_size=4).digest()
    return int.from_bytes(digest, "little") % (vocab_size - 1) + 1


def tokenize_text(text: str, max_tokens: int = 96) -> list[int]:
    tokens = re.findall(r"[a-z0-9']+", text.lower())
    token_ids = [stable_token_id(token) for token in tokens[:max_tokens]]
    return token_ids or [0]


def load_ag_news_sequences(data_root: str) -> list[list[int]]:
    dataset_dir = os.path.join(data_root, "ag_news")
    dataset_path = os.path.join(dataset_dir, "test.csv")
    download_file(
        "https://raw.githubusercontent.com/mhjabreel/CharCNN_Keras/master/data/ag_news_csv/test.csv",
        dataset_path,
    )

    sequences: list[list[int]] = []
    max_samples = get_env_int("WORKLOAD_TEXT_MAX_SAMPLES", 20000)
    with open(dataset_path, "r", encoding="utf-8", newline="") as fh:
        reader = csv.reader(fh)
        for index, row in enumerate(reader):
            if index >= max_samples:
                break
            if len(row) < 3:
                continue
            title = row[1].strip()
            description = row[2].strip()
            sequences.append(tokenize_text(f"{title}. {description}"))
    if not sequences:
        raise RuntimeError(f"no AG News text samples loaded from {dataset_path}")
    return sequences


def sample_text_batch(sequences: list[list[int]], batch_size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    max_tokens = get_env_int("WORKLOAD_TEXT_SEQ_LEN", 128)
    selected = [sequences[index] for index in torch.randint(0, len(sequences), (batch_size,)).tolist()]
    tokens = torch.zeros((batch_size, max_tokens), device=device, dtype=torch.long)
    attention_mask = torch.zeros((batch_size, max_tokens), device=device, dtype=torch.bool)
    for row_index, sequence in enumerate(selected):
        sequence_slice = sequence[:max_tokens]
        sequence_length = len(sequence_slice)
        if sequence_length <= 0:
            continue
        tokens[row_index, :sequence_length] = torch.tensor(sequence_slice, device=device, dtype=torch.long)
        attention_mask[row_index, :sequence_length] = True
    return tokens, attention_mask


def build_cifar10_dataset(data_root: str, train: bool, image_size: int, augment: bool) -> torch.utils.data.Dataset:
    dataset_root = os.path.join(data_root, "cifar10")
    transform_steps: list[transforms.Compose | transforms.Normalize | transforms.ToTensor | transforms.Resize] = []
    if augment:
        transform_steps.extend(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
            ]
        )
    if image_size != 32:
        transform_steps.append(transforms.Resize((image_size, image_size)))
    transform_steps.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )
    return torchvision.datasets.CIFAR10(
        root=dataset_root,
        train=train,
        download=True,
        transform=transforms.Compose(transform_steps),
    )


def infinite_loader(loader: torch.utils.data.DataLoader) -> torch.Tensor:
    while True:
        for batch in loader:
            yield batch


def build_dataloader(dataset: torch.utils.data.Dataset, batch_size: int, shuffle: bool) -> torch.utils.data.DataLoader:
    num_workers = max(0, get_env_int("WORKLOAD_DATALOADER_WORKERS", 4))
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
        drop_last=True,
    )


def build_resnet18(num_classes: int = 1000, use_pretrained: bool = True) -> nn.Module:
    weights = None
    if use_pretrained:
        try:
            weights = torchvision.models.ResNet18_Weights.DEFAULT
        except Exception:
            weights = None
    model = torchvision.models.resnet18(weights=weights)
    if num_classes != 1000:
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


class TextEmbeddingModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        vocab_size = get_env_int("WORKLOAD_TEXT_VOCAB_SIZE", 50000)
        model_dim = get_env_int("WORKLOAD_TEXT_MODEL_DIM", 512)
        num_heads = get_env_int("WORKLOAD_TEXT_HEADS", 8)
        num_layers = get_env_int("WORKLOAD_TEXT_LAYERS", 4)
        max_tokens = get_env_int("WORKLOAD_TEXT_SEQ_LEN", 128)
        ff_dim = get_env_int("WORKLOAD_TEXT_FF_DIM", model_dim * 4)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=0.1,
            batch_first=True,
            activation="gelu",
        )
        self.token_embedding = nn.Embedding(vocab_size, model_dim, padding_idx=0)
        self.position_embedding = nn.Embedding(max_tokens, model_dim)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(model_dim)
        self.projection = nn.Sequential(
            nn.Linear(model_dim, model_dim),
            nn.GELU(),
            nn.Linear(model_dim, model_dim),
        )

    def forward(self, tokens: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(tokens.size(1), device=tokens.device).unsqueeze(0)
        hidden = self.token_embedding(tokens) + self.position_embedding(positions)
        hidden = self.encoder(hidden, src_key_padding_mask=~attention_mask)
        mask = attention_mask.unsqueeze(-1)
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
        return self.projection(self.norm(pooled))


class VisionInferenceModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = build_resnet18(num_classes=1000, use_pretrained=get_env_int("WORKLOAD_USE_PRETRAINED_VISION", 0) == 1)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.model(images)


class TrainModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        use_pretrained = get_env_int("WORKLOAD_USE_PRETRAINED_VISION", 0) == 1
        self.model = build_resnet18(num_classes=10, use_pretrained=use_pretrained)
        if use_pretrained:
            for name, parameter in self.model.named_parameters():
                parameter.requires_grad = name.startswith("fc.")

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.model(images)


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
    data_root = get_data_root()
    model = TextEmbeddingModel().to(device).eval()
    sequences = load_ag_news_sequences(data_root)
    _buffer = maybe_reserve_memory(device, reserve_mib)
    started = time.monotonic()
    batches = 0
    processed = 0
    with torch.inference_mode():
        while time.monotonic() - started < runtime_seconds:
            tokens, attention_mask = sample_text_batch(sequences, batch_size, device)
            _ = model(tokens, attention_mask)
            if device.type == "cuda":
                torch.cuda.synchronize()
            batches += 1
            processed += batch_size
    return build_result("text-embedding-inference", task_name, priority_tier, runtime_seconds, batches, processed, started)


def vision_inference(device: torch.device, runtime_seconds: int, reserve_mib: int, batch_size: int, task_name: str, priority_tier: str) -> dict:
    data_root = get_data_root()
    model = VisionInferenceModel().to(device).eval()
    dataset = build_cifar10_dataset(data_root=data_root, train=False, image_size=get_env_int("WORKLOAD_VISION_IMAGE_SIZE", 320), augment=False)
    loader = build_dataloader(dataset, batch_size=batch_size, shuffle=True)
    batches_iter = infinite_loader(loader)
    _buffer = maybe_reserve_memory(device, reserve_mib)
    started = time.monotonic()
    batches = 0
    processed = 0
    with torch.inference_mode():
        while time.monotonic() - started < runtime_seconds:
            images, _labels = next(batches_iter)
            images = images.to(device, non_blocking=True)
            _ = model(images)
            if device.type == "cuda":
                torch.cuda.synchronize()
            batches += 1
            processed += batch_size
    return build_result("vision-inference", task_name, priority_tier, runtime_seconds, batches, processed, started)


def background_cnn_training(device: torch.device, runtime_seconds: int, reserve_mib: int, batch_size: int, task_name: str, priority_tier: str) -> dict:
    data_root = get_data_root()
    model = TrainModel().to(device).train()
    optimizer = torch.optim.AdamW((parameter for parameter in model.parameters() if parameter.requires_grad), lr=1e-3)
    dataset = build_cifar10_dataset(
        data_root=data_root,
        train=True,
        image_size=get_env_int("WORKLOAD_TRAIN_IMAGE_SIZE", 224),
        augment=True,
    )
    loader = build_dataloader(dataset, batch_size=batch_size, shuffle=True)
    batches_iter = infinite_loader(loader)
    _buffer = maybe_reserve_memory(device, reserve_mib)
    samples_per_epoch = get_env_int("WORKLOAD_TRAIN_SAMPLES_PER_EPOCH", 4096)
    batches_per_epoch = max(1, math.ceil(samples_per_epoch / max(batch_size, 1)))
    train_stop_mode = get_env_str("WORKLOAD_TRAIN_STOP_MODE", "fixed_steps").strip().lower()
    if train_stop_mode not in {"fixed_steps", "slope"}:
        train_stop_mode = "fixed_steps"
    train_min_epochs = max(1, get_env_int("WORKLOAD_TRAIN_MIN_EPOCHS", 8))
    slope_window = max(3, get_env_int("WORKLOAD_TRAIN_SLOPE_WINDOW", 5))
    slope_threshold = get_env_float("WORKLOAD_TRAIN_SLOPE_THRESHOLD", 0.0015)
    slope_patience = max(1, get_env_int("WORKLOAD_TRAIN_SLOPE_PATIENCE", 2))
    slope_min_accuracy = get_env_float("WORKLOAD_TRAIN_SLOPE_MIN_ACCURACY", 0.8)
    fixed_epochs = max(1, get_env_int("WORKLOAD_TRAIN_FIXED_EPOCHS", 90))
    fixed_steps_default = fixed_epochs * batches_per_epoch
    fixed_steps_raw = get_env_int("WORKLOAD_TRAIN_FIXED_STEPS", 0)
    fixed_steps_target = fixed_steps_default if fixed_steps_raw <= 0 else max(1, fixed_steps_raw)
    training_threshold = get_env_float("TRAINING_ACCURACY_THRESHOLD", 0.8)
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
    epoch_accuracy_history: list[float] = []
    slope_converged_windows = 0
    last_accuracy_slope: float | None = None
    best_accuracy = 0.0
    threshold_hit_seconds: float | None = None
    training_converged = False
    training_stop_reason = "time_budget_exhausted"
    fixed_steps_reached = False
    while time.monotonic() - started < runtime_seconds:
        images, labels = next(batches_iter)
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
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
        time_budget_exhausted = time.monotonic() - started >= runtime_seconds
        fixed_steps_hit_now = train_stop_mode == "fixed_steps" and batches >= fixed_steps_target
        should_close_epoch = epoch_batches >= batches_per_epoch or time_budget_exhausted or fixed_steps_hit_now
        if should_close_epoch:
            epoch_index += 1
            elapsed_seconds = time.monotonic() - started
            epoch_average_loss = epoch_loss_total / max(epoch_batches, 1)
            epoch_average_accuracy = epoch_accuracy_total / max(epoch_batches, 1)
            emit_training_epoch_metric(
                task_name=task_name,
                priority_tier=priority_tier,
                epoch_index=epoch_index,
                elapsed_seconds=elapsed_seconds,
                epoch_batches=epoch_batches,
                epoch_processed_samples=epoch_processed,
                total_processed_samples=processed,
                epoch_average_loss=epoch_average_loss,
                epoch_average_accuracy=epoch_average_accuracy,
                smoothed_loss=loss_ema,
                smoothed_accuracy=acc_ema,
            )
            epoch_accuracy_history.append(epoch_average_accuracy)
            if epoch_average_accuracy > best_accuracy:
                best_accuracy = epoch_average_accuracy
            if threshold_hit_seconds is None and epoch_average_accuracy >= training_threshold:
                threshold_hit_seconds = elapsed_seconds

            if train_stop_mode == "slope":
                if epoch_index >= train_min_epochs and len(epoch_accuracy_history) >= slope_window:
                    window_values = epoch_accuracy_history[-slope_window:]
                    last_accuracy_slope = compute_series_slope(window_values)
                    if abs(last_accuracy_slope) <= slope_threshold and epoch_average_accuracy >= slope_min_accuracy:
                        slope_converged_windows += 1
                    else:
                        slope_converged_windows = 0
                    if slope_converged_windows >= slope_patience:
                        training_converged = True
                        training_stop_reason = "converged_slope"
                        break
            elif fixed_steps_hit_now:
                fixed_steps_reached = True
                training_stop_reason = "fixed_steps_reached"
                break

            epoch_batches = 0
            epoch_processed = 0
            epoch_loss_total = 0.0
            epoch_accuracy_total = 0.0
    result = build_result(
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
    result.update(
        {
            "training_final_loss": loss_ema,
            "training_final_accuracy": acc_ema,
            "training_best_accuracy": best_accuracy,
            "training_time_to_accuracy_threshold_seconds": threshold_hit_seconds,
            "training_accuracy_threshold": training_threshold,
            "training_converged": training_converged,
            "training_stop_mode": train_stop_mode,
            "training_stop_reason": training_stop_reason,
            "training_slope_window": slope_window,
            "training_slope_threshold": slope_threshold,
            "training_slope_patience": slope_patience,
            "training_slope_min_accuracy": slope_min_accuracy,
            "training_slope_last": last_accuracy_slope,
            "training_fixed_epochs": fixed_epochs,
            "training_fixed_steps_target": fixed_steps_target,
            "training_fixed_steps_reached": fixed_steps_reached,
        }
    )
    return result


def prepare_assets(device: torch.device, runtime_seconds: int, reserve_mib: int, batch_size: int, task_name: str, priority_tier: str) -> dict:
    del device, runtime_seconds, reserve_mib, batch_size
    data_root = get_data_root()
    started = time.monotonic()
    sequences = load_ag_news_sequences(data_root)
    cifar_train = build_cifar10_dataset(data_root=data_root, train=True, image_size=32, augment=False)
    cifar_test = build_cifar10_dataset(data_root=data_root, train=False, image_size=32, augment=False)
    if get_env_int("WORKLOAD_USE_PRETRAINED_VISION", 1) == 1:
        _vision_model = build_resnet18(num_classes=1000, use_pretrained=True)
    result = build_result(
        "prepare-assets",
        task_name,
        priority_tier,
        0,
        1,
        len(sequences) + len(cifar_train) + len(cifar_test),
        started,
    )
    result["prepared_data_root"] = data_root
    return result


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
        elif workload_kind == "prepare-assets":
            result = prepare_assets(device, runtime_seconds, reserve_mib, batch_size, task_name, priority_tier)
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
