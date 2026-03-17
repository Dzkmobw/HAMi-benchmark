from __future__ import annotations

import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F


def env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    return int(value)


def select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def maybe_reserve_memory(device: torch.device, reserve_mib: int) -> torch.Tensor | None:
    if device.type != "cuda" or reserve_mib <= 0:
        return None
    elements = reserve_mib * 1024 * 1024 // 2
    try:
        buf = torch.empty(elements, dtype=torch.float16, device=device)
        buf.fill_(0)
        return buf
    except RuntimeError as exc:
        print(f"[warn] failed to reserve {reserve_mib} MiB on GPU: {exc}", flush=True)
        return None


class EmbeddingInferenceModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(32000, 256)
        self.proj = nn.Sequential(
            nn.Linear(256, 512),
            nn.GELU(),
            nn.Linear(512, 384),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        x = self.embedding(tokens).mean(dim=1)
        return self.proj(x)


class VisionInferenceModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.head = nn.Linear(256, 256)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        x = self.net(images).flatten(1)
        return self.head(x)


class TrainModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.fc = nn.Linear(256, 10)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        x = self.net(images).flatten(1)
        return self.fc(x)


def short_inference(device: torch.device, runtime_seconds: int, reserve_mib: int, batch_size: int) -> None:
    model = EmbeddingInferenceModel().to(device).eval()
    _buffer = maybe_reserve_memory(device, reserve_mib)
    deadline = time.time() + runtime_seconds
    steps = 0

    with torch.no_grad():
        while time.time() < deadline:
            tokens = torch.randint(0, 32000, (batch_size, 32), device=device)
            _ = model(tokens)
            if device.type == "cuda":
                torch.cuda.synchronize()
            steps += 1

    print(f"[done] short-inference steps={steps}", flush=True)


def medium_inference(device: torch.device, runtime_seconds: int, reserve_mib: int, batch_size: int) -> None:
    model = VisionInferenceModel().to(device).eval()
    _buffer = maybe_reserve_memory(device, reserve_mib)
    deadline = time.time() + runtime_seconds
    steps = 0

    with torch.no_grad():
        while time.time() < deadline:
            images = torch.randn(batch_size, 3, 320, 320, device=device)
            _ = model(images)
            if device.type == "cuda":
                torch.cuda.synchronize()
            steps += 1

    print(f"[done] medium-inference steps={steps}", flush=True)


def background_training(device: torch.device, runtime_seconds: int, reserve_mib: int, batch_size: int) -> None:
    model = TrainModel().to(device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    _buffer = maybe_reserve_memory(device, reserve_mib)
    deadline = time.time() + runtime_seconds
    steps = 0

    while time.time() < deadline:
        images = torch.randn(batch_size, 3, 64, 64, device=device)
        labels = torch.randint(0, 10, (batch_size,), device=device)
        logits = model(images)
        loss = F.cross_entropy(logits, labels)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if device.type == "cuda":
            torch.cuda.synchronize()
        steps += 1

    print(f"[done] background-training steps={steps}", flush=True)


def main() -> None:
    workload = os.getenv("WORKLOAD_KIND", "short-inference")
    runtime_seconds = env_int("WORKLOAD_RUNTIME_SECONDS", 30)
    reserve_mib = env_int("WORKLOAD_RESERVE_MIB", 0)
    batch_size = env_int("WORKLOAD_BATCH_SIZE", 64)

    device = select_device()
    print(
        f"[start] workload={workload} runtime_seconds={runtime_seconds} "
        f"reserve_mib={reserve_mib} batch_size={batch_size} device={device}",
        flush=True,
    )

    if device.type == "cuda":
        print(f"[gpu] name={torch.cuda.get_device_name(0)}", flush=True)

    if workload == "short-inference":
        short_inference(device, runtime_seconds, reserve_mib, batch_size)
    elif workload == "medium-inference":
        medium_inference(device, runtime_seconds, reserve_mib, batch_size)
    elif workload == "background-training":
        background_training(device, runtime_seconds, reserve_mib, batch_size)
    else:
        raise ValueError(f"unknown workload: {workload}")


if __name__ == "__main__":
    main()
