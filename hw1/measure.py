"""Measure latency, allocated memory and whole-GPU energy on the full grid."""

from __future__ import annotations

import argparse
import gc
import json
import math
import platform
import statistics
import threading
import time
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import torch
from torch import nn

from equations import bytes_moved, flops, memory
from models import build_model


BASE_IMAGE_SIZES = [32, 64, 128, 224, 256, 384, 512]
BASE_BATCH_SIZES = [1, 2, 4, 8, 16, 32, 64, 128, 256]


def measurement_grid(seed: int = 2026) -> tuple[list[int], list[int]]:
    """Return the required base grid plus deterministic random additions."""

    rng = np.random.default_rng(seed)
    candidates_s = [s for s in range(32, 513, 16) if s not in BASE_IMAGE_SIZES]
    random_s = sorted(int(x) for x in rng.choice(candidates_s, 4, replace=False))
    candidates_b = [b for b in range(1, 257) if b not in BASE_BATCH_SIZES]
    random_b = sorted(int(x) for x in rng.choice(candidates_b, 3, replace=False))
    return sorted(BASE_IMAGE_SIZES + random_s), sorted(BASE_BATCH_SIZES + random_b)


def is_validation_point(image_size: int, batch: int) -> bool:
    """Points involving a random coordinate are held out from calibration."""

    return image_size not in BASE_IMAGE_SIZES or batch not in BASE_BATCH_SIZES


def reference_flops_and_forward(
    model: nn.Module, x: torch.Tensor
) -> tuple[int, torch.Tensor]:
    """Independently tally Conv/Linear MAC FLOPs from actual output tensors."""

    total = 0

    def hook(module: nn.Module, _inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
        nonlocal total
        if isinstance(module, nn.Conv2d):
            kh, kw = module.kernel_size
            macs_per_output = kh * kw * module.in_channels // module.groups
            total += int(2 * output.numel() * macs_per_output)
        elif isinstance(module, nn.Linear):
            total += int(2 * output.numel() * module.in_features)

    handles = [
        module.register_forward_hook(hook)
        for module in model.modules()
        if isinstance(module, (nn.Conv2d, nn.Linear))
    ]
    try:
        y = model(x)
    finally:
        for handle in handles:
            handle.remove()
    return total, y


def estimate_latency_seconds(model: nn.Module, x: torch.Tensor, iterations: int = 5) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        model(x)
    end.record()
    end.synchronize()
    return max(start.elapsed_time(end) / (1_000.0 * iterations), 1e-7)


def latency_samples(
    model: nn.Module,
    x: torch.Tensor,
    estimate: float,
    target_seconds: float,
    minimum: int = 15,
    maximum: int = 1_000,
) -> np.ndarray:
    repeats = int(np.clip(math.ceil(target_seconds / estimate), minimum, maximum))
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(repeats)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(repeats)]
    for start, end in zip(starts, ends, strict=True):
        start.record()
        model(x)
        end.record()
    ends[-1].synchronize()
    return np.asarray(
        [start.elapsed_time(end) / 1_000.0 for start, end in zip(starts, ends, strict=True)],
        dtype=np.float64,
    )


class NvmlEnergyMeter:
    """Measure board energy, preferring NVML's monotonic energy counter."""

    def __init__(self, index: int = 0) -> None:
        import pynvml

        self.nvml = pynvml
        pynvml.nvmlInit()
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(index)
        try:
            pynvml.nvmlDeviceGetTotalEnergyConsumption(self.handle)
            self.counter_supported = True
        except pynvml.NVMLError:
            self.counter_supported = False

    def power_w(self) -> float:
        return float(self.nvml.nvmlDeviceGetPowerUsage(self.handle)) / 1_000.0

    def idle_power_w(self, samples: int = 20, interval: float = 0.02) -> float:
        values = []
        for _ in range(samples):
            try:
                values.append(self.power_w())
            except self.nvml.NVMLError:
                break
            time.sleep(interval)
        return float(statistics.median(values)) if values else float("nan")

    def measure(
        self, run_once: Callable[[], None], iterations: int
    ) -> tuple[float, float, float, str, int]:
        """Return joules/pass, mean watts, duration, method, sample count."""

        torch.cuda.synchronize()
        if self.counter_supported:
            before = self.nvml.nvmlDeviceGetTotalEnergyConsumption(self.handle)
            t0 = time.perf_counter()
            for _ in range(iterations):
                run_once()
            torch.cuda.synchronize()
            duration = time.perf_counter() - t0
            after = self.nvml.nvmlDeviceGetTotalEnergyConsumption(self.handle)
            total_j = max(float(after - before) / 1_000.0, 0.0)
            if total_j > 0:
                return total_j / iterations, total_j / duration, duration, "nvml_counter", 2

        samples: list[tuple[float, float]] = []
        stop = threading.Event()

        def poll() -> None:
            while not stop.is_set():
                try:
                    samples.append((time.perf_counter(), self.power_w()))
                except self.nvml.NVMLError:
                    pass
                stop.wait(0.005)

        thread = threading.Thread(target=poll, daemon=True)
        thread.start()
        time.sleep(0.03)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iterations):
            run_once()
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        time.sleep(0.03)
        stop.set()
        thread.join(timeout=1.0)
        duration = t1 - t0
        inside = [power for timestamp, power in samples if t0 <= timestamp <= t1]
        if not inside:
            inside = [power for _, power in samples]
        if not inside:
            return float("nan"), float("nan"), duration, "unavailable", 0
        mean_power = float(np.mean(inside))
        return mean_power * duration / iterations, mean_power, duration, "nvml_power", len(inside)

    def close(self) -> None:
        try:
            self.nvml.nvmlShutdown()
        except self.nvml.NVMLError:
            pass


def environment_info(
    image_sizes: list[int], batch_sizes: list[int], idle_power_w: float
) -> dict[str, object]:
    props = torch.cuda.get_device_properties(0)
    return {
        "timestamp_local": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda_build": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "gpu": props.name,
        "compute_capability": f"{props.major}.{props.minor}",
        "gpu_total_memory_bytes": int(props.total_memory),
        "idle_power_w_before_sweep": idle_power_w,
        "dtype": "float32",
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "image_sizes": image_sizes,
        "batch_sizes": batch_sizes,
    }


def save_rows(rows: list[dict[str, object]], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows).sort_values(["image_size", "batch"], ignore_index=True)
    frame.to_csv(destination, index=False, float_format="%.17g")


def run(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this assignment")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    image_sizes, batch_sizes = measurement_grid(args.seed)
    if args.quick:
        image_sizes, batch_sizes = [32, 128, 256], [1, 8, 64]

    existing: list[dict[str, object]] = []
    done: set[tuple[int, int]] = set()
    if args.resume and output.exists():
        old = pd.read_csv(output)
        existing = old.to_dict("records")
        done = {(int(row["image_size"]), int(row["batch"])) for row in existing}

    model = build_model().cuda().eval()
    parameter_count = sum(p.numel() for p in model.parameters())
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Parameters: {parameter_count:,}")
    print(f"Image sizes: {image_sizes}")
    print(f"Batch sizes: {batch_sizes}")

    meter: NvmlEnergyMeter | None
    try:
        meter = NvmlEnergyMeter(0)
        idle_power = meter.idle_power_w()
    except Exception as exc:
        print(f"NVML unavailable ({exc!r}); energy will be NaN")
        meter = None
        idle_power = float("nan")

    metadata = environment_info(image_sizes, batch_sizes, idle_power)
    metadata["parameter_count"] = parameter_count
    metadata_path = output.parent / "environment.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    jobs = [(s, b) for s in image_sizes for b in batch_sizes if (s, b) not in done]
    rng = np.random.default_rng(args.seed + 1)
    rng.shuffle(jobs)
    if args.max_points is not None:
        jobs = jobs[: args.max_points]

    rows = list(existing)
    total = len(jobs)
    for index, (image_size, batch) in enumerate(jobs, start=1):
        predicted_memory = int(round(memory(image_size, batch)))
        row: dict[str, object] = {
            "S": image_size,
            "B": batch,
            "image_size": image_size,
            "batch": batch,
            "is_validation": is_validation_point(image_size, batch),
            "status": "ok",
            "flops_pred": int(round(flops(image_size, batch))),
            "flops_reference": float("nan"),
            "bytes_moved_pred": int(round(bytes_moved(image_size, batch))),
            "memory_pred_bytes": predicted_memory,
            "memory_measured_bytes": float("nan"),
            "predicted_oom": predicted_memory > torch.cuda.get_device_properties(0).total_memory,
            "latency_seconds": float("nan"),
            "latency_p10_seconds": float("nan"),
            "latency_p90_seconds": float("nan"),
            "latency_repeats": 0,
            "energy_joules": float("nan"),
            "mean_power_w": float("nan"),
            "energy_duration_seconds": float("nan"),
            "energy_repeats": 0,
            "energy_method": "unavailable",
            "energy_samples": 0,
        }
        x: torch.Tensor | None = None
        y: torch.Tensor | None = None
        try:
            gc.collect()
            torch.cuda.empty_cache()
            x = torch.randn(batch, 3, image_size, image_size, device="cuda", dtype=torch.float32)
            with torch.inference_mode():
                for _ in range(args.warmup):
                    model(x)
                torch.cuda.synchronize()

                torch.cuda.reset_peak_memory_stats()
                reference, y = reference_flops_and_forward(model, x)
                torch.cuda.synchronize()
                row["flops_reference"] = reference
                row["memory_measured_bytes"] = int(torch.cuda.max_memory_allocated())

                estimate = estimate_latency_seconds(model, x)
                samples = latency_samples(model, x, estimate, args.latency_seconds)
                row["latency_seconds"] = float(np.median(samples))
                row["latency_p10_seconds"] = float(np.percentile(samples, 10))
                row["latency_p90_seconds"] = float(np.percentile(samples, 90))
                row["latency_repeats"] = len(samples)

                if meter is not None and not args.skip_energy:
                    energy_repeats = int(
                        np.clip(math.ceil(args.energy_seconds / estimate), 10, 5_000)
                    )
                    for _ in range(3):
                        model(x)
                    energy, power, duration, method, count = meter.measure(
                        lambda: model(x), energy_repeats
                    )
                    row["energy_joules"] = energy
                    row["mean_power_w"] = power
                    row["energy_duration_seconds"] = duration
                    row["energy_repeats"] = energy_repeats
                    row["energy_method"] = method
                    row["energy_samples"] = count
        except (torch.OutOfMemoryError, RuntimeError) as exc:
            is_oom = isinstance(exc, torch.OutOfMemoryError) or "out of memory" in str(exc).lower()
            if not is_oom:
                raise
            row["status"] = "OOM"
            row["error"] = str(exc).splitlines()[0]
            print(f"[{index:03d}/{total:03d}] S={image_size:3d} B={batch:3d}: OOM")
        else:
            print(
                f"[{index:03d}/{total:03d}] S={image_size:3d} B={batch:3d}: "
                f"{1e3 * float(row['latency_seconds']):8.3f} ms, "
                f"{float(row['memory_measured_bytes']) / 2**20:8.1f} MiB, "
                f"{float(row['energy_joules']):8.4f} J"
            )
        finally:
            del y, x
            gc.collect()
            torch.cuda.empty_cache()

        rows.append(row)
        save_rows(rows, output)

    if meter is not None:
        meter.close()
    print(f"Wrote {len(rows)} rows to {output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="results/measurements.csv")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--latency-seconds", type=float, default=0.20)
    parser.add_argument("--energy-seconds", type=float, default=1.00)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip-energy", action="store_true")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--max-points", type=int)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
