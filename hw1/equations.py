"""Closed-form performance equations for the Homework 1 CNN.

All public functions accept scalars or NumPy-broadcastable arrays.  The FLOP
count includes convolution and linear multiply-accumulates (one MAC = two
FLOPs).  Pooling, ReLU, indexing and comparisons are excluded from FLOPs but
included in the compulsory-byte estimate used by the roofline models.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np


DTYPE_BYTES = 4
NUM_LOGICAL_KERNELS = 17
PARAMETER_COUNT = 1_040_324
PARAMETER_BYTES = DTYPE_BYTES * PARAMETER_COUNT


def _inputs(image_size: Any, batch: Any) -> tuple[np.ndarray, np.ndarray]:
    s, b = np.broadcast_arrays(
        np.asarray(image_size, dtype=np.float64),
        np.asarray(batch, dtype=np.float64),
    )
    if np.any(s <= 0) or np.any(b <= 0):
        raise ValueError("image_size and batch must be positive")
    return s, b


def _scalar_or_array(value: np.ndarray) -> float | np.ndarray:
    return float(value) if value.ndim == 0 else value


def flops(image_size: Any, batch: Any) -> float | np.ndarray:
    """Return forward-pass convolution/linear FLOPs.

    The six convolution coefficients are 2352, 6400, 2304, 1024, 4608,
    and 1024 FLOPs per ``B*S^2``.  The two linear layers add 313344 FLOPs
    per sample.
    """

    s, b = _inputs(image_size, batch)
    value = b * (17_712.0 * s**2 + 313_344.0)
    return _scalar_or_array(value)


def memory(image_size: Any, batch: Any) -> float | np.ndarray:
    """Return the ideal peak live-tensor memory in bytes.

    At the first convolution, input and output coexist: ``(3 + 8) B S^2``
    FP32 values.  This is the largest adjacent-activation pair.  Parameters
    remain live throughout.  CUDA allocator rounding, library context,
    cuDNN workspaces, and the caching allocator's *reserved* memory are not
    part of this parameter-free equation.
    """

    s, b = _inputs(image_size, batch)
    value = PARAMETER_BYTES + 44.0 * b * s**2
    return _scalar_or_array(value)


def bytes_moved(image_size: Any, batch: Any) -> float | np.ndarray:
    """Return one-pass compulsory tensor traffic in bytes.

    Each logical operator reads its input(s) and parameters once and writes
    its output once.  In-place ReLU still reads and writes its activation.
    The expression is ``4 * (91 B S^2 + 2148 B + 1040324)``.
    Cache-line effects, repeated DRAM transactions and implementation-specific
    workspace traffic are intentionally excluded.
    """

    s, b = _inputs(image_size, batch)
    value = DTYPE_BYTES * (91.0 * b * s**2 + 2_148.0 * b + PARAMETER_COUNT)
    return _scalar_or_array(value)


def latency(image_size: Any, batch: Any, theta: Mapping[str, Any]) -> float | np.ndarray:
    """Predict seconds per forward with a launch-aware roofline model.

    ``t = N*L + max(F/P, Q/W)`` where ``N=17`` logical kernels, ``L`` is
    seconds per logical launch, ``P`` is effective FLOP/s, and ``W`` is
    effective byte/s.  All three values are calibrated on the target GPU.
    """

    f = np.asarray(flops(image_size, batch), dtype=np.float64)
    q = np.asarray(bytes_moved(image_size, batch), dtype=np.float64)
    launch = float(theta["launch_seconds_per_kernel"])
    throughput = float(theta["compute_flops_per_second"])
    bandwidth = float(theta["memory_bytes_per_second"])
    if launch < 0 or throughput <= 0 or bandwidth <= 0:
        raise ValueError("latency parameters must be physically positive")
    value = NUM_LOGICAL_KERNELS * launch + np.maximum(f / throughput, q / bandwidth)
    return _scalar_or_array(value)


def latency_components(
    image_size: Any, batch: Any, theta: Mapping[str, Any]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return launch, compute and memory time components in seconds."""

    f = np.asarray(flops(image_size, batch), dtype=np.float64)
    q = np.asarray(bytes_moved(image_size, batch), dtype=np.float64)
    launch = np.broadcast_to(
        NUM_LOGICAL_KERNELS * float(theta["launch_seconds_per_kernel"]), f.shape
    )
    compute = f / float(theta["compute_flops_per_second"])
    memory_time = q / float(theta["memory_bytes_per_second"])
    return launch, compute, memory_time


def regime(image_size: Any, batch: Any, theta: Mapping[str, Any]) -> np.ndarray:
    """Classify points as ``launch``, ``memory`` or ``compute`` bound."""

    launch, compute, memory_time = latency_components(image_size, batch, theta)
    stacked = np.stack([launch, memory_time, compute], axis=0)
    labels = np.array(["launch", "memory", "compute"], dtype=object)
    return labels[np.argmax(stacked, axis=0)]


def energy(
    image_size: Any, batch: Any, theta_energy: Mapping[str, Any]
) -> float | np.ndarray:
    """Predict whole-GPU joules per forward pass.

    The calibrated model is additive:

    ``E = P_idle * latency_hat + e_F * FLOPs + e_Q * bytes_moved``.

    It does not subtract idle energy because the assignment defines energy on
    the whole GPU.  ``theta_energy`` must contain ``latency_theta`` plus the
    three non-negative energy coefficients.
    """

    t = np.asarray(
        latency(image_size, batch, theta_energy["latency_theta"]),
        dtype=np.float64,
    )
    f = np.asarray(flops(image_size, batch), dtype=np.float64)
    q = np.asarray(bytes_moved(image_size, batch), dtype=np.float64)
    value = (
        float(theta_energy["idle_power_w"]) * t
        + float(theta_energy["joules_per_flop"]) * f
        + float(theta_energy["joules_per_byte"]) * q
    )
    return _scalar_or_array(value)


__all__ = [
    "DTYPE_BYTES",
    "NUM_LOGICAL_KERNELS",
    "PARAMETER_COUNT",
    "PARAMETER_BYTES",
    "bytes_moved",
    "energy",
    "flops",
    "latency",
    "latency_components",
    "memory",
    "regime",
]
