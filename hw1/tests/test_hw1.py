from __future__ import annotations

import numpy as np
import pytest
import torch

from equations import (
    PARAMETER_BYTES,
    PARAMETER_COUNT,
    bytes_moved,
    energy,
    flops,
    latency,
    memory,
)
from measure import measurement_grid, reference_flops_and_forward
from models import build_model


def test_model_shape_and_parameter_count() -> None:
    model = build_model()
    assert sum(p.numel() for p in model.parameters()) == PARAMETER_COUNT
    with torch.inference_mode():
        assert model(torch.randn(2, 3, 32, 32)).shape == (2, 100)


@pytest.mark.parametrize("image_size,batch", [(32, 1), (64, 2), (112, 3)])
def test_closed_form_flops_matches_layer_tally(image_size: int, batch: int) -> None:
    model = build_model()
    with torch.inference_mode():
        reference, output = reference_flops_and_forward(
            model, torch.randn(batch, 3, image_size, image_size)
        )
    assert output.shape == (batch, 100)
    assert reference == flops(image_size, batch)


def test_known_equations() -> None:
    assert flops(32, 1) == 17_712 * 32**2 + 313_344
    assert memory(32, 1) == PARAMETER_BYTES + 44 * 32**2
    assert bytes_moved(32, 1) == 4 * (91 * 32**2 + 2_148 + PARAMETER_COUNT)


def test_numpy_broadcasting() -> None:
    sizes = np.array([[32], [64]])
    batches = np.array([1, 2, 4])
    for function in (flops, memory, bytes_moved):
        result = function(sizes, batches)
        assert isinstance(result, np.ndarray)
        assert result.shape == (2, 3)
        assert np.all(result > 0)


def test_latency_and_energy_models() -> None:
    theta_latency = {
        "launch_seconds_per_kernel": 2e-6,
        "compute_flops_per_second": 5e12,
        "memory_bytes_per_second": 2e11,
    }
    theta_energy = {
        "idle_power_w": 20.0,
        "joules_per_flop": 1e-12,
        "joules_per_byte": 1e-10,
        "latency_theta": theta_latency,
    }
    values_t = latency(np.array([32, 224]), np.array([1, 8]), theta_latency)
    values_e = energy(np.array([32, 224]), np.array([1, 8]), theta_energy)
    assert values_t.shape == (2,)
    assert np.all(values_t > 0)
    assert np.all(values_e > 0)


def test_grid_is_reproducible_and_complete() -> None:
    sizes, batches = measurement_grid(2026)
    assert sizes == [32, 48, 64, 112, 128, 224, 256, 352, 384, 400, 512]
    assert batches == [1, 2, 4, 8, 16, 32, 64, 95, 98, 128, 167, 256]
    assert len(sizes) * len(batches) == 132
