"""Tests for the MPS-safe _assign_column_masked helper in lebesgue.py."""

import pytest
import torch

from prism.partial_responses.lebesgue import _assign_column_masked


def _available_devices():
    """Return list of available torch devices for parametrized tests."""
    devices = [torch.device("cpu")]
    if torch.cuda.is_available():
        devices.append(torch.device("cuda"))
    return devices


DEVICES = _available_devices()
DEVICE_IDS = [str(d) for d in DEVICES]


def _reference_assign(x, row_mask, col_idx, value):
    """Naive direct assignment used as ground-truth on CPU."""
    ref = x.clone()
    ref[row_mask, col_idx] = value
    return ref


@pytest.fixture(params=DEVICES, ids=DEVICE_IDS)
def device(request):
    return request.param


# ---------- core equivalence ----------


class TestEquivalence:
    """_assign_column_masked must match direct tensor[mask, col] = val."""

    def test_mixed_mask(self, device):
        x = torch.randn(8, 4, device=device)
        mask = torch.tensor([True, False, True, False, True, False, True, False], device=device)
        ref = _reference_assign(x, mask, 2, 0.0)
        _assign_column_masked(x, mask, 2, 0.0)
        assert torch.equal(x, ref)

    def test_all_true_mask(self, device):
        x = torch.randn(6, 3, device=device)
        mask = torch.ones(6, dtype=torch.bool, device=device)
        ref = _reference_assign(x, mask, 0, -1.5)
        _assign_column_masked(x, mask, 0, -1.5)
        assert torch.equal(x, ref)

    def test_all_false_mask(self, device):
        x = torch.randn(6, 3, device=device)
        original = x.clone()
        mask = torch.zeros(6, dtype=torch.bool, device=device)
        _assign_column_masked(x, mask, 1, 99.0)
        assert torch.equal(x, original)

    def test_single_true(self, device):
        x = torch.randn(5, 4, device=device)
        mask = torch.tensor([False, False, True, False, False], device=device)
        ref = _reference_assign(x, mask, 3, 7.0)
        _assign_column_masked(x, mask, 3, 7.0)
        assert torch.equal(x, ref)


# ---------- in-place semantics ----------


class TestInPlace:
    """The function must modify the tensor in-place (no copy)."""

    def test_modifies_original(self, device):
        x = torch.zeros(4, 3, device=device)
        mask = torch.tensor([True, True, False, False], device=device)
        _assign_column_masked(x, mask, 1, 5.0)
        assert x[0, 1].item() == 5.0
        assert x[1, 1].item() == 5.0

    def test_non_masked_rows_unchanged(self, device):
        x = torch.ones(4, 3, device=device)
        mask = torch.tensor([True, False, True, False], device=device)
        _assign_column_masked(x, mask, 0, 0.0)
        assert x[1, 0].item() == 1.0
        assert x[3, 0].item() == 1.0

    def test_other_columns_unchanged(self, device):
        x = torch.ones(4, 3, device=device)
        original = x.clone()
        mask = torch.tensor([True, True, True, True], device=device)
        _assign_column_masked(x, mask, 1, 0.0)
        assert torch.equal(x[:, 0], original[:, 0])
        assert torch.equal(x[:, 2], original[:, 2])


# ---------- one-hot group simulation ----------


class TestOneHotGroupPattern:
    """Simulate the loop pattern used in lebesgue.py one-hot handling."""

    def test_sequential_columns(self, device):
        """Apply to multiple columns in sequence, as done for one-hot groups."""
        x = torch.randn(10, 5, device=device)
        mask = x[:, 0] > 0  # arbitrary boolean mask
        group_cols = [1, 2, 3]

        ref = x.clone()
        for col in group_cols:
            ref[mask, col] = 0.0

        for col in group_cols:
            _assign_column_masked(x, mask, col, 0.0)

        assert torch.equal(x, ref)

    def test_two_complementary_masks(self, device):
        """Simulate zero_mask / non_zero_mask split used for one-hot features."""
        x = torch.randn(8, 4, device=device)
        feature_values = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1], device=device)
        zero_mask = feature_values == 0
        non_zero_mask = feature_values != 0

        ref = x.clone()
        ref[zero_mask, 1] = -0.5
        ref[non_zero_mask, 2] = -0.5

        _assign_column_masked(x, zero_mask, 1, -0.5)
        _assign_column_masked(x, non_zero_mask, 2, -0.5)
        assert torch.equal(x, ref)


# ---------- edge cases ----------


class TestEdgeCases:
    def test_single_row(self, device):
        x = torch.randn(1, 3, device=device)
        mask = torch.tensor([True], device=device)
        ref = _reference_assign(x, mask, 0, 42.0)
        _assign_column_masked(x, mask, 0, 42.0)
        assert torch.equal(x, ref)

    def test_single_column(self, device):
        x = torch.randn(5, 1, device=device)
        mask = torch.tensor([True, False, True, False, True], device=device)
        ref = _reference_assign(x, mask, 0, -1.0)
        _assign_column_masked(x, mask, 0, -1.0)
        assert torch.equal(x, ref)

    def test_large_tensor(self, device):
        """Larger tensor -- closer to the sizes that trigger MPS bugs."""
        x = torch.randn(2000, 50, device=device)
        mask = torch.rand(2000, device=device) > 0.5
        ref = _reference_assign(x, mask, 25, 0.0)
        _assign_column_masked(x, mask, 25, 0.0)
        assert torch.equal(x, ref)


# ---------- dtype preservation ----------


class TestDtype:
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_preserves_dtype(self, device, dtype):
        x = torch.randn(6, 3, device=device, dtype=dtype)
        mask = torch.tensor([True, False, True, False, True, False], device=device)
        _assign_column_masked(x, mask, 1, 0.0)
        assert x.dtype == dtype


# ---------- cross-device consistency ----------


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestCrossCPUCUDA:
    """Verify CPU and CUDA produce identical results."""

    def test_results_match_across_devices(self):
        torch.manual_seed(42)
        x_cpu = torch.randn(20, 6)
        x_cuda = x_cpu.clone().cuda()
        mask_cpu = torch.tensor([i % 3 == 0 for i in range(20)])
        mask_cuda = mask_cpu.cuda()

        _assign_column_masked(x_cpu, mask_cpu, 3, -2.5)
        _assign_column_masked(x_cuda, mask_cuda, 3, -2.5)

        assert torch.equal(x_cpu, x_cuda.cpu())
