from __future__ import annotations

import pytest
import torch

from torchscalers import ShiftScaleScaler


class TestShiftScaleScalerConstruction:
    def test_scalar_float_args(self) -> None:
        scaler = ShiftScaleScaler(shift=2.0, scale=0.5)
        assert torch.allclose(scaler.shift_, torch.tensor(2.0))
        assert torch.allclose(scaler.scale_, torch.tensor(0.5))

    def test_tensor_args(self) -> None:
        shift = torch.tensor([1.0, 2.0])
        scale = torch.tensor([3.0, 4.0])
        scaler = ShiftScaleScaler(shift=shift, scale=scale)
        assert torch.allclose(scaler.shift_, shift)
        assert torch.allclose(scaler.scale_, scale)

    def test_fitted_true_at_init(self) -> None:
        scaler = ShiftScaleScaler(shift=0.0, scale=1.0)
        assert scaler.fitted.item()

    def test_zero_scale_raises(self) -> None:
        with pytest.raises(ValueError, match="strictly positive"):
            ShiftScaleScaler(shift=0.0, scale=0.0)

    def test_negative_scale_raises(self) -> None:
        with pytest.raises(ValueError, match="strictly positive"):
            ShiftScaleScaler(shift=0.0, scale=-1.0)

    def test_negative_scale_in_tensor_raises(self) -> None:
        with pytest.raises(ValueError, match="strictly positive"):
            ShiftScaleScaler(shift=torch.zeros(2), scale=torch.tensor([1.0, -0.5]))


class TestShiftScaleScalerFit:
    def test_fit_is_noop_returns_self(self) -> None:
        scaler = ShiftScaleScaler(shift=1.0, scale=2.0)
        data = torch.tensor([1.0, 2.0, 3.0])
        result = scaler.fit(data)
        assert result is scaler

    def test_fit_does_not_change_parameters(self) -> None:
        scaler = ShiftScaleScaler(shift=1.0, scale=2.0)
        data = torch.tensor([100.0, 200.0, 300.0])
        scaler.fit(data)
        assert torch.allclose(scaler.shift_, torch.tensor(1.0))
        assert torch.allclose(scaler.scale_, torch.tensor(2.0))

    def test_transform_works_without_explicit_fit(self) -> None:
        scaler = ShiftScaleScaler(shift=1.0, scale=2.0)
        data = torch.tensor([1.0, 2.0, 3.0])
        result = scaler.transform(data)
        expected = (data + 1.0) * 2.0
        assert torch.allclose(result, expected)


class TestShiftScaleScalerTransform:
    def test_scalar_shift_and_scale(self) -> None:
        scaler = ShiftScaleScaler(shift=1.0, scale=2.0)
        data = torch.tensor([0.0, 1.0, 2.0])
        result = scaler.transform(data)
        expected = torch.tensor([2.0, 4.0, 6.0])
        assert torch.allclose(result, expected)

    def test_feature_wise_shift_and_scale(self) -> None:
        scaler = ShiftScaleScaler(
            shift=torch.tensor([0.0, -10.0]),
            scale=torch.tensor([1.0, 0.1]),
        )
        data = torch.tensor([[1.0, 20.0], [2.0, 30.0]])
        result = scaler.transform(data)
        expected = torch.tensor([[1.0, 1.0], [2.0, 2.0]])
        assert torch.allclose(result, expected)

    def test_shape_preserved(self) -> None:
        data = torch.randn(5, 4)
        scaler = ShiftScaleScaler(shift=0.0, scale=1.0)
        assert scaler.transform(data).shape == data.shape

    def test_identity_shift_zero_scale_one(self) -> None:
        data = torch.randn(10)
        scaler = ShiftScaleScaler(shift=0.0, scale=1.0)
        assert torch.allclose(scaler.transform(data), data)


class TestShiftScaleScalerInverseTransform:
    def test_inverse_round_trip_scalar(self) -> None:
        data = torch.tensor([1.0, 2.0, 3.0])
        scaler = ShiftScaleScaler(shift=5.0, scale=2.0)
        recovered = scaler.inverse_transform(scaler.transform(data))
        assert torch.allclose(recovered, data, atol=1e-6)

    def test_inverse_round_trip_feature_wise(self) -> None:
        data = torch.tensor([[1.0, 100.0], [2.0, 200.0], [3.0, 300.0]])
        scaler = ShiftScaleScaler(
            shift=torch.tensor([-1.0, -100.0]),
            scale=torch.tensor([2.0, 0.01]),
        )
        recovered = scaler.inverse_transform(scaler.transform(data))
        assert torch.allclose(recovered, data, atol=1e-5)


class TestShiftScaleScalerModule:
    def test_state_dict_round_trip(self) -> None:
        scaler = ShiftScaleScaler(shift=3.0, scale=0.5)
        state = scaler.state_dict()
        restored = ShiftScaleScaler(shift=0.0, scale=1.0)
        restored.load_state_dict(state)
        assert torch.allclose(restored.shift_, scaler.shift_)
        assert torch.allclose(restored.scale_, scaler.scale_)

    def test_buffers_on_cpu(self) -> None:
        scaler = ShiftScaleScaler(shift=1.0, scale=2.0)
        scaler.to("cpu")
        assert scaler.shift_.device.type == "cpu"
        assert scaler.scale_.device.type == "cpu"
