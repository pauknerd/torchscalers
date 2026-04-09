from __future__ import annotations

import pytest
import torch

from torchscalers import MaxAbsScaler


def make_data() -> torch.Tensor:
    """4x3 float tensor with positive and negative values."""
    return torch.tensor(
        [
            [-4.0, -40.0, -400.0],
            [-2.0, -10.0, -100.0],
            [2.0, 10.0, 100.0],
            [4.0, 40.0, 400.0],
        ]
    )


class TestMaxAbsScalerFit:
    def test_fit_stores_max_abs(self) -> None:
        data = make_data()
        scaler = MaxAbsScaler()
        scaler.fit(data)
        expected = data.abs().max(dim=0).values
        assert torch.allclose(scaler.max_abs_, expected)

    def test_fit_1d(self) -> None:
        data = torch.tensor([-3.0, 1.0, 2.0])
        scaler = MaxAbsScaler()
        scaler.fit(data)
        assert scaler.max_abs_.shape == ()
        assert torch.allclose(scaler.max_abs_, torch.tensor(3.0))

    def test_fit_returns_self(self) -> None:
        scaler = MaxAbsScaler()
        assert scaler.fit(make_data()) is scaler

    def test_fitted_flag_set_after_fit(self) -> None:
        scaler = MaxAbsScaler()
        assert not scaler.fitted.item()
        scaler.fit(make_data())
        assert scaler.fitted.item()


class TestMaxAbsScalerTransform:
    def test_output_in_minus_one_to_one(self) -> None:
        data = make_data()
        result = MaxAbsScaler().fit_transform(data)
        assert result.min().item() >= -1.0
        assert result.max().item() <= 1.0

    def test_max_abs_maps_to_one(self) -> None:
        data = make_data()
        scaler = MaxAbsScaler()
        result = scaler.fit_transform(data)
        assert torch.allclose(result.abs().max(dim=0).values, torch.ones(3))

    def test_shape_preserved(self) -> None:
        data = make_data()
        assert MaxAbsScaler().fit_transform(data).shape == data.shape

    def test_shape_preserved_1d(self) -> None:
        data = torch.tensor([-3.0, 1.0, 2.0])
        assert MaxAbsScaler().fit_transform(data).shape == data.shape

    def test_transform_before_fit_raises(self) -> None:
        with pytest.raises(RuntimeError, match="fit\\(\\)"):
            MaxAbsScaler().transform(make_data())

    def test_all_zero_feature_no_nan(self) -> None:
        data = torch.tensor([[0.0, 1.0], [0.0, 2.0], [0.0, 3.0]])
        result = MaxAbsScaler().fit_transform(data)
        assert torch.all(torch.isfinite(result))
        assert torch.allclose(result[:, 0], torch.zeros(3))

    def test_sign_preserved(self) -> None:
        data = make_data()
        result = MaxAbsScaler().fit_transform(data)
        assert (result[:2] <= 0).all()
        assert (result[2:] >= 0).all()


class TestMaxAbsScalerInverseTransform:
    def test_inverse_round_trip(self) -> None:
        data = make_data()
        scaler = MaxAbsScaler()
        recovered = scaler.inverse_transform(scaler.fit_transform(data))
        assert torch.allclose(recovered, data, atol=1e-5)

    def test_inverse_round_trip_1d(self) -> None:
        data = torch.tensor([-3.0, 1.0, 2.0])
        scaler = MaxAbsScaler()
        recovered = scaler.inverse_transform(scaler.fit_transform(data))
        assert torch.allclose(recovered, data, atol=1e-5)

    def test_inverse_before_fit_raises(self) -> None:
        with pytest.raises(RuntimeError, match="fit\\(\\)"):
            MaxAbsScaler().inverse_transform(make_data())


class TestMaxAbsScalerModule:
    def test_buffers_on_cpu(self) -> None:
        scaler = MaxAbsScaler()
        scaler.fit(make_data())
        scaler.to("cpu")
        assert scaler.max_abs_.device.type == "cpu"
        assert scaler.fitted.device.type == "cpu"

    def test_state_dict_round_trip(self) -> None:
        data = make_data()
        original = MaxAbsScaler()
        original.fit(data)

        state = original.state_dict()

        restored = MaxAbsScaler()
        restored.load_state_dict(state)

        assert torch.allclose(restored.max_abs_, original.max_abs_)
        assert restored.fitted.item() == original.fitted.item()
        assert torch.allclose(restored.transform(data), original.transform(data))
