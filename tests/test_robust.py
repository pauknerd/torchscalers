from __future__ import annotations

import pytest
import torch

from torchscalers import RobustScaler


def make_data() -> torch.Tensor:
    """4x3 float tensor with distinct per-column distributions."""
    return torch.tensor(
        [
            [1.0, 10.0, 100.0],
            [2.0, 20.0, 200.0],
            [3.0, 30.0, 300.0],
            [4.0, 40.0, 400.0],
        ]
    )


class TestRobustScalerFit:
    def test_fit_stores_median_and_iqr(self) -> None:
        data = make_data()
        scaler = RobustScaler()
        scaler.fit(data)
        expected_median = data.median(dim=0).values
        assert torch.allclose(scaler.median_, expected_median)
        assert scaler.iqr_.shape == expected_median.shape

    def test_fit_1d(self) -> None:
        data = torch.tensor([1.0, 2.0, 3.0, 4.0])
        scaler = RobustScaler()
        scaler.fit(data)
        assert scaler.median_.shape == ()
        assert scaler.iqr_.shape == ()

    def test_fit_returns_self(self) -> None:
        scaler = RobustScaler()
        assert scaler.fit(make_data()) is scaler

    def test_fitted_flag_set_after_fit(self) -> None:
        scaler = RobustScaler()
        assert not scaler.fitted.item()
        scaler.fit(make_data())
        assert scaler.fitted.item()

    def test_iqr_correct_values(self) -> None:
        data = make_data()
        scaler = RobustScaler()
        scaler.fit(data)
        expected_q75 = data.quantile(0.75, dim=0)
        expected_q25 = data.quantile(0.25, dim=0)
        assert torch.allclose(scaler.iqr_, expected_q75 - expected_q25)


class TestRobustScalerTransform:
    def test_median_maps_to_zero(self) -> None:
        data = torch.tensor([[1.0], [2.0], [3.0], [5.0], [9.0]])
        scaler = RobustScaler()
        scaler.fit(data)
        median_tensor = scaler.median_.unsqueeze(0)
        out = scaler.transform(median_tensor)
        assert torch.allclose(out, torch.zeros_like(out), atol=1e-6)

    def test_shape_preserved(self) -> None:
        data = make_data()
        assert RobustScaler().fit_transform(data).shape == data.shape

    def test_shape_preserved_1d(self) -> None:
        data = torch.arange(10, dtype=torch.float32)
        assert RobustScaler().fit_transform(data).shape == data.shape

    def test_transform_before_fit_raises(self) -> None:
        with pytest.raises(RuntimeError, match="fit\\(\\)"):
            RobustScaler().transform(make_data())

    def test_constant_feature_no_nan(self) -> None:
        data = torch.tensor([[5.0, 1.0], [5.0, 2.0], [5.0, 3.0]])
        result = RobustScaler().fit_transform(data)
        assert torch.all(torch.isfinite(result))
        assert torch.allclose(result[:, 0], torch.zeros(3))

    def test_outlier_resistance(self) -> None:
        # With an extreme outlier the IQR-based scaler should still map
        # the middle quantile region to a modest range.
        data = torch.tensor([[1.0], [2.0], [3.0], [4.0], [1000.0]])
        result = RobustScaler().fit_transform(data)
        # The non-outlier values should be within a reasonable range
        assert result[:4].abs().max().item() < 5.0


class TestRobustScalerInverseTransform:
    def test_inverse_round_trip(self) -> None:
        data = make_data()
        scaler = RobustScaler()
        recovered = scaler.inverse_transform(scaler.fit_transform(data))
        assert torch.allclose(recovered, data, atol=1e-5)

    def test_inverse_round_trip_1d(self) -> None:
        data = torch.arange(10, dtype=torch.float32)
        scaler = RobustScaler()
        recovered = scaler.inverse_transform(scaler.fit_transform(data))
        assert torch.allclose(recovered, data, atol=1e-5)

    def test_inverse_before_fit_raises(self) -> None:
        with pytest.raises(RuntimeError, match="fit\\(\\)"):
            RobustScaler().inverse_transform(make_data())


class TestRobustScalerModule:
    def test_buffers_on_cpu(self) -> None:
        scaler = RobustScaler()
        scaler.fit(make_data())
        scaler.to("cpu")
        assert scaler.median_.device.type == "cpu"
        assert scaler.iqr_.device.type == "cpu"

    def test_state_dict_round_trip(self) -> None:
        data = make_data()
        original = RobustScaler()
        original.fit(data)

        state = original.state_dict()

        restored = RobustScaler()
        restored.load_state_dict(state)

        assert torch.allclose(restored.median_, original.median_)
        assert torch.allclose(restored.iqr_, original.iqr_)
        assert restored.fitted.item() == original.fitted.item()
        assert torch.allclose(restored.transform(data), original.transform(data))
