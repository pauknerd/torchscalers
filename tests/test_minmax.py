from __future__ import annotations

import pytest
import torch

from torchscalers import MinMaxScaler


def make_data() -> torch.Tensor:
    """4x3 float tensor with known per-column variation."""
    return torch.tensor(
        [
            [1.0, 10.0, 100.0],
            [2.0, 20.0, 200.0],
            [3.0, 30.0, 300.0],
            [4.0, 40.0, 400.0],
        ]
    )


class TestMinMaxScalerFit:
    def test_fit_stores_min_max(self) -> None:
        data = make_data()
        norm = MinMaxScaler()
        norm.fit(data)
        assert torch.allclose(norm.min_, data.min(dim=0).values)
        assert torch.allclose(norm.max_, data.max(dim=0).values)

    def test_fit_1d(self) -> None:
        data = torch.tensor([1.0, 2.0, 3.0, 4.0])
        norm = MinMaxScaler()
        norm.fit(data)
        assert norm.min_.shape == ()
        assert norm.max_.shape == ()

    def test_fit_returns_self(self) -> None:
        norm = MinMaxScaler()
        result = norm.fit(make_data())
        assert result is norm

    def test_fitted_flag_set_after_fit(self) -> None:
        norm = MinMaxScaler()
        assert not norm.fitted.item()
        norm.fit(make_data())
        assert norm.fitted.item()


class TestMinMaxScalerTransform:
    def test_output_range(self) -> None:
        data = make_data()
        result = MinMaxScaler().fit_transform(data)
        assert result.min().item() >= 0.0
        assert result.max().item() <= 1.0

    def test_first_row_all_zeros(self) -> None:
        data = make_data()
        result = MinMaxScaler().fit_transform(data)
        assert torch.allclose(result[0], torch.zeros(3))

    def test_last_row_all_ones(self) -> None:
        data = make_data()
        result = MinMaxScaler().fit_transform(data)
        assert torch.allclose(result[-1], torch.ones(3))

    def test_shape_preserved(self) -> None:
        data = make_data()
        assert MinMaxScaler().fit_transform(data).shape == data.shape

    def test_transform_before_fit_raises(self) -> None:
        with pytest.raises(RuntimeError, match="fit\\(\\)"):
            MinMaxScaler().transform(make_data())

    def test_constant_feature_no_nan(self) -> None:
        data = torch.tensor([[5.0, 1.0], [5.0, 2.0], [5.0, 3.0]])
        result = MinMaxScaler().fit_transform(data)
        assert torch.all(torch.isfinite(result))
        assert torch.allclose(result[:, 0], torch.zeros(3))


class TestMinMaxScalerInverseTransform:
    def test_inverse_round_trip(self) -> None:
        data = make_data()
        norm = MinMaxScaler()
        recovered = norm.inverse_transform(norm.fit_transform(data))
        assert torch.allclose(recovered, data, atol=1e-5)

    def test_inverse_before_fit_raises(self) -> None:
        with pytest.raises(RuntimeError, match="fit\\(\\)"):
            MinMaxScaler().inverse_transform(make_data())


class TestMinMaxScalerModule:
    def test_buffers_on_cpu_after_to(self) -> None:
        norm = MinMaxScaler()
        norm.fit(make_data())
        norm.to("cpu")
        assert norm.min_.device.type == "cpu"
        assert norm.max_.device.type == "cpu"
        assert norm.fitted.device.type == "cpu"

    def test_state_dict_round_trip(self) -> None:
        data = make_data()
        original = MinMaxScaler()
        original.fit(data)

        state = original.state_dict()

        restored = MinMaxScaler()
        restored.load_state_dict(state)

        assert torch.allclose(restored.min_, original.min_)
        assert torch.allclose(restored.max_, original.max_)
        assert restored.fitted.item() == original.fitted.item()
        assert torch.allclose(restored.transform(data), original.transform(data))
