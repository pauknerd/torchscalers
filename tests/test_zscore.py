from __future__ import annotations

import pytest
import torch

from torchscalers import ZScoreScaler


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


class TestZScoreScalerFit:
    def test_fit_2d_mean_correct(self) -> None:
        data = make_data()
        norm = ZScoreScaler()
        norm.fit(data)
        expected_mean = data.mean(dim=0)
        assert torch.allclose(norm.mean, expected_mean)

    def test_fit_2d_std_correct(self) -> None:
        data = make_data()
        norm = ZScoreScaler()
        norm.fit(data)
        expected_std = data.std(dim=0)
        assert torch.allclose(norm.std, expected_std)

    def test_fit_1d(self) -> None:
        data = torch.tensor([1.0, 2.0, 3.0, 4.0])
        norm = ZScoreScaler()
        norm.fit(data)
        assert norm.mean.shape == ()  # scalar
        assert norm.std.shape == ()

    def test_fit_returns_self(self) -> None:
        norm = ZScoreScaler()
        result = norm.fit(make_data())
        assert result is norm

    def test_fitted_flag_set_after_fit(self) -> None:
        norm = ZScoreScaler()
        assert not norm.fitted.item()
        norm.fit(make_data())
        assert norm.fitted.item()


class TestZScoreScalerTransform:
    def test_transform_zero_mean(self) -> None:
        data = make_data()
        norm = ZScoreScaler()
        result = norm.fit_transform(data)
        assert torch.allclose(result.mean(dim=0), torch.zeros(3), atol=1e-5)

    def test_transform_unit_variance(self) -> None:
        data = make_data()
        norm = ZScoreScaler()
        result = norm.fit_transform(data)
        assert torch.allclose(result.std(dim=0), torch.ones(3), atol=1e-5)

    def test_transform_shape_preserved(self) -> None:
        data = make_data()
        norm = ZScoreScaler()
        result = norm.fit_transform(data)
        assert result.shape == data.shape

    def test_transform_before_fit_raises(self) -> None:
        with pytest.raises(RuntimeError, match="fit\\(\\)"):
            ZScoreScaler().transform(make_data())

    def test_constant_feature_no_nan(self) -> None:
        data = torch.tensor([[5.0, 1.0], [5.0, 2.0], [5.0, 3.0]])
        norm = ZScoreScaler()
        result = norm.fit_transform(data)
        assert torch.all(torch.isfinite(result))
        assert torch.allclose(result[:, 0], torch.zeros(3))


class TestZScoreScalerInverseTransform:
    def test_inverse_round_trip(self) -> None:
        data = make_data()
        norm = ZScoreScaler()
        recovered = norm.inverse_transform(norm.fit_transform(data))
        assert torch.allclose(recovered, data, atol=1e-5)

    def test_inverse_before_fit_raises(self) -> None:
        with pytest.raises(RuntimeError, match="fit\\(\\)"):
            ZScoreScaler().inverse_transform(make_data())


class TestZScoreScalerModule:
    def test_buffers_on_cpu_after_to(self) -> None:
        norm = ZScoreScaler()
        norm.fit(make_data())
        norm.to("cpu")
        assert norm.mean.device.type == "cpu"
        assert norm.std.device.type == "cpu"
        assert norm.fitted.device.type == "cpu"

    def test_state_dict_round_trip(self) -> None:
        data = make_data()
        original = ZScoreScaler()
        original.fit(data)

        state = original.state_dict()

        restored = ZScoreScaler()
        restored.load_state_dict(state)

        assert torch.allclose(restored.mean, original.mean)
        assert torch.allclose(restored.std, original.std)
        assert restored.fitted.item() == original.fitted.item()
        assert torch.allclose(restored.transform(data), original.transform(data))

    def test_call_equals_transform(self) -> None:
        data = make_data()
        norm = ZScoreScaler()
        norm.fit(data)
        assert torch.allclose(norm(data), norm.transform(data))
