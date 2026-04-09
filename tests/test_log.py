from __future__ import annotations

import math

import torch

from torchscalers import LogScaler, ZScoreScaler


class TestLogScalerInit:
    def test_fitted_true_at_init(self) -> None:
        scaler = LogScaler()
        assert scaler.fitted.item()

    def test_custom_eps(self) -> None:
        scaler = LogScaler(eps=1e-4)
        assert scaler.eps == 1e-4


class TestLogScalerFit:
    def test_fit_is_noop_returns_self(self) -> None:
        scaler = LogScaler()
        data = torch.tensor([1.0, 2.0, 3.0])
        assert scaler.fit(data) is scaler

    def test_transform_works_without_explicit_fit(self) -> None:
        scaler = LogScaler()
        data = torch.tensor([1.0, math.e - 1e-8, math.e**2 - 1e-8])
        result = scaler.transform(data)
        assert torch.all(torch.isfinite(result))


class TestLogScalerTransform:
    def test_known_values(self) -> None:
        eps = 1e-8
        scaler = LogScaler(eps=eps)
        # log(1 + eps) ≈ 0, log(e + eps - eps) = log(e) = 1
        data = torch.tensor([0.0, math.e - eps])
        result = scaler.transform(data)
        assert torch.allclose(result[0], torch.tensor(math.log(eps)), atol=1e-5)
        assert torch.allclose(result[1], torch.tensor(1.0), atol=1e-5)

    def test_shape_preserved(self) -> None:
        data = torch.rand(5, 4) + 1.0
        assert LogScaler().transform(data).shape == data.shape

    def test_output_finite_for_positive_input(self) -> None:
        data = torch.rand(100) + 1.0
        result = LogScaler().transform(data)
        assert torch.all(torch.isfinite(result))

    def test_compresses_large_values(self) -> None:
        # The log of a large spread should be much smaller than the raw spread
        data = torch.tensor([1.0, 1e6])
        raw_range = (data.max() - data.min()).item()
        log_out = LogScaler().transform(data)
        log_range = (log_out.max() - log_out.min()).item()
        assert log_range < raw_range / 10


class TestLogScalerInverseTransform:
    def test_inverse_round_trip(self) -> None:
        data = torch.rand(10) + 1.0  # strictly positive
        scaler = LogScaler()
        recovered = scaler.inverse_transform(scaler.transform(data))
        assert torch.allclose(recovered, data, atol=1e-5)

    def test_inverse_round_trip_2d(self) -> None:
        data = torch.rand(5, 3) + 1.0
        scaler = LogScaler()
        recovered = scaler.inverse_transform(scaler.transform(data))
        assert torch.allclose(recovered, data, atol=1e-5)


class TestLogScalerChaining:
    """
    Use case: heavy-tailed (log-normal) data.
    Apply LogScaler then ZScoreScaler to obtain approximately N(0, 1) output,
    and verify the full inverse chain recovers the original values.
    """

    def test_log_then_zscore_approaches_standard_normal(self) -> None:
        # Generate log-normal data: exp(N(0,1)) is heavy-tailed
        torch.manual_seed(0)
        normal_samples = torch.randn(1000)
        log_normal_data = torch.exp(normal_samples)  # heavy-tailed

        log_scaler = LogScaler()
        zscore_scaler = ZScoreScaler()

        log_transformed = log_scaler.transform(log_normal_data)
        standardized = zscore_scaler.fit_transform(log_transformed)

        # After the full pipeline the output should be close to N(0, 1)
        assert abs(standardized.mean().item()) < 0.05
        assert abs(standardized.std().item() - 1.0) < 0.05

        # The raw data is *not* close to N(0, 1) — verify the pipeline helps
        raw_mean_offset = abs(log_normal_data.mean().item())
        pipeline_mean_offset = abs(standardized.mean().item())
        assert pipeline_mean_offset < raw_mean_offset

    def test_log_then_zscore_full_inverse_recovers_original(self) -> None:
        torch.manual_seed(42)
        log_normal_data = torch.exp(torch.randn(200))

        log_scaler = LogScaler()
        zscore_scaler = ZScoreScaler()

        # Forward
        log_transformed = log_scaler.transform(log_normal_data)
        standardized = zscore_scaler.fit_transform(log_transformed)

        # Inverse (reversed order)
        unscaled = zscore_scaler.inverse_transform(standardized)
        recovered = log_scaler.inverse_transform(unscaled)

        assert torch.allclose(recovered, log_normal_data, atol=1e-5)
