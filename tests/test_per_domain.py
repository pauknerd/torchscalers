from __future__ import annotations

import pytest
import torch

from torchscalers import MinMaxScaler, PerDomainScaler, ZScoreScaler


def domain_a_data() -> torch.Tensor:
    return torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])


def domain_b_data() -> torch.Tensor:
    # Deliberately different scale so we can verify independence
    return torch.tensor([[100.0, 200.0], [300.0, 400.0], [500.0, 600.0]])


class TestPerDomainScalerZScore:
    def test_domains_fitted_independently(self) -> None:
        pdn = PerDomainScaler(ZScoreScaler)
        pdn.fit("a", domain_a_data())
        pdn.fit("b", domain_b_data())

        # Each domain's internal normalizer should have its own mean
        mean_a = pdn._scalers["a"].mean  # type: ignore[union-attr]
        mean_b = pdn._scalers["b"].mean  # type: ignore[union-attr]
        assert not torch.allclose(mean_a, mean_b)

    def test_transform_produces_zero_mean_per_domain(self) -> None:
        pdn = PerDomainScaler(ZScoreScaler)
        pdn.fit("a", domain_a_data())
        pdn.fit("b", domain_b_data())

        result_a = pdn.transform("a", domain_a_data())
        result_b = pdn.transform("b", domain_b_data())

        assert torch.allclose(result_a.mean(dim=0), torch.zeros(2), atol=1e-5)
        assert torch.allclose(result_b.mean(dim=0), torch.zeros(2), atol=1e-5)

    def test_inverse_transform_round_trip(self) -> None:
        pdn = PerDomainScaler(ZScoreScaler)
        pdn.fit("a", domain_a_data())
        normed = pdn.transform("a", domain_a_data())
        recovered = pdn.inverse_transform("a", normed)
        assert torch.allclose(recovered, domain_a_data(), atol=1e-5)

    def test_fit_returns_self(self) -> None:
        pdn = PerDomainScaler(ZScoreScaler)
        result = pdn.fit("a", domain_a_data())
        assert result is pdn

    def test_transform_unknown_domain_raises(self) -> None:
        pdn = PerDomainScaler(ZScoreScaler)
        with pytest.raises(KeyError, match="unknown"):
            pdn.transform("unknown", domain_a_data())

    def test_inverse_transform_unknown_domain_raises(self) -> None:
        pdn = PerDomainScaler(ZScoreScaler)
        with pytest.raises(KeyError, match="unknown"):
            pdn.inverse_transform("unknown", domain_a_data())


class TestPerDomainScalerMinMax:
    def test_works_with_minmax_normalizer(self) -> None:
        pdn = PerDomainScaler(MinMaxScaler)
        pdn.fit("a", domain_a_data())
        result = pdn.transform("a", domain_a_data())
        assert result.min().item() >= 0.0
        assert result.max().item() <= 1.0

    def test_domains_independent_with_minmax(self) -> None:
        pdn = PerDomainScaler(MinMaxScaler)
        pdn.fit("a", domain_a_data())
        pdn.fit("b", domain_b_data())

        min_a = pdn._scalers["a"].min_  # type: ignore[union-attr]
        min_b = pdn._scalers["b"].min_  # type: ignore[union-attr]
        assert not torch.allclose(min_a, min_b)

    def test_eps_kwarg_forwarded(self) -> None:
        pdn = PerDomainScaler(ZScoreScaler, eps=1e-4)
        pdn.fit("a", domain_a_data())
        norm = pdn._scalers["a"]
        assert norm.eps == 1e-4  # type: ignore[union-attr]


class TestPerDomainScalerModule:
    def test_state_dict_contains_all_domains(self) -> None:
        pdn = PerDomainScaler(ZScoreScaler)
        pdn.fit("a", domain_a_data())
        pdn.fit("b", domain_b_data())
        state = pdn.state_dict()
        keys = list(state.keys())
        assert any("a" in k for k in keys)
        assert any("b" in k for k in keys)
