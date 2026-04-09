from __future__ import annotations

import io

import pytest
import torch

from torchscalers import MaxAbsScaler, MinMaxScaler, MixedDomainScaler, ZScoreScaler


def domain_a_data() -> torch.Tensor:
    return torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])


def domain_b_data() -> torch.Tensor:
    # Deliberately different scale to verify independence
    return torch.tensor([[100.0, 200.0], [300.0, 400.0], [500.0, 600.0]])


class TestMixedDomainScalerBasic:
    def test_different_scalers_per_domain(self) -> None:
        mds = MixedDomainScaler({"a": ZScoreScaler(), "b": MinMaxScaler()})
        mds.fit("a", domain_a_data())
        mds.fit("b", domain_b_data())

        result_a = mds.transform("a", domain_a_data())
        result_b = mds.transform("b", domain_b_data())

        # ZScoreScaler: column means should be ~0
        assert torch.allclose(result_a.mean(dim=0), torch.zeros(2), atol=1e-5)
        # MinMaxScaler: outputs in [0, 1]
        assert result_b.min().item() >= 0.0
        assert result_b.max().item() <= 1.0

    def test_scalers_are_independent_objects(self) -> None:
        zscore = ZScoreScaler()
        maxabs = MaxAbsScaler()
        mds = MixedDomainScaler({"a": zscore, "b": maxabs})
        assert mds._scalers["a"] is zscore
        assert mds._scalers["b"] is maxabs

    def test_fit_returns_self(self) -> None:
        mds = MixedDomainScaler({"a": ZScoreScaler()})
        result = mds.fit("a", domain_a_data())
        assert result is mds

    def test_inverse_transform_round_trip(self) -> None:
        mds = MixedDomainScaler({"a": ZScoreScaler(), "b": MinMaxScaler()})
        for domain, data in [("a", domain_a_data()), ("b", domain_b_data())]:
            mds.fit(domain, data)
            normed = mds.transform(domain, data)
            recovered = mds.inverse_transform(domain, normed)
            assert torch.allclose(recovered, data, atol=1e-5)

    def test_empty_constructor(self) -> None:
        mds = MixedDomainScaler()
        assert len(mds._scalers) == 0


class TestMixedDomainScalerRegister:
    def test_register_adds_domain(self) -> None:
        mds = MixedDomainScaler()
        mds.register("a", ZScoreScaler())
        mds.fit("a", domain_a_data())
        result = mds.transform("a", domain_a_data())
        assert torch.allclose(result.mean(dim=0), torch.zeros(2), atol=1e-5)

    def test_register_returns_self(self) -> None:
        mds = MixedDomainScaler()
        result = mds.register("a", ZScoreScaler())
        assert result is mds

    def test_register_replaces_existing_scaler(self) -> None:
        mds = MixedDomainScaler({"a": ZScoreScaler()})
        new_scaler = MinMaxScaler()
        mds.register("a", new_scaler)
        assert mds._scalers["a"] is new_scaler

    def test_register_chaining(self) -> None:
        mds = MixedDomainScaler()
        mds.register("a", ZScoreScaler()).register("b", MinMaxScaler())
        assert "a" in mds._scalers
        assert "b" in mds._scalers

    def test_fit_before_register_raises(self) -> None:
        mds = MixedDomainScaler()
        with pytest.raises(KeyError, match="unregistered"):
            mds.fit("unregistered", domain_a_data())


class TestMixedDomainScalerErrors:
    def test_transform_unregistered_raises(self) -> None:
        mds = MixedDomainScaler({"a": ZScoreScaler()})
        mds.fit("a", domain_a_data())
        with pytest.raises(KeyError, match="unknown"):
            mds.transform("unknown", domain_a_data())

    def test_inverse_transform_unregistered_raises(self) -> None:
        mds = MixedDomainScaler({"a": ZScoreScaler()})
        mds.fit("a", domain_a_data())
        with pytest.raises(KeyError, match="unknown"):
            mds.inverse_transform("unknown", domain_a_data())


class TestMixedDomainScalerModule:
    def test_state_dict_contains_all_domains(self) -> None:
        mds = MixedDomainScaler({"a": ZScoreScaler(), "b": MinMaxScaler()})
        mds.fit("a", domain_a_data())
        mds.fit("b", domain_b_data())
        keys = list(mds.state_dict().keys())
        assert any("a" in k for k in keys)
        assert any("b" in k for k in keys)

    def test_state_dict_round_trip(self) -> None:
        mds = MixedDomainScaler({"a": ZScoreScaler(), "b": MinMaxScaler()})
        mds.fit("a", domain_a_data())
        mds.fit("b", domain_b_data())
        expected_a = mds.transform("a", domain_a_data())
        expected_b = mds.transform("b", domain_b_data())

        buf = io.BytesIO()
        torch.save(mds.state_dict(), buf)
        buf.seek(0)

        mds2 = MixedDomainScaler({"a": ZScoreScaler(), "b": MinMaxScaler()})
        mds2.load_state_dict(torch.load(buf, weights_only=True))

        assert torch.allclose(mds2.transform("a", domain_a_data()), expected_a)
        assert torch.allclose(mds2.transform("b", domain_b_data()), expected_b)
