from __future__ import annotations

from typing import Any

import torch.nn as nn
from torch import Tensor

from torchscalers.scaler import Scaler


class PerDomainScaler(nn.Module):
    """Apply a separate scaler instance per string domain ID.

    Useful in GNN pipelines and other settings where different node or edge
    types have distinct feature distributions that should be normalized
    independently.

    Each domain's scaler is stored inside an :class:`torch.nn.ModuleDict`
    (``_scalers``), so all per-domain statistics are part of the module's
    :meth:`state_dict` and move with ``.to(device)``.

    Parameters
    ----------
    scaler_class:
        The :class:`Scaler` subclass to instantiate for each new domain
        (e.g. :class:`~torchscalers.ZScoreScaler`).
    **scaler_kwargs:
        Keyword arguments forwarded to *scaler_class* when a new instance
        is created (e.g. ``eps=1e-6``).

    Examples
    --------
    >>> from torchscalers import PerDomainScaler, ZScoreScaler
    >>> pdn = PerDomainScaler(ZScoreScaler, eps=1e-8)
    >>> pdn.fit("nodes", node_features)
    >>> pdn.fit("edges", edge_features)
    >>> node_norm = pdn.transform("nodes", node_features)
    """

    def __init__(
        self,
        scaler_class: type[Scaler],
        **scaler_kwargs: Any,
    ) -> None:
        super().__init__()
        self._scaler_class = scaler_class
        self._scaler_kwargs = scaler_kwargs
        self._scalers: nn.ModuleDict = nn.ModuleDict()

    def fit(self, domain_id: str, x: Tensor) -> "PerDomainScaler":
        """Fit a scaler for *domain_id* on *x*.

        A new scaler instance is created the first time a *domain_id* is
        seen. Subsequent calls re-fit the existing instance.

        Parameters
        ----------
        domain_id:
            String key identifying the domain (e.g. ``"nodes"``).
        x:
            Input tensor of shape ``[N]`` or ``[N, D]``.

        Returns
        -------
        self
            Returns the ``PerDomainScaler`` instance to allow chaining.
        """
        if domain_id not in self._scalers:
            self._scalers[domain_id] = self._scaler_class(**self._scaler_kwargs)
        self._scalers[domain_id].fit(x)
        return self

    def transform(self, domain_id: str, x: Tensor) -> Tensor:
        """Normalize *x* using the fitted scaler for *domain_id*.

        Parameters
        ----------
        domain_id:
            String key identifying the domain.
        x:
            Input tensor of shape ``[N]`` or ``[N, D]``.

        Returns
        -------
        Tensor
            Scaled tensor with the same shape as *x*.

        Raises
        ------
        KeyError
            If *domain_id* has not been fitted yet.
        """
        if domain_id not in self._scalers:
            raise KeyError(
                f"PerDomainScaler: domain '{domain_id}' has not been fitted. "
                "Call fit() first."
            )
        return self._scalers[domain_id].transform(x)

    def inverse_transform(self, domain_id: str, x: Tensor) -> Tensor:
        """Reverse the scaling for *domain_id*.

        Parameters
        ----------
        domain_id:
            String key identifying the domain.
        x:
            Scaled tensor of shape ``[N]`` or ``[N, D]``.

        Returns
        -------
        Tensor
            Tensor in the original scale, same shape as *x*.

        Raises
        ------
        KeyError
            If *domain_id* has not been fitted yet.
        """
        if domain_id not in self._scalers:
            raise KeyError(
                f"PerDomainScaler: domain '{domain_id}' has not been fitted. "
                "Call fit() first."
            )
        return self._scalers[domain_id].inverse_transform(x)
