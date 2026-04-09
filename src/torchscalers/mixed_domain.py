from __future__ import annotations

import torch.nn as nn
from torch import Tensor

from torchscalers.scaler import Scaler


class MixedDomainScaler(nn.Module):
    """Apply a different scaler per string domain ID.

    Unlike :class:`~torchscalers.PerDomainScaler`, each domain can use a
    *different* :class:`Scaler` subclass with its own constructor arguments,
    making it suitable for heterogeneous feature spaces where different node or
    edge types require fundamentally different normalisation strategies.

    Scalers must be registered—either at construction time or via
    :meth:`register`—before calling :meth:`fit`.

    Each domain's scaler is stored inside an :class:`torch.nn.ModuleDict`
    (``_scalers``), so all per-domain statistics are part of the module's
    :meth:`state_dict` and move with ``.to(device)``.

    Parameters
    ----------
    scalers:
        Optional mapping of domain ID to pre-instantiated :class:`Scaler`
        objects.  Additional domains can be added later via :meth:`register`.

    Examples
    --------
    >>> from torchscalers import MixedDomainScaler, MaxAbsScaler, ZScoreScaler
    >>> mds = MixedDomainScaler({
    ...     "nodes": ZScoreScaler(eps=1e-8),
    ...     "edges": MaxAbsScaler(eps=1e-8),
    ... })
    >>> mds.fit("nodes", node_features)
    >>> mds.fit("edges", edge_features)
    >>> node_norm = mds.transform("nodes", node_features)
    """

    def __init__(
        self,
        scalers: dict[str, Scaler] | None = None,
    ) -> None:
        super().__init__()
        self._scalers: nn.ModuleDict = nn.ModuleDict()
        if scalers is not None:
            for domain_id, scaler in scalers.items():
                self._scalers[domain_id] = scaler

    def register(self, domain_id: str, scaler: Scaler) -> "MixedDomainScaler":
        """Register a scaler for *domain_id*.

        Can be called before or after fitting other domains.  If *domain_id*
        already exists, the existing scaler is replaced.

        Parameters
        ----------
        domain_id:
            String key identifying the domain.
        scaler:
            Pre-instantiated :class:`Scaler` to use for this domain.

        Returns
        -------
        self
            Returns the ``MixedDomainScaler`` instance to allow chaining.
        """
        self._scalers[domain_id] = scaler
        return self

    def fit(self, domain_id: str, x: Tensor) -> "MixedDomainScaler":
        """Fit the scaler for *domain_id* on *x*.

        Parameters
        ----------
        domain_id:
            String key identifying the domain (e.g. ``"nodes"``).
        x:
            Input tensor of shape ``[N]`` or ``[N, D]``.

        Returns
        -------
        self
            Returns the ``MixedDomainScaler`` instance to allow chaining.

        Raises
        ------
        KeyError
            If *domain_id* has not been registered yet.
        """
        if domain_id not in self._scalers:
            raise KeyError(
                f"MixedDomainScaler: domain '{domain_id}' has not been registered. "
                "Call register() first."
            )
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
            If *domain_id* has not been registered yet.
        """
        if domain_id not in self._scalers:
            raise KeyError(
                f"MixedDomainScaler: domain '{domain_id}' has not been registered. "
                "Call register() first."
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
            If *domain_id* has not been registered yet.
        """
        if domain_id not in self._scalers:
            raise KeyError(
                f"MixedDomainScaler: domain '{domain_id}' has not been registered. "
                "Call register() first."
            )
        return self._scalers[domain_id].inverse_transform(x)
