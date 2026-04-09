from __future__ import annotations

import torch
from torch import Tensor

from torchscalers.scaler import Scaler


class MinMaxScaler(Scaler):
    """Scale features to the range ``[0, 1]`` using per-feature min/max statistics.

    Statistics are computed column-wise (over the sample dimension) for 2-D
    inputs ``[N, D]``, or as a single scalar for 1-D inputs ``[N]``.

    The fitted ``min_``, ``max_``, and ``fitted`` flag are stored as
    :class:`torch.nn.Module` buffers so they are included in
    :meth:`state_dict` and move automatically with ``.to(device)``.

    Parameters
    ----------
    eps:
        Minimum range value used as a fallback for constant features (where
        ``max == min``), preventing division by zero.  Defaults to ``1e-8``.
    """

    min_: Tensor
    max_: Tensor
    fitted: Tensor

    def __init__(self, eps: float = 1e-8) -> None:
        super().__init__()
        self.eps = eps
        self.register_buffer("min_", torch.empty(0))
        self.register_buffer("max_", torch.empty(0))
        self.register_buffer("fitted", torch.tensor(False))

    def fit(self, x: Tensor) -> "MinMaxScaler":
        """Compute and store per-feature min and max from *x*.

        Parameters
        ----------
        x:
            Input tensor of shape ``[N]`` or ``[N, D]``.

        Returns
        -------
        self
            Returns the scaler instance to allow method chaining.
        """
        if x.dim() == 1:
            min_ = x.min()
            max_ = x.max()
        else:
            min_ = x.min(dim=0).values
            max_ = x.max(dim=0).values

        self.register_buffer("min_", min_)
        self.register_buffer("max_", max_)
        self.fitted.fill_(True)
        return self

    def transform(self, x: Tensor) -> Tensor:
        """Scale *x* to ``[0, 1]`` using the fitted min and max.

        Parameters
        ----------
        x:
            Input tensor of shape ``[N]`` or ``[N, D]``.

        Returns
        -------
        Tensor
            Scaled tensor with the same shape as *x*.

        Raises
        ------
        RuntimeError
            If :meth:`fit` has not been called yet.
        """
        if not self.fitted.item():
            raise RuntimeError("MinMaxScaler: call fit() before transform().")
        range_ = (self.max_ - self.min_).clamp(min=self.eps)
        return (x - self.min_) / range_

    def inverse_transform(self, x: Tensor) -> Tensor:
        """Reverse the scaling applied by :meth:`transform`.

        Parameters
        ----------
        x:
            Scaled tensor of shape ``[N]`` or ``[N, D]``.

        Returns
        -------
        Tensor
            Tensor in the original scale, same shape as *x*.

        Raises
        ------
        RuntimeError
            If :meth:`fit` has not been called yet.
        """
        if not self.fitted.item():
            raise RuntimeError("MinMaxScaler: call fit() before inverse_transform().")
        range_ = (self.max_ - self.min_).clamp(min=self.eps)
        return x * range_ + self.min_
