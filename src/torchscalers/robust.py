from __future__ import annotations

import torch
from torch import Tensor

from torchscalers.scaler import Scaler


class RobustScaler(Scaler):
    """Scale features using statistics that are robust to outliers.

    Centers features by subtracting the median, then scales by the
    interquartile range (IQR = Q75 - Q25).  Because the median and IQR
    are not affected by extreme values, this scaler is well-suited to
    data containing outliers.

    Statistics are computed column-wise (over the sample dimension) for 2-D
    inputs ``[N, D]``, or as a single scalar for 1-D inputs ``[N]``.

    The fitted ``median_``, ``iqr_``, and ``fitted`` flag are stored as
    :class:`torch.nn.Module` buffers so they are included in
    :meth:`state_dict` and move automatically with ``.to(device)``.

    Parameters
    ----------
    eps:
        Minimum IQR value used as a fallback for constant features (where
        IQR is zero), preventing division by zero.  Defaults to ``1e-8``.
    """

    median_: Tensor
    iqr_: Tensor
    fitted: Tensor

    def __init__(self, eps: float = 1e-8) -> None:
        super().__init__()
        self.eps = eps
        self.register_buffer("median_", torch.empty(0))
        self.register_buffer("iqr_", torch.empty(0))
        self.register_buffer("fitted", torch.tensor(False))

    def fit(self, x: Tensor) -> "RobustScaler":
        """Compute and store per-feature median and IQR from *x*.

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
            median_ = x.median()
            q75 = x.quantile(0.75)
            q25 = x.quantile(0.25)
        else:
            median_ = x.median(dim=0).values
            q75 = x.quantile(0.75, dim=0)
            q25 = x.quantile(0.25, dim=0)

        self.register_buffer("median_", median_)
        self.register_buffer("iqr_", (q75 - q25).clamp(min=self.eps))
        self.fitted.fill_(True)
        return self

    def transform(self, x: Tensor) -> Tensor:
        """Center and scale *x* using the fitted median and IQR.

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
            raise RuntimeError("RobustScaler: call fit() before transform().")
        return (x - self.median_) / self.iqr_

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
            raise RuntimeError("RobustScaler: call fit() before inverse_transform().")
        return x * self.iqr_ + self.median_
