from __future__ import annotations

import torch
from torch import Tensor

from torchscalers.scaler import Scaler


class ZScoreScaler(Scaler):
    """Standardize features to zero mean and unit variance (z-score normalization).

    Statistics are computed column-wise (over the sample dimension) for 2-D
    inputs ``[N, D]``, or as a single scalar for 1-D inputs ``[N]``.

    The fitted ``mean``, ``std``, and ``fitted`` flag are stored as
    :class:`torch.nn.Module` buffers so they are included in
    :meth:`state_dict` and move automatically with ``.to(device)``.

    Parameters
    ----------
    eps:
        Minimum value the standard deviation is clamped to, preventing
        division by zero for constant features.  Defaults to ``1e-8``.
    """

    mean: Tensor
    std: Tensor
    fitted: Tensor

    def __init__(self, eps: float = 1e-8) -> None:
        super().__init__()
        self.eps = eps
        self.register_buffer("mean", torch.empty(0))
        self.register_buffer("std", torch.empty(0))
        self.register_buffer("fitted", torch.tensor(False))

    def fit(self, x: Tensor) -> "ZScoreScaler":
        """Compute and store mean and standard deviation from *x*.

        Parameters
        ----------
        x:
            Input tensor of shape ``[N]`` or ``[N, D]``.

        Returns
        -------
        self
            Returns the normalizer instance to allow method chaining.
        """
        if x.dim() == 1:
            mean = x.mean()
            std = x.std()
        else:
            mean = x.mean(dim=0)
            std = x.std(dim=0)

        self.register_buffer("mean", mean)
        self.register_buffer("std", std.clamp(min=self.eps))
        self.fitted.fill_(True)
        return self

    def transform(self, x: Tensor) -> Tensor:
        """Normalize *x* using the fitted mean and standard deviation.

        Parameters
        ----------
        x:
            Input tensor of shape ``[N]`` or ``[N, D]``.

        Returns
        -------
        Tensor
            Standardized tensor with the same shape as *x*.

        Raises
        ------
        RuntimeError
            If :meth:`fit` has not been called yet.
        """
        if not self.fitted.item():
            raise RuntimeError("ZScoreScaler: call fit() before transform().")
        return (x - self.mean) / self.std

    def inverse_transform(self, x: Tensor) -> Tensor:
        """Reverse the standardization applied by :meth:`transform`.

        Parameters
        ----------
        x:
            Standardized tensor of shape ``[N]`` or ``[N, D]``.

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
            raise RuntimeError("ZScoreScaler: call fit() before inverse_transform().")
        return x * self.std + self.mean
