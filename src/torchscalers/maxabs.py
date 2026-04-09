from __future__ import annotations

import torch
from torch import Tensor

from torchscalers.scaler import Scaler


class MaxAbsScaler(Scaler):
    """Scale features by their maximum absolute value to the range ``[-1, 1]``.

    Does not shift the data, so sparsity is preserved.  Useful for data
    that is already centred around zero or that is sparse.

    Statistics are computed column-wise (over the sample dimension) for 2-D
    inputs ``[N, D]``, or as a single scalar for 1-D inputs ``[N]``.

    The fitted ``max_abs_``, and ``fitted`` flag are stored as
    :class:`torch.nn.Module` buffers so they are included in
    :meth:`state_dict` and move automatically with ``.to(device)``.

    Parameters
    ----------
    eps:
        Minimum value ``max_abs_`` is clamped to, preventing division by
        zero for all-zero features.  Defaults to ``1e-8``.
    """

    max_abs_: Tensor
    fitted: Tensor

    def __init__(self, eps: float = 1e-8) -> None:
        super().__init__()
        self.eps = eps
        self.register_buffer("max_abs_", torch.empty(0))
        self.register_buffer("fitted", torch.tensor(False))

    def fit(self, x: Tensor) -> "MaxAbsScaler":
        """Compute and store per-feature maximum absolute value from *x*.

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
            max_abs_ = x.abs().max()
        else:
            max_abs_ = x.abs().max(dim=0).values

        self.register_buffer("max_abs_", max_abs_.clamp(min=self.eps))
        self.fitted.fill_(True)
        return self

    def transform(self, x: Tensor) -> Tensor:
        """Scale *x* to ``[-1, 1]`` using the fitted maximum absolute value.

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
            raise RuntimeError("MaxAbsScaler: call fit() before transform().")
        return x / self.max_abs_

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
            raise RuntimeError("MaxAbsScaler: call fit() before inverse_transform().")
        return x * self.max_abs_
