from __future__ import annotations

import torch
from torch import Tensor

from torchscalers.scaler import Scaler


class LogScaler(Scaler):
    """Apply a natural log transformation: ``log(x + eps)``.

    Useful for compressing heavy-tailed or skewed distributions (e.g. power-law
    data, financial returns, sensor readings).  Can be chained with another
    scaler such as :class:`~torchscalers.ZScoreScaler` to both compress the
    dynamic range and standardize.

    Requires no fitting step — calling :meth:`fit` is a no-op.

    Parameters
    ----------
    eps:
        Small constant added to *x* before taking the log to keep the
        argument strictly positive.  Defaults to ``1e-8``.
    """

    fitted: Tensor

    def __init__(self, eps: float = 1e-8) -> None:
        super().__init__()
        self.eps = eps
        self.register_buffer("fitted", torch.tensor(True))

    def fit(self, x: Tensor) -> "LogScaler":  # noqa: ARG002
        """No-op — no statistics to compute.

        Parameters
        ----------
        x:
            Ignored.

        Returns
        -------
        self
        """
        return self

    def transform(self, x: Tensor) -> Tensor:
        """Apply ``log(x + eps)`` element-wise.

        Parameters
        ----------
        x:
            Input tensor.  Values must satisfy ``x + eps > 0``.

        Returns
        -------
        Tensor
            Log-transformed tensor with the same shape as *x*.
        """
        return torch.log(x + self.eps)

    def inverse_transform(self, x: Tensor) -> Tensor:
        """Reverse the log transform: ``exp(x) - eps``.

        Parameters
        ----------
        x:
            Log-transformed tensor.

        Returns
        -------
        Tensor
            Tensor in the original scale, same shape as *x*.
        """
        return torch.exp(x) - self.eps
