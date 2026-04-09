from __future__ import annotations

from typing import Union

import torch
from torch import Tensor

from torchscalers.scaler import Scaler


class ShiftScaleScaler(Scaler):
    """Apply a pre-specified shift and scale transformation: ``(x + shift) * scale``.

    Unlike data-driven scalers, ``ShiftScaleScaler`` requires no fitting step —
    the ``shift`` and ``scale`` parameters are provided at construction time.
    Calling :meth:`fit` is a no-op.

    This is useful when the normalization statistics are already known (e.g.
    from a dataset publication or domain knowledge) and should not be inferred
    from the data.

    The ``shift_`` and ``scale_`` tensors are stored as
    :class:`torch.nn.Module` buffers so they are included in
    :meth:`state_dict` and move automatically with ``.to(device)``.

    Parameters
    ----------
    shift:
        Value(s) to add to *x* before scaling.  Scalar or 1-D tensor of
        length ``D`` for feature-wise shifts.
    scale:
        Value(s) to multiply ``(x + shift)`` by.  Must be strictly positive.
        Scalar or 1-D tensor of length ``D``.

    Raises
    ------
    ValueError
        If any value in *scale* is not strictly positive.
    """

    shift_: Tensor
    scale_: Tensor
    fitted: Tensor

    def __init__(
        self,
        shift: Union[float, Tensor],
        scale: Union[float, Tensor],
    ) -> None:
        super().__init__()
        shift_t = torch.as_tensor(shift, dtype=torch.float32)
        scale_t = torch.as_tensor(scale, dtype=torch.float32)

        if (scale_t <= 0).any():
            raise ValueError(
                "ShiftScaleScaler: all scale values must be strictly positive."
            )

        self.register_buffer("shift_", shift_t)
        self.register_buffer("scale_", scale_t)
        self.register_buffer("fitted", torch.tensor(True))

    def fit(self, x: Tensor) -> "ShiftScaleScaler":  # noqa: ARG002
        """No-op — statistics are pre-specified at construction time.

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
        """Apply ``(x + shift) * scale`` to *x*.

        Parameters
        ----------
        x:
            Input tensor of shape ``[N]`` or ``[N, D]``.

        Returns
        -------
        Tensor
            Scaled tensor with the same shape as *x*.
        """
        return (x + self.shift_) * self.scale_

    def inverse_transform(self, x: Tensor) -> Tensor:
        """Reverse the transformation: ``x / scale - shift``.

        Parameters
        ----------
        x:
            Scaled tensor of shape ``[N]`` or ``[N, D]``.

        Returns
        -------
        Tensor
            Tensor in the original scale, same shape as *x*.
        """
        return x / self.scale_ - self.shift_
