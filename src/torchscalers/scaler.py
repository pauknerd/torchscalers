from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch
import torch.nn as nn
from torch import Tensor


class Scaler(nn.Module, ABC):
    """Abstract base class for all scalers.

    Subclasses must implement :meth:`fit`, :meth:`transform`, and
    :meth:`inverse_transform`. The concrete :meth:`fit_transform` method is
    provided here and delegates to those two methods, so subclasses get it for
    free.

    All scalers inherit from :class:`torch.nn.Module` so that they can be
    embedded in larger models, participate in :meth:`state_dict` checkpointing,
    and have their buffers moved automatically via ``.to(device)``.

    Calling a scaler instance directly (``scaler(x)``) is equivalent to
    calling :meth:`transform` and is the idiomatic way to use a scaler inside
    a model's ``forward`` method or in an :class:`torch.nn.Sequential` pipeline.
    """

    @abstractmethod
    def fit(self, x: Tensor) -> "Scaler":
        """Compute and store statistics from *x*.

        Parameters
        ----------
        x:
            Input tensor of shape ``[N]`` or ``[N, D]``.

        Returns
        -------
        self
            Returns the scaler instance to allow method chaining.
        """

    @abstractmethod
    def transform(self, x: Tensor) -> Tensor:
        """Apply the scaler to *x* using previously fitted statistics.

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

    @abstractmethod
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

    def fit_transform(self, x: Tensor) -> Tensor:
        """Fit to *x* and immediately transform it.

        Equivalent to calling ``self.fit(x).transform(x)``.

        Parameters
        ----------
        x:
            Input tensor of shape ``[N]`` or ``[N, D]``.

        Returns
        -------
        Tensor
            Scaled tensor with the same shape as *x*.
        """
        return self.fit(x).transform(x)

    def forward(self, x: Tensor) -> Tensor:
        """Apply the scaler to *x*; delegates to :meth:`transform`.

        This makes scalers usable as standard :class:`torch.nn.Module` objects
        — ``scaler(x)`` works identically to ``scaler.transform(x)``.

        Parameters
        ----------
        x:
            Input tensor of shape ``[N]`` or ``[N, D]``.

        Returns
        -------
        Tensor
            Scaled tensor with the same shape as *x*.
        """
        return self.transform(x)

    def _load_from_state_dict(
        self,
        state_dict: dict[str, Any],
        prefix: str,
        local_metadata: dict[str, Any],
        strict: bool,
        missing_keys: list[str],
        unexpected_keys: list[str],
        error_msgs: list[str],
    ) -> None:
        """Pre-resize placeholder buffers before the standard state_dict copy.

        Subclasses use ``torch.empty(0)`` as a lazy placeholder for buffers
        whose shape is only known after ``fit()``.  When loading a fitted
        state dict into a fresh (unfitted) instance, the placeholder shape
        ``[0]`` would mismatch the stored tensor and cause a copy error.
        This override resizes any such placeholder to match the shape in
        *state_dict* before delegating to the standard implementation.
        """
        for name, buf in self._buffers.items():
            key = prefix + name
            if (
                key in state_dict
                and buf is not None
                and buf.shape != state_dict[key].shape
            ):
                self.register_buffer(name, torch.empty_like(state_dict[key]))
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )
