torchscalers
============

A simple collection of scalers for PyTorch pipelines.

All scalers are :class:`torch.nn.Module` subclasses. Their fitted statistics
are stored as module buffers, which means they:

- are included in :meth:`~torch.nn.Module.state_dict` and saved/restored with
  every checkpoint automatically,
- move to the correct device with :meth:`~torch.nn.Module.to`,
- work inside :class:`torch.nn.Sequential` pipelines (calling ``scaler(x)`` is
  equivalent to ``scaler.transform(x)``).

.. toctree::
   :maxdepth: 2
   :caption: Contents

   installation
   examples
   api
