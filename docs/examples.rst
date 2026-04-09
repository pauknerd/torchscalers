Examples
========

Pure PyTorch
------------

Shows how to embed scalers in an :class:`torch.nn.Module`, fit them on
training data, run a mini training loop, and round-trip through a checkpoint:

.. literalinclude:: ../examples/pytorch_example.py
   :language: python
   :linenos:

.. code-block:: bash

   uv run examples/pytorch_example.py

PyTorch Lightning
-----------------

Shows how to fit scalers inside a ``LightningDataModule`` (on the train split
only, to prevent data leakage), pass them to a ``LightningModule`` so they are
checkpointed automatically, and restore a run from the best checkpoint:

.. literalinclude:: ../examples/lightning_example.py
   :language: python
   :linenos:

Requires the optional ``examples`` dependency group:

.. code-block:: bash

   uv sync --extra examples
   uv run examples/lightning_example.py
