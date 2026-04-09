Installation
============

Install from PyPI using **uv** (recommended):

.. code-block:: bash

   uv add torchscalers

Or with **pip**:

.. code-block:: bash

   pip install torchscalers

PyTorch (``torch>=2.0``) is the only runtime dependency and is installed
automatically.

Example scripts
---------------

The `example scripts <https://github.com/pauknerd/torchscalers/tree/main/examples>`_
require `Lightning <https://lightning.ai>`_.  Install the optional
``examples`` group to pull it in:

.. code-block:: bash

   uv sync --extra examples
