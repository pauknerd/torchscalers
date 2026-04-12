# torchscalers

[![CI](https://github.com/pauknerd/torchscalers/actions/workflows/ci.yml/badge.svg)](https://github.com/pauknerd/torchscalers/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/pauknerd/torchscalers/graph/badge.svg)](https://codecov.io/gh/pauknerd/torchscalers)
[![PyPI](https://img.shields.io/pypi/v/torchscalers)](https://pypi.org/project/torchscalers/)
[![Docs](https://img.shields.io/badge/docs-github%20pages-blue)](https://pauknerd.github.io/torchscalers)

A simple collection of scalers for PyTorch pipelines.

All scalers are `nn.Module` subclasses. Their fitted statistics are stored as
module buffers, which means they:

- are included in `model.state_dict()` and saved/restored with every checkpoint automatically,
- move to the correct device with `.to(device)`,
- work inside `nn.Sequential` pipelines (calling `scaler(x)` is equivalent to `scaler.transform(x)`).

## Installation

```bash
uv add torchscalers
```

To run the [example scripts](examples/), install the optional `examples` group
(adds [Lightning](https://lightning.ai)):

```bash
uv sync --extra examples
```

## Quick start

```python
import torch
from torchscalers import ZScoreScaler

scaler = ZScoreScaler()

X_train = torch.randn(500, 8)
X_test  = torch.randn(100, 8)

# Fit on training data, then scale
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler(X_test)                  # equivalent to scaler.transform(X_test)

# Recover original scale
X_recovered = scaler.inverse_transform(X_test_scaled)
```

## Embedding in a model

Scalers stored as child modules are checkpointed automatically — no extra
steps required.

```python
import torch
import torch.nn as nn
from torchscalers import ZScoreScaler


class MyModel(nn.Module):
    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.feature_scaler = ZScoreScaler()
        self.target_scaler  = ZScoreScaler()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        return self.linear(self.feature_scaler(x))


model = MyModel(8, 1)

# Fit scalers on training data before the training loop.
model.feature_scaler.fit(X_train)
model.target_scaler.fit(y_train)

# Save — scaler statistics are included in state_dict automatically.
torch.save(model.state_dict(), "checkpoint.pt")

# Reload into a fresh model.
fresh = MyModel(8, 1)
fresh.load_state_dict(torch.load("checkpoint.pt", weights_only=True))

# Inverse-transform predictions back to the original target scale.
with torch.no_grad():
    pred_scaled = fresh(X_test)
    pred_orig   = fresh.target_scaler.inverse_transform(pred_scaled)
```

See [examples/pytorch_example.py](examples/pytorch_example.py) for the full
runnable script.

## PyTorch Lightning

Fit the scalers inside `DataModule.setup()` (on the training split only to
avoid data leakage), then pass the fitted instances to the `LightningModule`
so they become part of `model.state_dict()`.

```python
import lightning as L
from torchscalers import ZScoreScaler


class MyDataModule(L.LightningDataModule):
    def __init__(self, X, y):
        super().__init__()
        self.X, self.y = X, y
        self.feature_scaler = ZScoreScaler()
        self.target_scaler  = ZScoreScaler()

    def setup(self, stage):
        if stage == "fit":
            n = int(len(self.X) * 0.8)
            self.feature_scaler.fit(self.X[:n])
            self.target_scaler.fit(self.y[:n])
            # build train/val datasets …


class MyModel(L.LightningModule):
    def __init__(self, feature_scaler, target_scaler):
        super().__init__()
        # Storing as attributes registers them as child modules.
        self.feature_scaler = feature_scaler
        self.target_scaler  = target_scaler
        # define layers …

    def forward(self, x):
        return self.net(self.feature_scaler(x))

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss = F.mse_loss(self(x), self.target_scaler(y))
        self.log("train_loss", loss)
        return loss


# Call setup() manually so the fitted scalers can be passed to the model.
dm = MyDataModule(X_all, y_all)
dm.setup(stage="fit")

model = MyModel(
    feature_scaler=dm.feature_scaler,
    target_scaler=dm.target_scaler,
)

trainer = L.Trainer(max_epochs=10)
trainer.fit(model, dm)
# Scaler statistics are saved in every checkpoint automatically.
```

See [examples/lightning_example.py](examples/lightning_example.py) for the
full runnable script, including checkpoint restoration and the optional
DataModule `state_dict` / `load_state_dict` hooks.

## Available scalers

| Class               | Description                                                             |
| ------------------- | ----------------------------------------------------------------------- |
| `ZScoreScaler`      | Standardise to zero mean and unit variance.                             |
| `MinMaxScaler`      | Scale to the range \[0, 1\] using per-feature min/max.                  |
| `MaxAbsScaler`      | Scale to the range \[−1, 1\] by dividing by the maximum absolute value. |
| `RobustScaler`      | Scale using statistics robust to outliers (median and IQR).             |
| `ShiftScaleScaler`  | Apply a pre-specified `(x + shift) * scale` transformation.             |
| `LogScaler`         | Apply a log transformation: `log(x + eps)`.                             |
| `PerDomainScaler`   | Apply a separate scaler per string domain ID.                           |
| `MixedDomainScaler` | Apply a different scaler type per string domain ID.                     |

## Contributing

Contributions and feedback are more than welcome! Here is the standard workflow:

1. **Fork** the repository and clone your fork locally.
2. **Create a branch** for your change:
   ```bash
   git checkout -b my-feature
   ```
3. **Install dev dependencies:**
   ```bash
   uv sync --extra dev
   ```
4. **Make your changes**, then lint and format:
   ```bash
   ruff check src/
   ruff format src/
   ```
5. **Run the test suite** (coverage must stay above 80 %):
   ```bash
   pytest
   ```
6. **Open a pull request** against `master` describing what you changed and why.

Please open an issue first for larger changes or new scalers so we can discuss the design before implementation.

## License

This project is licensed under the [MIT License](LICENSE).
