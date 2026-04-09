"""PyTorch Lightning example: scalers in a DataModule with checkpointing.

Shows how to:
- Fit scalers inside a LightningDataModule (on train split only, to avoid
  data leakage).
- Pass the fitted scaler instances to a LightningModule so they become part
  of the model's state_dict and are saved automatically with every checkpoint.
- Use DataModule.state_dict / load_state_dict hooks to also persist scaler
  stats in the DataModule checkpoint (optional — documented here because the
  pattern is useful when the DataModule lives independently of the model).
- Restore a run from a checkpoint.

Run with (requires the 'examples' optional dependency group):
    uv sync --extra examples
    uv run examples/lightning_example.py
"""

import lightning as L
import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset

from torchscalers import ZScoreScaler

# ---------------------------------------------------------------------------
# Synthetic dataset
# ---------------------------------------------------------------------------
torch.manual_seed(0)

N_TOTAL, N_FEATURES, N_TARGETS = 500, 8, 1
X_all = torch.randn(N_TOTAL, N_FEATURES) * 3 + 5
y_all = torch.randn(N_TOTAL, N_TARGETS) * 10 - 2


# ---------------------------------------------------------------------------
# DataModule
# ---------------------------------------------------------------------------
class ExampleDataModule(L.LightningDataModule):
    """DataModule that fits scalers on the training split only."""

    def __init__(self, X: Tensor, y: Tensor, batch_size: int = 64) -> None:
        super().__init__()
        self.X = X
        self.y = y
        self.batch_size = batch_size

        # Scalers are kept here so they can be passed to the model after
        # setup() has been called manually (see usage pattern below).
        self.feature_scaler = ZScoreScaler()
        self.target_scaler = ZScoreScaler()

    def setup(self, stage: str) -> None:
        if stage == "fit":
            n_train = int(len(self.X) * 0.8)
            X_train, X_val = self.X[:n_train], self.X[n_train:]
            y_train, y_val = self.y[:n_train], self.y[n_train:]

            # Fit on training data only to prevent leakage into validation.
            self.feature_scaler.fit(X_train)
            self.target_scaler.fit(y_train)

            self.train_dataset = TensorDataset(X_train, y_train)
            self.val_dataset = TensorDataset(X_val, y_val)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_dataset, batch_size=self.batch_size)

    # ------------------------------------------------------------------
    # Optional DataModule checkpoint hooks.
    #
    # Because the scalers will also live inside the model as child modules
    # (see ExampleModel below), their statistics are already saved in the
    # model checkpoint automatically.  These hooks are shown here for cases
    # where the DataModule must be restored independently of the model.
    # ------------------------------------------------------------------
    def state_dict(self) -> dict:
        return {
            "feature_scaler": self.feature_scaler.state_dict(),
            "target_scaler": self.target_scaler.state_dict(),
        }

    def load_state_dict(self, state_dict: dict) -> None:
        self.feature_scaler.load_state_dict(state_dict["feature_scaler"])
        self.target_scaler.load_state_dict(state_dict["target_scaler"])


# ---------------------------------------------------------------------------
# LightningModule
# ---------------------------------------------------------------------------
class ExampleModel(L.LightningModule):
    """Linear regression model that owns the fitted scalers as submodules."""

    def __init__(
        self,
        feature_scaler: ZScoreScaler,
        target_scaler: ZScoreScaler,
        in_features: int,
        out_features: int,
    ) -> None:
        super().__init__()
        # Storing the scalers as attributes registers them as child modules,
        # so their buffers (fitted statistics) are saved in every checkpoint.
        self.feature_scaler = feature_scaler
        self.target_scaler = target_scaler
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x: Tensor) -> Tensor:
        # scaler(x) calls scaler.transform(x) via forward().
        return self.linear(self.feature_scaler(x))

    def training_step(self, batch: tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        x, y = batch
        pred = self(x)
        target = self.target_scaler(y)  # scale targets into normalised space
        loss = nn.functional.mse_loss(pred, target)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch: tuple[Tensor, Tensor], batch_idx: int) -> None:
        x, y = batch
        pred = self(x)
        target = self.target_scaler(y)
        loss = nn.functional.mse_loss(pred, target)
        self.log("val_loss", loss, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)


# ---------------------------------------------------------------------------
# Usage pattern
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    dm = ExampleDataModule(X_all, y_all, batch_size=64)

    # Call setup() manually before constructing the model so the fitted
    # scalers can be passed in as constructor arguments.
    dm.setup(stage="fit")

    model = ExampleModel(
        feature_scaler=dm.feature_scaler,
        target_scaler=dm.target_scaler,
        in_features=N_FEATURES,
        out_features=N_TARGETS,
    )

    # Scaler statistics are now part of model.state_dict() and will be saved
    # automatically in every checkpoint written by the Trainer.
    trainer = L.Trainer(
        max_epochs=3,
        enable_model_summary=False,
        logger=False,
    )
    trainer.fit(model, dm)

    # -----------------------------------------------------------------------
    # Restore from the best checkpoint (Lightning writes one automatically)
    # -----------------------------------------------------------------------
    ckpt_path = trainer.checkpoint_callback.best_model_path
    if ckpt_path:
        # Reconstruct the DataModule and set up scalers before loading.
        dm2 = ExampleDataModule(X_all, y_all)
        dm2.setup(stage="fit")

        restored_model = ExampleModel(
            feature_scaler=dm2.feature_scaler,
            target_scaler=dm2.target_scaler,
            in_features=N_FEATURES,
            out_features=N_TARGETS,
        )
        restored_model.load_state_dict(
            torch.load(ckpt_path, weights_only=True)["state_dict"]
        )
        print(f"Model restored from {ckpt_path!r}")

        restored_model.eval()
        with torch.no_grad():
            X_sample = X_all[:5]
            pred_scaled = restored_model(X_sample)
            pred_orig = restored_model.target_scaler.inverse_transform(pred_scaled)
        print(f"Predictions (original scale): {pred_orig.squeeze().tolist()}")
