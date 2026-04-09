"""Pure PyTorch example: embedding scalers in a model with checkpointing.

Shows how to:
- Embed ZScoreScaler instances as nn.Module submodules.
- Fit the scalers on training data before training begins.
- Save and reload the full model (scaler statistics are included automatically
  in the state_dict because they are registered as nn.Module buffers).

Run with:
    uv run examples/pytorch_example.py
"""

import os
import tempfile

import torch
import torch.nn as nn

from torchscalers import ZScoreScaler

# ---------------------------------------------------------------------------
# Synthetic dataset
# ---------------------------------------------------------------------------
torch.manual_seed(0)

N_TRAIN, N_TEST = 400, 100
N_FEATURES, N_TARGETS = 8, 1

X_train = torch.randn(N_TRAIN, N_FEATURES) * 3 + 5  # deliberately off-centre
y_train = torch.randn(N_TRAIN, N_TARGETS) * 10 - 2
X_test = torch.randn(N_TEST, N_FEATURES) * 3 + 5
y_test = torch.randn(N_TEST, N_TARGETS) * 10 - 2


# ---------------------------------------------------------------------------
# Model definition
# ---------------------------------------------------------------------------
class SimpleModel(nn.Module):
    """Linear regression model with embedded input and target scalers.

    The scalers are stored as child modules, so their fitted statistics are
    automatically included in state_dict() and moved with .to(device).
    """

    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.feature_scaler = ZScoreScaler()
        self.target_scaler = ZScoreScaler()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Calling the scaler directly (scaler(x)) is equivalent to
        # scaler.transform(x) — forward() delegates to transform().
        x = self.feature_scaler(x)
        return self.linear(x)


# ---------------------------------------------------------------------------
# Step 1: fit scalers on training data only (before any training loop)
# ---------------------------------------------------------------------------
model = SimpleModel(N_FEATURES, N_TARGETS)

model.feature_scaler.fit(X_train)
model.target_scaler.fit(y_train)

print("After fitting:")
print(f"  feature_scaler.mean = {model.feature_scaler.mean}")
print(f"  target_scaler.mean  = {model.target_scaler.mean}")

# ---------------------------------------------------------------------------
# Step 2: minimal training loop
# ---------------------------------------------------------------------------
optimiser = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

model.train()
for epoch in range(3):
    optimiser.zero_grad()
    pred = model(X_train)
    # Scale targets into normalised space before computing the loss.
    target = model.target_scaler(y_train)
    loss = loss_fn(pred, target)
    loss.backward()
    optimiser.step()
    print(f"  epoch {epoch + 1}/3  loss={loss.item():.4f}")

# ---------------------------------------------------------------------------
# Step 3: save checkpoint — scaler statistics are included automatically
# ---------------------------------------------------------------------------
with tempfile.TemporaryDirectory() as tmpdir:
    ckpt_path = os.path.join(tmpdir, "checkpoint.pt")
    torch.save(model.state_dict(), ckpt_path)
    print(f"\nCheckpoint saved to {ckpt_path!r}")

    # Step 4: reload into a fresh (unfitted) model
    fresh_model = SimpleModel(N_FEATURES, N_TARGETS)
    state = torch.load(ckpt_path, weights_only=True)
    fresh_model.load_state_dict(state)
    print("Checkpoint reloaded successfully.")

    # Step 5: verify the statistics survived the round-trip
    assert torch.allclose(
        model.feature_scaler.mean, fresh_model.feature_scaler.mean
    ), "feature_scaler.mean mismatch after reload!"
    assert torch.allclose(
        model.target_scaler.mean, fresh_model.target_scaler.mean
    ), "target_scaler.mean mismatch after reload!"
    print("Scaler statistics verified — round-trip OK.")

    # Step 6: inference with inverse transform
    fresh_model.eval()
    with torch.no_grad():
        pred_scaled = fresh_model(X_test)
        # Bring predictions back to the original target scale.
        pred_orig = fresh_model.target_scaler.inverse_transform(pred_scaled)

    print(f"\nFirst 5 predictions (original scale): {pred_orig[:5].squeeze().tolist()}")
