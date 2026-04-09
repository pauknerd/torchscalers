# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-04-10

### Added

- `Scaler` — abstract base class for all scalers (`nn.Module` subclass).
- `MinMaxScaler` — scales features to the range \[0, 1\] using per-feature min/max.
- `ZScoreScaler` — standardises features to zero mean and unit variance.
- `RobustScaler` — scales using median and IQR, robust to outliers.
- `MaxAbsScaler` — scales to \[−1, 1\] by dividing by the maximum absolute value.
- `ShiftScaleScaler` — applies a pre-specified `(x + shift) * scale` transformation.
- `LogScaler` — applies `log(x + eps)` transformation.
- `PerDomainScaler` — applies a separate scaler instance per string domain ID.
- `MixedDomainScaler` — applies a different scaler type per string domain ID.
- All scalers store fitted statistics as `nn.Module` buffers (included in `state_dict`, move with `.to(device)`).
- PyTorch Lightning integration via `fit_scalers` helper.
