# Changelog

All notable changes to `Glacier.Clavier` are documented here.

## v1.2.1

- Added 4-way FMA unrolled `SimdKernels.Dot` kernels for decision/verification/score forward passes.
- Consolidated batch inference under single reader-lock acquisitions (`DecideBatch`, `VerifyBatch`, `ScoreBatch`).
- Enforced atomic conformal quantile updates under write-lock safety.

## v1.1.0

- Added online supervised training (`TrainDecision`, `TrainVerify`, `TrainScore`) across all three heads.
- Added split conformal gating (`CalibrateConformal`, `DecideConformalSet`, `ClavierPredictionSetSummary`).
- Added `.clvr` binary checkpoint save/load persistence on `ClavierSession`.
