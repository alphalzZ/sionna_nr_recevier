# Repository Guidelines

## Project Structure & Module Organization

This is a small Python package for configurable NR PUSCH waveform generation and captured-IQ analysis. Production code is under `src/nr_pusch/`: `config.py` parses TOML and builds Sionna settings, `transmitter.py` creates the multi-user waveform, `iq/` reads references, `artifacts.py` writes NPZ/JSON outputs, and `cli/` contains commands. Put run profiles in `configs/`, design notes in `docs/`, and tests in `tests/unit/` or `tests/integration/`. MATLAB reference files belong under `tests/fixtures/matlab_h5/`; do not edit or regenerate supplied fixtures as part of ordinary development.

## Build, Test, and Development Commands

Use the repository virtual environment at `/home/le-lei/workspace/test/.venv`.

- `pip install -e .` installs the package and `nr-pusch-tx` command in editable mode.
- `PYTHONPATH=src python -m unittest discover -s tests -p 'test_*.py' -v` runs unit and integration tests.
- `nr-pusch-tx --config configs/pusch_4ue.toml --output /tmp/pusch.npz --batch-size 1 --seed 13 --device cpu` generates IQ and a JSON manifest outside the repository.
- `nr-pusch-channel --config configs/cdl_38_901_4x4.toml --input /tmp/pusch.npz --sample-rate-hz 18000000 --output /tmp/pusch_rx.npz` applies the configured CDL channel.
- `nr-pusch-bler --tx-config configs/pusch_4ue.toml --channel-config configs/cdl_38_901_4x4.toml --simulation-config configs/bler_4ue_cdl.toml --output /tmp/pusch_bler.csv --device cpu` runs an SNR versus BLER sweep.
- `nr-pusch-rx --tx-config configs/pusch_4ue.toml --input /tmp/pusch_rx.npz --noise-variance 0.001 --output /tmp/pusch_decode.npz` decodes receive IQ and reports TB CRC status.

## Coding Style & Naming Conventions

Use Python 3.11+ syntax, four-space indentation, type hints for public APIs, `snake_case` for modules/functions/variables, and `PascalCase` for classes. Keep Sionna adapters explicit about tensor axis order, waveform stage, sample rate, and MCS mapping. Configuration values belong in TOML rather than hard-coded experiment scripts. No formatter or linter is currently configured.

## Testing Guidelines

Tests use the standard-library `unittest` runner and may use NumPy assertions. Name files `test_*.py` and test methods `test_*`. Add focused unit tests for project-owned parsing and export behavior, and integration tests for Sionna channel blocks and MATLAB reference comparisons. For waveform changes, include the compared stage, tolerance, MCS/TB size, and relevant manifest values. Run tests with the command above before submitting.

## Documentation Sync

Documentation is release evidence here, not a follow-up task. Every behavior change must land with the matching documentation update in the same change.

- Changing a CLI flag, default, output artifact, schema key, or config field updates `README.md` (command examples and the affected section) and `docs/feature_catalog.md` (entry-point table and the implementation-evidence row).
- Changing simulation or validation semantics, acceptance gates, or the shared DMRS prior store updates `docs/dmrs_tap_power_prior.md` (§4-§7 rules, §9 measured status) plus the README prior section.
- Changing receiver algorithms, capture tuning, or measured results updates `docs/receiver_optimization.md`; changing transmitter, DMRS, precoding, or MCS/TBS mapping updates `docs/transmitter.md`; changing module layout, entry points, or run order updates `docs/architecture.md` and `docs/release_0_1_simulation_plan.md`.
- Adding, renaming, or materially changing a `configs/*.toml` profile, or any `configs/scenarios.toml` bundle, updates the profile and scenario tables in `README.md` and `docs/feature_catalog.md`, plus any command example in this file.
- Citations written as `path/to/file.py:123-145` must point at the code they describe. Re-check every cited range in each file you touch and refresh shifted line numbers; prefer citing the symbol's current range over a remembered one.
- Values quoted in documentation (SNR lists, batch sizes, frame budgets, detector and estimator lists, device, BWP/RB, layers and ports, channel model, speed, delay spread, antenna rows/cols) must be read back from the TOML or source, not from memory. Preset scenarios and smoke profiles are bounded diagnostics, never validated BLER benchmarks.
- Measured tables stay historical records: keep the date, configuration names, stage and tolerances. Do not silently rewrite numbers; when a conclusion changes, state the new measurement and what it supersedes.

## Commit & Pull Request Guidelines

Recent history uses short imperative messages, sometimes with prefixes such as `docs:`. Prefer concise subjects like `feat: add DFT-s-OFDM DMRS mapping` or `fix: validate payload size`. Pull requests should summarize behavior, list validation commands/results, identify any 3GPP/Sionna coverage limits, and include reference-comparison metrics when applicable. Keep generated IQ, caches, and large experiment outputs out of commits.

## Configuration and Artifacts

Do not silently change the active profile in `configs/pusch_4ue.toml`; explain parameter changes against the MATLAB fixture or target MCS. Preserve user-provided IQ/H5 files. Store generated NPZ, JSON manifests, and temporary captures under `/tmp` or a dedicated ignored output directory.
