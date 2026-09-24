# Repository Guidelines

## Project Structure & Module Organization

This is a small Python package for configurable NR PUSCH waveform generation and captured-IQ analysis. Production code is under `src/nr_pusch/`: `config.py` parses TOML and builds Sionna settings, `transmitter.py` creates the multi-user waveform, `iq/` reads references, `artifacts.py` writes NPZ/JSON outputs, and `cli/` contains commands. Put run profiles in `configs/`, design notes in `docs/`, and tests in `tests/unit/` or `tests/integration/`. MATLAB reference files belong under `tests/fixtures/matlab_h5/`; do not edit or regenerate supplied fixtures as part of ordinary development.

## Build, Test, and Development Commands

Use the repository virtual environment at `/home/le-lei/workspace/test/.venv`.

- `pip install -e .` installs the package and `nr-pusch-tx` command in editable mode.
- `PYTHONPATH=src python -m unittest discover -s tests -p 'test_*.py' -v` runs unit and integration tests.
- `nr-pusch-tx --config configs/pusch_4ue.toml --output /tmp/pusch.npz --batch-size 1 --seed 13 --device cpu` generates IQ and a JSON manifest outside the repository.

## Coding Style & Naming Conventions

Use Python 3.11+ syntax, four-space indentation, type hints for public APIs, `snake_case` for modules/functions/variables, and `PascalCase` for classes. Keep Sionna adapters explicit about tensor axis order, waveform stage, sample rate, and MCS mapping. Configuration values belong in TOML rather than hard-coded experiment scripts. No formatter or linter is currently configured.

## Testing Guidelines

Tests use the standard-library `unittest` runner and may use NumPy assertions. Name files `test_*.py` and test methods `test_*`. Add focused unit tests for project-owned parsing and export behavior, and integration tests for Sionna composite blocks and MATLAB reference comparisons. For waveform changes, include the compared stage, tolerance, MCS/TB size, and relevant manifest values. Run tests with the command above before submitting.

## Commit & Pull Request Guidelines

Recent history uses short imperative messages, sometimes with prefixes such as `docs:`. Prefer concise subjects like `feat: add DFT-s-OFDM DMRS mapping` or `fix: validate payload size`. Pull requests should summarize behavior, list validation commands/results, identify any 3GPP/Sionna coverage limits, and include reference-comparison metrics when applicable. Keep generated IQ, caches, and large experiment outputs out of commits.

## Configuration and Artifacts

Do not silently change the active profile in `configs/pusch_4ue.toml`; explain parameter changes against the MATLAB fixture or target MCS. Preserve user-provided IQ/H5 files. Store generated NPZ, JSON manifests, and temporary captures under `/tmp` or a dedicated ignored output directory.
