# Repository Guidelines

## Project Structure & Module Organization

This is a small Python package for configurable NR PUSCH waveform generation and captured-IQ analysis. Production code is under `src/nr_pusch/`: `config.py` parses TOML and builds Sionna settings, `transmitter.py` creates the multi-user waveform, `iq/` reads references, `artifacts.py` writes NPZ/JSON outputs, and `cli/` contains commands. Put run profiles in `configs/`, design notes in `docs/`, and tests in `tests/unit/` or `tests/integration/`. MATLAB reference files belong under `tests/fixtures/matlab_h5/`; do not edit or regenerate supplied fixtures as part of ordinary development.
`rt_config.py` and `rt_scene_assets.py` validate RT profiles and scene packages; `web.py` owns the local FIFO API and job lifecycle.

## Build, Test, and Development Commands

Use the repository virtual environment at `/home/le-lei/workspace/test/.venv`.

- `pip install -e .` installs the package and `nr-pusch-tx` command in editable mode.
- `PYTHONPATH=src python -m unittest discover -s tests -p 'test_*.py' -v` runs unit and integration tests.
- `nr-pusch-tx --config configs/pusch_4ue.toml --output /tmp/pusch.npz --batch-size 1 --seed 13 --device cpu` generates IQ and a JSON manifest outside the repository.
- `nr-pusch-channel --config configs/cdl_38_901_4x4.toml --input /tmp/pusch.npz --sample-rate-hz 18000000 --output /tmp/pusch_rx.npz` applies the configured CDL channel.
- `nr-pusch-bler --tx-config configs/pusch_4ue.toml --channel-config configs/cdl_38_901_4x4.toml --simulation-config configs/bler_4ue_cdl.toml --output /tmp/pusch_bler.csv --device cpu` runs an SNR versus BLER sweep.
- `nr-pusch-rx --tx-config configs/pusch_4ue.toml --input /tmp/pusch_rx.npz --noise-variance 0.001 --output /tmp/pusch_decode.npz` decodes receive IQ and reports TB CRC status.
- `nr-pusch-web --host 127.0.0.1 --port 8765 --config-dir configs --runs-dir /tmp/nr-rt-web-validation/runs` starts the local CDL/RT dashboard; it is not authenticated for non-loopback exposure.
- `nr-pusch-rt-channel --tx-config configs/pusch_4ue.toml --rt-config configs/rt_beam_los.toml --output /tmp/rt-channel.npz` prepares a format-2 snapshot; custom/parameterized snapshot replay uses the matching `--scene-root`.
`nr-pusch-beam-bler --tx-config configs/pusch_4ue.toml --channel-snapshot /tmp/rt-channel.npz --simulation-config configs/bler_rt_beam_web_quick.toml --output /tmp/rt-bler.csv --device cpu` runs the fixed-snapshot profile as configured (the checked-in quick-named profile currently allows up to 2,000 frames/SNR); inspect the profile before launching, and do not treat its name as a statistical benchmark.

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

## Blender MCP (Oh My Pi)

This repository uses the project-scoped `dcc-mcp-blender` connection; do not add it to the global `~/.omp/agent/mcp.json` and do not install a Codex plugin.

- Blender: 4.5.14 LTS at `/home/le-lei/.local/opt/blender-4.5.14-linux-x64/blender`.
- Adapter runtime: `dcc-mcp-blender` 0.2.14, `dcc-mcp-core` 0.20.41, and `dcc-mcp-server` 0.20.41 in Blender's bundled Python 3.11.15. The repository venv is not used for the adapter.
- OMP project config: `.omp/mcp.json`. It registers `blender` as an HTTP MCP server at `http://127.0.0.1:9765/mcp`. Keep this entry project-local; `~/.omp/agent/mcp.json` intentionally has no Blender server.
- The local MCP gateway is loopback-only. Do not expose port 9765 outside the host.
- The adapter's Python packages remain in Blender's embedded runtime, but the global Blender startup hook was removed. The server does not auto-start in unrelated Blender sessions.

### Start the project-scoped Blender host

From the repository root, run the official headless bootstrap and leave it running while using Blender MCP:

```sh
mkdir -p /tmp/dcc-mcp-empty-user-scripts
BLENDER_USER_SCRIPTS=/tmp/dcc-mcp-empty-user-scripts \
  /home/le-lei/.local/opt/blender-4.5.14-linux-x64/blender \
  --background \
  --python /home/le-lei/.local/opt/blender-4.5.14-linux-x64/4.5/python/lib/python3.11/site-packages/dcc_mcp_blender/blender_bootstrap.py
```

The empty user-scripts override prevents unrelated global startup scripts from enabling the adapter. The bootstrap runs Blender headlessly and blocks while serving MCP requests; stop it with Ctrl+C. OMP's project config only applies when OMP is started with this repository as its working directory.

### Use and verify through OMP

Start OMP from this repository root and request typed Blender operations in the conversation, for example:

```text
Use the Blender MCP server to list the current scene objects and return their names.
```

Project-scoped smoke command:

```sh
omp --cwd /home/le-lei/workspace/test/sionna_nr_recevier \
  --no-session --max-time 120 -p \
  'Use the Blender MCP server configured for this project, not shell or Python. Call the scene object-list tool and report the names and count.'
```

Verified on Blender 4.5.14: OMP called Blender's scene object-list tool and returned `Camera`, `Cube`, `Light` (3 objects). Re-run this check after changing `.omp/mcp.json` or upgrading Blender/adapter versions. The `dcc-mcp-blender verify` lifecycle command requires its global install receipt; this project intentionally removes that receipt to avoid global auto-start, so verify the live OMP connection instead.

Prefer typed scene/object tools. The adapter also exposes arbitrary Python execution; treat it as a high-risk escape hatch and use it only when no typed tool can perform the requested operation.
