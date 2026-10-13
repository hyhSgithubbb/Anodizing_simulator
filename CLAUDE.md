# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

Al/Ti anodizing simulator: predicts Al₂O₃ / TiO₂ oxide thickness over time from voltage, current, electrolyte recipe, cathode material and electrode layout. Python + PyQt6 + PyQtGraph (+ NumPy, SciPy, PyOpenGL). UI text, docstrings and comments are in Korean — keep that convention. Pure metals only (Al 1xxx, CP Ti); alloys are out of scope by decision.

`docs/DESIGN.md` is the source of truth for the math model, verification results and roadmap (P0–P1 done; P2 = per-position thickness h(s,t) via L1 resistance matrix). Update it when the model or architecture changes.

## Commands

```bash
pip install -r requirements-dev.txt && pip install -e .   # src/ layout — editable install required
python -m anodsim                                          # GUI
python -m anodsim.cli --list                               # list presets
python -m anodsim.cli --metal Al --mode CCCV --voltage 15 --current 5 --minutes 60 [--csv out.csv]
pytest                                                     # all tests (GUI tests use Qt offscreen if no DISPLAY)
pytest tests/test_simulator.py::test_720_rule              # single test
ruff check src tests --select F,E9,B,UP --target-version py311 --line-length 120
```

Headless GUI screenshots/checks: run under `xvfb-run -a -s "-screen 0 1600x1000x24" ...` (OpenGL 3D view needs a real X server; offscreen works for tests).

## Architecture

- `src/anodsim/core/` — physics engine, **must not import Qt**. All internal values are SI; convert only at the GUI/CLI/JSON boundary (`core/units.py`, `core/presets.py`).
- `src/anodsim/data/*.json` — metal / electrolyte / cathode presets in human units (nm/V, μm/h, °C, A/dm²); `core/presets.py` (`PresetDB`) converts to SI and validates. Adding a recipe = editing JSON, not code. All presets are `calibrated: false` (initial estimates; measured data pending → P5).
- `src/anodsim/gui/` — PyQt6 front end; `cli.py` is the headless equivalent.

Simulation pipeline (spans several files):
1. `ProcessConfig` (frozen, `core/state.py`) → `Simulator` (`core/simulator.py`) resolves presets into `ElectrolyteProps` at bath temperature/concentration and builds a `GrowthModel` (`core/growth/`: `CompactGrowth` or `PorousGrowth` by preset `regime`).
2. Everything is written as length-N arrays over anode surface elements with an N×N electrolyte resistance matrix `R`. Currently L0: N=1, `R` from `Geometry.lumped_resistance` (side-view geometry: x horizontal, z height, y depth). L1/L2 will only change how `R` and areas are built.
3. `CircuitSolver` (`core/circuit.py`) finds the operating point each step: Newton in x = ln j over the voltage balance U_app = U_off + U_f(j,δ_b) + R·I + R_ext·I + |η_c|. The rectifier runs at whichever of voltage/current limit binds first (CV/CC/CCCV differ only in which value is the setpoint and where the ramp applies). `barrier_sensitivity` gives dj/dδ_b and dU_f/dδ_b by implicit differentiation.
4. Time integration: the barrier ODE is very stiff (high-field law, relaxation ~0.1 s), so δ_b uses a **linearly implicit Euler** step with Jacobian from `barrier_sensitivity` × `GrowthModel.barrier_rate_partials`. Charge, porous layer and metal loss are advanced with the same linearized current `j_lin`, which makes charge conservation exact — keep this invariant if you touch `Simulator.step` (tests check it to 1e-9). Explicit schemes oscillated at steady state; don't revert to them.
5. Porous model charge partition: only current above the steady-state j_ss(U_f) thickens the barrier; the rest grows pores (`GrowthModel.pore_formation`). The porous layer has no electrical resistance, so it never feeds back into the circuit.
6. `core/limits.py` raises flags (breakdown, burning, dissolution limit, CC operation) and config range / cathode-compatibility warnings.

GUI: `MainWindow` wires input panels (`gui/panels/`) → builds/validates `ProcessConfig` on every change → updates `SideView` (default 2D) and `Geometry3DView` (pyqtgraph.opengl; left-drag orbit, right-drag pan; labels are QLabels positioned by projecting 3D points because GL text doesn't render reliably). Runs execute in `SimulationWorker` on a `QThread`; results go to `ResultsView` (4 stacked plots sharing a time axis, single y-axis each).

## Testing notes

- Physics regressions to preserve: Faraday coefficients (Al₂O₃ 0.550, TiO₂ 0.545 μm per C/cm²), 720-rule reproduction, charge conservation, CC current constraint, Ti ≈2.5 nm/V, no steady-state oscillation, time-step convergence.
- `tests/conftest.py` provides `make_config(...)` and `ideal_presets(...)` (η=1, P=0, no dissolution) for conservation-style tests.
