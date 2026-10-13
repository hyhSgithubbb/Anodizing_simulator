# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

Al/Ti anodizing simulator: predicts Al₂O₃ / TiO₂ oxide thickness over time from voltage, current, electrolyte recipe, cathode material and electrode layout. Python + PyQt6 + PyQtGraph (+ NumPy, SciPy, PyOpenGL). UI text, docstrings and comments are in Korean — keep that convention. Pure metals only (Al 1xxx, CP Ti); alloys are out of scope by decision.

`docs/DESIGN.md` is the source of truth for the math model, verification results and roadmap (P0–P4 done; P5 = calibration against measured thickness–time data once the user supplies it, plus per-project preset overrides). Update it when the model or architecture changes.

## Commands

```bash
pip install -r requirements-dev.txt && pip install -e .   # src/ layout — editable install required
python -m anodsim                                          # GUI
python -m anodsim.cli --list                               # list presets
python -m anodsim.cli --metal Al --mode CCCV --voltage 15 --current 5 --minutes 60 [--csv out.csv] [--profile-csv p.csv] [--fidelity L0|L1|L2] [--cooling W_PER_K --coolant C] [--save-project p.anod.json]
python -m anodsim.cli --project p.anod.json                # run a saved project (the only CLI route to multi-step/pulse programs)
pytest                                                     # all tests (GUI tests use Qt offscreen if no DISPLAY)
pytest tests/test_simulator.py::test_720_rule              # single test
ruff check src tests --select F,E9,B,UP --target-version py311 --line-length 120
```

Headless GUI screenshots/checks: run under `xvfb-run -a -s "-screen 0 1600x1000x24" ...` (OpenGL 3D view needs a real X server; offscreen works for tests).

## Architecture

- `src/anodsim/core/` — physics engine, **must not import Qt**. All internal values are SI; convert only at the GUI/CLI/JSON boundary (`core/units.py`, `core/presets.py`).
- `src/anodsim/data/*.json` — metal / electrolyte / cathode presets in human units (nm/V, μm/h, °C, A/dm²); `core/presets.py` (`PresetDB`) converts to SI and validates. Adding a recipe = editing JSON, not code. All presets are `calibrated: false` (initial estimates; measured data pending → P5).
- `src/anodsim/io/` — Qt-free file I/O: `project.py` (`*.anod.json` v1 ↔ `ProcessConfig`, human units in key names, `null` = no current limit, Korean `ProjectError` messages) and `export.py` (time-series / profile CSV). GUI File menu and CLI share it.
- `src/anodsim/gui/` — PyQt6 front end; `cli.py` is the headless equivalent.

Simulation pipeline (spans several files):
1. `ProcessConfig` (frozen, `core/state.py`) → `Simulator` (`core/simulator.py`) resolves presets into `ElectrolyteProps` at bath temperature/concentration and builds a `GrowthModel` (`core/growth/`: `CompactGrowth` or `PorousGrowth` by preset `regime`).
2. Everything is written as length-N arrays over anode surface elements with an N×N electrolyte resistance matrix `R`. `core/field/build_field(geometry, fidelity)` is the only place that differs per fidelity: L0 → one element, `R` from `Geometry.lumped_resistance`; L1 → `SurfaceMesh` along the side-view (x–z) perimeter of the immersed anode (−x face ↓, bottom, +x face ↑, top if fully immersed; y-end faces folded into ±x elements so Σa = wetted area) and diagonal `R_ii = L_eff/(κ a_i)` from shortest paths that wrap around the plate and never cross the liquid surface (`path_model.py`); L2 → same mesh, dense symmetric `R = BᵀK⁻¹B/(κD)` from a vertex-centred finite-volume Laplace solve on a graded rectilinear grid of the side-view cross-section (`fdm.py`: cathodes φ=0, walls/floor/surface insulating, unit current per anode element, `splu` once + N back-solves; D = Σa/Σℓ equivalent depth). Grid lines are generated from both ends of each key interval so mirror-symmetric layouts give exactly symmetric grids/solutions — keep that. `Field` stores `unit_resistance` (κ = 1) and, for L2, the `FDMSolution` whose `basis` (K⁻¹B) turns element currents into the electrolyte potential field for display. `build_field` is LRU-cached on (geometry, fidelity, elements, grid_scale). Geometry coords: x horizontal, z height, y depth. Validation requires ≥1 mm anode clearance from walls/floor and from every cathode in the side-view projection (otherwise L2 elements lose electrolyte contact).
3. `CircuitSolver` (`core/circuit.py`) finds the operating point each step: Newton in x = ln j over the voltage balance U_app = U_off + U_f(j,δ_b) + R·I + R_ext·I + |η_c|. The rectifier runs at whichever of voltage/current limit binds first (CV/CC/CCCV differ only in which value is the setpoint and where the ramp applies). `barrier_sensitivity` gives dj/dδ_b and dU_f/dδ_b by implicit differentiation.
4. Time integration: the barrier ODE is very stiff (high-field law, relaxation ~0.1 s), so δ_b uses a **linearly implicit Euler** step with Jacobian from `barrier_sensitivity` × `GrowthModel.barrier_rate_partials`. Charge, porous layer and metal loss are advanced with the same linearized current `j_lin`, which makes charge conservation exact — keep this invariant if you touch `Simulator.step` (tests check it to 1e-9). Explicit schemes oscillated at steady state; don't revert to them.
5. Porous model charge partition: only current above the steady-state j_ss(U_f) thickens the barrier; the rest grows pores (`GrowthModel.pore_formation`). The porous layer has no electrical resistance, so it never feeds back into the circuit.
6. `core/limits.py` raises flags (breakdown, burning, dissolution limit, CC operation, bath temperature out of preset range) and config range / cathode-compatibility warnings (voltage check uses `power.max_voltage`).
7. `SimulationResult`: scalar series are area-weighted means (+ `total_min/max`), plus bath `temperature` (K), program `step` index and a `thermal` flag; per-element history is subsampled (~400 times) into `profile_time/profile_total/profile_current_density`; plus `mesh`, `wagner`, `uniformity`.
8. Power: `ProcessConfig.power` is a `PowerSetting` or a `PowerProgram` (steps of mode/V/I/duration/ramp + repeat). `PowerSupply` turns both into a step list; at a step change both limits ramp linearly from the previous step's limits (an infinite limit switches immediately); the first step soft-starts from 0. `Simulator._step_size` snaps step ends to `supply.next_breakpoint()` (step starts, ramp ends) — a boundary instant belongs to the new step. `duration` may truncate a program but not exceed it.
9. Thermal (optional `ProcessConfig.thermal`): lumped bath `C dT/dt = P − UA(T − T_cool)` with `P = (U − R_ext·I)·I` (cable heat stays outside the bath) (`core/thermal.py`, C from preset `density`/`heat_capacity` × `Geometry.liquid_volume()`), advanced per step by the exact solution for constant P (`BathThermal.advance`; explicit Euler oscillated once C/UA < 2Δt), ≤0.25 K/step. When T drifts >0.05 K, `Simulator._update_properties` rebuilds `props`, the growth model and `R = field.resistance(κ(T))` (also pushed into `circuit.resistance`) — so never cache `props`/`model`/`resistance` across steps. Adiabatic runs satisfy `C·ΔT = Σ P Δt` exactly (tested).
10. `core/optics.py`: Ti interference colour from thin-film reflectance (air/TiO₂ Cauchy n/Ti complex n, normal incidence) integrated with CIE 1931 CMF (analytic fit) under D65 → sRGB. Used for Ti compact films only (CLI line, GUI preview, readout swatch, side/3D colouring).

`anodsim/__init__.py` forces single-threaded BLAS (setdefault) before numpy loads — multithreaded OpenBLAS on these small systems burned all cores and made GUI runs 3× slower. Don't remove it.

GUI: `MainWindow` wires input panels (`gui/panels/`) → builds/validates `ProcessConfig` on every change → updates `SideView` (default 2D; electrodes are draggable `PlateHandle` ROIs — dropping emits `electrode_moved` (deferred with a 0 ms timer, snapped to 1 mm) which `GeometryPanel.set_electrode_position` writes into the spin boxes, so the normal change path redraws; L2 results add an electrolyte-potential `ImageItem` with its own orange ramp) and `Geometry3DView` (pyqtgraph.opengl; left-drag orbit, right-drag pan; labels are QLabels positioned by projecting 3D points because GL text doesn't render reliably). Runs execute in `SimulationWorker` on a `QThread` with a speed multiplier. Live updates are **pull-based**: the worker only swaps a reference to the latest immutable `Snapshot`; a `QTimer` in `MainWindow` calls `take_snapshot()` (33 ms, 100 ms at max speed). Don't go back to emitting a signal per snapshot — slow painting flooded the event queue and starved timers/input for seconds. `ResultsView` has tabs [시간 이력 | 위치별 분포] (profiles h(s), j(s) + heatmap); `SideView` draws per-face `ImageItem` colour strips and `Geometry3DView` a per-element coloured mesh (`gui/views/colormap.py`: viridis for thickness, CET-D1 for deviation). Painting is skipped when the result's geometry no longer matches the current inputs.

`MainWindow` also has a File menu (new/open/save/save-as, CSV/PNG export; dialogs go through `_ask_open_path`/`_ask_save_path` so tests monkeypatch them) and `apply_config()`, which loads a config into the panels in the order metal → electrolyte → cathode → process → geometry (choosing a preset resets temperature/voltage/cathode, so later setters must override); a failed `open_project` re-applies the previous config so the panels never stay half-loaded. Values without a GUI field (elements, offset voltage, external resistance, y_center) are carried through unchanged. `ProcessPanel` has the program editor (`panels/program_editor.py`, PROGRAM mode; the duration box then follows the program length) and the checkable thermal group. `SideViewPanel`'s 색칠 combo switches between the thickness colormap and Ti interference RGB (enabled only for Ti compact results); `ColorLegend.set_range(..., colors=...)` shows the matching swatch bar.

## Testing notes

- Physics regressions to preserve: Faraday coefficients (Al₂O₃ 0.550, TiO₂ 0.545 μm per C/cm²), 720-rule reproduction, charge conservation, CC current constraint, Ti ≈2.5 nm/V, no steady-state oscillation, time-step convergence.
- `tests/conftest.py` provides `make_config(...)` (fidelity defaults to L0) and `ideal_presets(...)` (η=1, P=0, no dissolution) for conservation-style tests, plus the shared GUI `window` fixture.
- Outside `app.exec()` (screenshot scripts, tests) `deleteLater` never runs — hide widgets before deleting them (see `OrbitView.set_labels`).
