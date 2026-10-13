# STATE

Current phase: 2 (Animated race and Garage), finishing
Last tag: v0.1
Environment: Windows, PowerShell, venv at ucs668\venv, Python 3.10.11, RTX 3050 4GB, torch 2.6.0+cu124
Works: 2-4 racers via sidebar Garage (7 optimizers, only relevant controls shown), start presets and custom start, animated Plotly race with Play/Pause/slider, rank_landscape, results table
Tests: 23 pytest tests, 7 smoke checks
Files and contents:
app.py - entry, calls ui.landscape_tab.render()
core/config.py - paths, DEVICE, DEFAULT_SEED, RACER_COLORS, OPTIMIZER_PARAMS, LR_CHOICES
core/schemas.py - RacerConfig, RaceConfig, RacerLog
core/terrains.py - bowl/canyon/banana, TERRAINS dict (4 more terrains come in Phase 3)
core/optimizers.py - OPTIMIZER_NAMES, build_optimizer (all 7), build_scheduler (already done)
core/landscape_engine.py - run_landscape_race, rank_landscape
core/viz.py - contour_base, paths_figure (unused now), animated_race_figure
ui/garage.py - render_garage(mode, default_lr) -> list[RacerConfig], writes to st.sidebar
ui/landscape_tab.py - render(): terrain, start, steps, speed, Run, animated figure, ranking table
scripts/smoke_test.py - 7 checks
tests/test_terrains.py, tests/test_optimizers.py, tests/test_phase2.py
Known issues: banana default lr to be tuned; Garage writes to the sidebar from the landscape tab, so Phase 4 must keep the two tabs' sidebars from clashing (use mode keys or a sidebar container per tab)
Decisions: seed 42, smoke test uses sys.path.insert(0, '.'), run from optimizer_race, width='stretch' in Streamlit calls, LR_CHOICES extended with 0.002/0.05/0.08/0.11/0.2, Garage gives every racer the terrain's nearest default lr, full files always delivered
Next task: Phase 3 (4 more terrains, schedulers in the engine and Garage, wind and danger-zone hazards, commentary.py, 3D view, Explain panel)