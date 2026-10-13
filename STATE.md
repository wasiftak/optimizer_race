# STATE

Current phase: 1 (Landscape Track static), finishing
Last tag: v0.0
Environment: Windows, PowerShell, venv at ucs668\venv, Python 3.10.11, RTX 3050 4GB, torch 2.6.0+cu124
Works: schemas (all 3 dataclasses), 3 terrains, all 7 optimizers + build_scheduler, landscape engine, contour and paths figures, landscape tab with 3 fixed racers (SGD, Momentum, Adam) sharing one lr
Tests: 11 pytest tests, 7 smoke checks
Files and contents:
app.py - entry, calls ui.landscape_tab.render()
core/config.py - paths, DEVICE, DEFAULT_SEED, RACER_COLORS
core/schemas.py - RacerConfig, RaceConfig, RacerLog
core/terrains.py - Terrain dataclass, bowl/canyon/banana, TERRAINS dict
core/optimizers.py - OPTIMIZER_NAMES, build_optimizer, build_scheduler
core/landscape_engine.py - run_landscape_race
core/viz.py - contour_base, paths_figure
ui/landscape_tab.py - render() with terrain select, shared lr, max steps, Run, figure, table
scripts/smoke_test.py - 7 checks
tests/test_terrains.py, tests/test_optimizers.py
Known issues: banana default lr to be tuned after visual check
Decisions: seed 42, smoke test uses sys.path.insert(0, '.'), run from optimizer_race, width='stretch' in Streamlit calls
Next task: Phase 2 (animated race, Garage with 2-4 racers, rank_landscape, OPTIMIZER_PARAMS in config)