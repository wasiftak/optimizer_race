# STATE

Current phase: 3 (Full landscape terrains, hazards, commentary) COMPLETE, tag v0.3
Last tag: v0.3
Environment: Windows, PowerShell, venv at ucs668\venv, Python 3.10.11, RTX 3050 4GB, torch 2.6.0+cu124
Works: Landscape Track complete: 7 terrains, 2-4 racers via sidebar Garage (7 optimizers, scheduler per racer, per-optimizer default lr), animated 2D race with danger-zone rectangle, 3D surface view, wind and danger-zone hazards with shared seed, rule-based commentary, Explain panel per racer, results ranking
Tests: 53 pytest tests, 14 smoke checks
Files and contents:
app.py - entry, calls ui.landscape_tab.render(); Real Model tab is still a placeholder
core/config.py - paths, DEVICE, DEFAULT_SEED, RACER_COLORS, OPTIMIZER_PARAMS, LR_CHOICES (16 values)
core/schemas.py - RacerConfig, RaceConfig, RacerLog
core/terrains.py - Terrain dataclass (with adaptive_lr), 7 terrains, TERRAINS dict
core/optimizers.py - OPTIMIZER_NAMES, build_optimizer (all 7), build_scheduler
core/landscape_engine.py - run_landscape_race(racers, terrain, start, max_steps, tol, bound, hazards, seed), rank_landscape
core/commentary.py - generate_commentary(logs, terrain), racer_stats, explain_racer; tunable constants at top
core/viz.py - contour_base(terrain, zone), paths_figure (unused), animated_race_figure(..., zone), surface_figure
ui/garage.py - render_garage(mode, default_lr, adaptive_lr) -> list[RacerConfig]; lr_for_optimizer; SCHEDULERS
ui/landscape_tab.py - render(): terrain, start, hazards expander, steps, speed, 2D/3D view, Run, figure, commentary box, results table, Explain expanders
scripts/smoke_test.py - 14 checks (run: python scripts/smoke_test.py)
tests/ - test_terrains, test_optimizers, test_phase2, test_phase3, test_commentary, test_viz
Tuned defaults (start, SGD-family lr, adaptive lr): bowl (-2.5,2.0) 0.1/0.1; canyon (-2.5,1.2) 0.08/0.08; banana (-1.5,2.0) 0.002/0.1; beale (0,0) 0.005/0.1; himmelblau (-0.3,0.3) 0.01/0.05; rastrigin (3.3,3.3) 0.003/0.1; saddle (1.0,0.01) 0.01/0.02
Known issues: Garage writes to st.sidebar from the landscape tab, so Phase 4 must keep the two tabs' sidebars from clashing (use mode keys or a sidebar container per tab); 3D view does not draw the danger zone; Rastrigin traps most racers by design
Decisions: seed 42, smoke test uses sys.path.insert(0, '.'), run from optimizer_race, width='stretch' in Streamlit calls, full files always delivered, hazards use two generators drawn every step so all racers get identical noise
Next task: Phase 4 (Real Model Track: data.py, models.py, race_engine.py lockstep generator, replay storage, lanes and curve figures, ui/model_tab.py)