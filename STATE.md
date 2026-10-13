# STATE

Current phase: 5 (Model dropdown with four presets) COMPLETE, tag v0.5
Last tag: v0.5
Environment: Windows, PowerShell, venv at ucs668\venv, Python 3.10.11, RTX 3050 4GB, torch 2.6.0+cu124
Works: Landscape Track (7 terrains, hazards, commentary, 3D, Explain) and Real Model Track (live lockstep training on FashionMNIST subset). Each racer can pick one of 5 models (default Small CNN, Mini-VGG, Tiny ResNet, Wide and shallow, Deep and narrow) or all share Racer 1's model (sidebar checkbox). Fairness badge above the lanes, per-racer layer summaries, parameter bar chart, Model column in the results table, saved replays with slider and Play.
Tests: 133 pytest tests (incl. AppTest UI tests), 15 smoke checks
Files and contents:
app.py - radio track switch, calls landscape_tab.render() or model_tab.render()
core/config.py, schemas.py, terrains.py, optimizers.py, landscape_engine.py, commentary.py - as in v0.3 (RacerConfig already has model and arch fields)
core/data.py - get_data(subset_size, test_size, seed); FashionMNIST only
core/registry_models.py - MODEL_PRESETS (label, params, lesson), MODEL_ORDER
core/models.py - SmallCNN, MiniVGG, ResBlock, TinyResNet, WideShallow, DeepNarrow, MODEL_CLASSES, build_model (KeyError for unknown names), count_params, same_model(racers), summarize(model, in_shape)
core/race_engine.py - run_model_race(racers, race_cfg, data, model_factory, device) generator (same model: deepcopy of one base; mixed models: seed reset before each build), run_to_end, rank_model, best_accuracy, evaluate
core/storage.py - save_run, load_run, list_runs, replay_logs, total_steps, run_hash
core/tables.py - badge_text, model_results_rows (with Model column)
core/viz.py - landscape figures + lanes_figure, loss_figure, acc_figure, params_figure, moving_average, latest_accuracy
ui/garage.py - render_garage(mode, default_lr, adaptive_lr); model dropdown and same-model checkbox only in mode 'model'
ui/landscape_tab.py, ui/model_tab.py (constants REDRAW_EVERY 5, EVAL_EVERY 10, SUBSET 3000, SGD_LR 0.03, ADAPTIVE_LR 0.003, DEFAULT_TARGET 0.85, IN_SHAPE (1,28,28), N_CLASSES 10)
scripts/smoke_test.py - 15 checks
tests/ - test_terrains, test_optimizers, test_phase2, test_phase3, test_commentary, test_viz, test_models (28), test_models_race (4), test_data, test_race_engine, test_storage, test_model_viz, test_tables, test_ui, test_phase5_ui
Parameter counts (1x28x28, 10 classes): default 20,490; mini_vgg 32,058; tiny_resnet 9,770; wide_shallow 63,562; deep_narrow 5,762
Calibration: default race (400 steps) Adam 0.003 best 88.1%, Momentum 0.03 88.2%, SGD 0.03 84.8%; with target 0.85 Adam finishes at 120, Momentum at 160, SGD times out (best 83.2% at 300 steps)
Known issues: Garage settings reset when switching tracks (fix in Phase 10); 3D view does not draw the danger zone; Rastrigin traps most racers by design; timeouts are ranked by best accuracy while lanes show latest accuracy; cache/run_*.json is git-ignored (demo replays in Phase 10 will be named demo_*.json); Streamlit AppTest selectbox API may differ by version (test_phase5_ui has a fallback)
Decisions: seed 42, smoke test uses sys.path.insert(0, '.'), run from optimizer_race, width='stretch' in Streamlit calls, full files always delivered, same_model() helper instead of a returned flag, torch is not installable in Claude's sandbox so all PyTorch code is first run on the user's machine
Next task: Phase 6 (dataset dropdown: FashionMNIST, MNIST, CIFAR-10, optional flowers; registry_datasets.py; generalise get_data; target and chance reset per dataset; sample preview)