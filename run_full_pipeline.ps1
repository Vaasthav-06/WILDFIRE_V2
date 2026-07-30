$ErrorActionPreference = "Stop"
$env:PYTHONPATH = "C:\dev\wildfire_v2"

# Write-Host "Starting Step 1: build_training_data.py"
# python -m scripts.build_training_data
# if ($LASTEXITCODE -ne 0) { throw "build_training_data.py failed" }

# Write-Host "Starting Step 2: extract_real_features.py"
# python -m scripts.extract_real_features
# if ($LASTEXITCODE -ne 0) { throw "extract_real_features.py failed" }

Write-Host "Starting Step 3: train_models.py"
python -m scripts.train_models
if ($LASTEXITCODE -ne 0) { throw "train_models.py failed" }

Write-Host "Starting Step 4: run_spatial_blocked_cv.py"
python -m scripts.run_spatial_blocked_cv
if ($LASTEXITCODE -ne 0) { throw "run_spatial_blocked_cv.py failed" }

Write-Host "Starting Step 5: analyze_extreme_vs_mild_days.py"
python -m scripts.analyze_extreme_vs_mild_days
if ($LASTEXITCODE -ne 0) { throw "analyze_extreme_vs_mild_days.py failed" }

Write-Host "Starting Step 6: evaluate_all_models_extended.py"
python -m scripts.evaluate_all_models_extended
if ($LASTEXITCODE -ne 0) { throw "evaluate_all_models_extended.py failed" }

Write-Host "Starting Step 7: optimize_decision_threshold.py"
python -m scripts.optimize_decision_threshold
if ($LASTEXITCODE -ne 0) { throw "optimize_decision_threshold.py failed" }

Write-Host "Starting Step 8: generate_all_models_calibration.py"
python -m scripts.generate_all_models_calibration
if ($LASTEXITCODE -ne 0) { throw "generate_all_models_calibration.py failed" }

Write-Host "Starting Step 9: benchmark_heuristic_baseline.py"
python -m scripts.benchmark_heuristic_baseline
if ($LASTEXITCODE -ne 0) { throw "benchmark_heuristic_baseline.py failed" }

Write-Host "Starting Step 10: run_static_dynamic_ablation.py"
python -m scripts.run_static_dynamic_ablation
if ($LASTEXITCODE -ne 0) { throw "run_static_dynamic_ablation.py failed" }

Write-Host "All tasks completed successfully."
