# src/compare_tuning_results.py
"""
Pulls all three tracked runs (baseline, grid search, random search)
from MLflow and prints a consolidated comparison table.
"""
from mlflow.tracking import MlflowClient
import mlflow

mlflow.set_tracking_uri("sqlite:///mlflow.db")
client = MlflowClient()

experiment = client.get_experiment_by_name("iris-hyperparameter-tuning")
runs = client.search_runs(experiment_ids=[experiment.experiment_id])

print(f"{'Run Name':<28}{'CV f1_macro':<15}{'Test Accuracy':<15}{'Total Fits':<12}")
print("-" * 70)
for run in sorted(runs, key=lambda r: r.data.tags.get("mlflow.runName", "")):
    name = run.data.tags.get("mlflow.runName", "unknown")
    cv_score = run.data.metrics.get("cv_f1_macro_mean", run.data.metrics.get("best_cv_f1_macro", 0))
    test_acc = run.data.metrics.get("test_accuracy", 0)
    n_iter = run.data.params.get("total_combinations") or run.data.params.get("n_iter") or "1"
    total_fits = int(n_iter) * 5 if name != "baseline_decision_tree" else 5
    print(f"{name:<28}{cv_score:<15.4f}{test_acc:<15.4f}{total_fits:<12}")

# Hyperparameter Tuning Analysis
## 1. Baseline Model
Model:
DecisionTreeClassifier
CV F1 Macro:
0.9328
Test Accuracy:
0.9333
## 2. Grid Search
Model:
RandomForestClassifier
Total combinations:
72
Cross-validation:
5-fold
Total fits:
360
Best CV F1 Macro:
0.9583
Test Accuracy:
0.9667
## 3. Random Search
Model:
RandomForestClassifier
Number of iterations:
30
Cross-validation:
5-fold
Total fits:
150
Best CV F1 Macro:
0.9581
Test Accuracy:
0.9667
## 4. Comparison
Grid Search evaluates all combinations in the specified
hyperparameter grid.
Random Search evaluates a selected number of combinations
from the search space.
In this experiment, Grid Search required 360 model fits,
while Random Search required 150 model fits.
Both tuning approaches produced higher CV F1 Macro
than the baseline model.
