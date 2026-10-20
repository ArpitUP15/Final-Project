"""Adaptive, leakage-safe search for tabular classification candidates."""
from __future__ import annotations

import time
from typing import Any, Callable

import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import train_test_split

from app.agents.execution_agent import augment_training_data
from app.agents.ollama_agent import ask_json, llm_enabled
from app.agents.evaluation_agent import (make_model, split_original, is_regression_target,
                                         make_regression_model)
from app.config import (HARD_MAX_ITERATIONS, MIN_PLANNED_ITERATIONS, DEFAULT_TARGET_ACCURACY,
                        DEFAULT_STRETCH_TARGET, DEFAULT_TOTAL_TIME_BUDGET_SECONDS,
                        DEFAULT_CANDIDATE_TIMEOUT_SECONDS)

MIN_ITERATIONS = MIN_PLANNED_ITERATIONS


def clamp_iterations(value: Any, default: int = HARD_MAX_ITERATIONS) -> int:
    """Validate user configuration and enforce the non-negotiable 60-iteration ceiling."""
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = default
    return max(1, min(HARD_MAX_ITERATIONS, parsed))


def adaptive_iteration_plan(
    *, baseline_accuracy: float, rows: int, features: int, imbalance_ratio: float,
    missing_rate: float = 0.0, duplicate_rate: float = 0.0,
    validation_stability: float = 0.0, max_iterations: int = HARD_MAX_ITERATIONS,
) -> dict[str, int | str]:
    """Plan a larger search for difficult datasets while respecting the hard cap."""
    cap = clamp_iterations(max_iterations)
    if baseline_accuracy >= 0.80:
        return {"max_iterations": cap, "planned_iterations": 1, "stop_reason": "baseline_at_or_above_80_percent"}

    # Low baseline starts with a substantially larger budget; complexity increases it.
    planned = 28
    if baseline_accuracy < 0.60:
        planned += 10
    elif baseline_accuracy < 0.70:
        planned += 5
    if rows < 500:
        planned += 5
    elif rows > 10_000:
        planned -= 10
    elif rows > 5_000:
        planned -= 5
    if features > 20:
        planned += 5
    if imbalance_ratio >= 3.0:
        planned += 8
    elif imbalance_ratio >= 2.0:
        planned += 4
    if missing_rate >= 0.10:
        planned += 3
    if duplicate_rate >= 0.10:
        planned += 2
    if validation_stability >= 0.08:
        planned += 5
    planned = min(cap, max(MIN_ITERATIONS, min(HARD_MAX_ITERATIONS, planned)))
    return {"max_iterations": cap, "planned_iterations": planned, "stop_reason": "adaptive_search_budget"}


def _validation_splits(X_train, y_train, seeds=(43, 44)):
    """Use fixed, repeatable validation splits; all preprocessing is fitted within each model pipeline."""
    splits = []
    stratify = y_train if y_train.value_counts().min() >= 2 else None
    for seed in seeds:
        try:
            parts = train_test_split(X_train, y_train, test_size=0.2, random_state=seed, stratify=stratify)
        except ValueError:
            parts = train_test_split(X_train, y_train, test_size=0.2, random_state=seed)
        X_fit, X_val, y_fit, y_val = parts
        splits.append((X_fit, X_val, y_fit, y_val))
    return splits


def _candidate_configs(strategies: list[str], model_names: list[str]) -> list[tuple[str, int, str]]:
    """Create distinct strategy/model combinations in round-robin order."""
    configs: list[tuple[str, int, str]] = []
    estimator_sizes = [80, 120, 180]
    for model_idx, model_name in enumerate(model_names):
        for strategy_idx, method in enumerate(strategies):
            if method == "ClassWeight" and model_name in {"KNeighbors"}:
                continue
            n_estimators = estimator_sizes[(model_idx + strategy_idx) % len(estimator_sizes)]
            config = (method, n_estimators, model_name)
            if config not in configs:
                configs.append(config)
    # Move a few feature-selection baselines to the front, without duplicating attempts.
    front = [("SimilarityOverSampler", 100, "ExtraTreesTopFeatures"),
             ("NoResampling", 100, "ExtraTreesTopFeatures")]
    return list(dict.fromkeys(front + configs))


def _orchestrate_regression(df, target_column, initial_strategy, *, max_iterations,
                            target_accuracy, stretch_target, total_time_budget_seconds,
                            progress_callback=None, cancel_callback=None):
    """Regression search: compare regressors on validation data; never fabricate class labels."""
    started = time.monotonic()
    cap = clamp_iterations(max_iterations)
    X_train, _X_test, y_train, _y_test = split_original(df, target_column)
    splits = _validation_splits(X_train, y_train, seeds=(43, 44, 45))
    baseline_scores = []
    for X_fit, X_val, y_fit, y_val in splits:
        model = make_regression_model(X_fit, "RandomForestRegressor", 100)
        model.fit(X_fit, y_fit)
        pred = model.predict(X_val)
        from sklearn.metrics import r2_score
        baseline_scores.append(float(r2_score(y_val, pred)))
    baseline_score = sum(baseline_scores) / len(baseline_scores)
    plan = min(cap, 18, max(MIN_ITERATIONS, 24 + (8 if len(df) < 500 else 0) + (5 if len(X_train.columns) > 20 else 0)))
    model_names = ["ExtraTreesRegressor", "RandomForestRegressor", "DecisionTreeRegressor",
                   "GradientBoostingRegressor", "HistGradientBoostingRegressor", "Ridge"]
    configs = [(name, n) for name in model_names for n in (80, 120, 180)][:plan]
    history, best = [], {"score": baseline_score, "model": "RandomForestRegressor", "n_estimators": 100, "baseline": True}
    best_iteration = 0
    stop_reason = "planned_iterations_completed"
    for idx, (name, n_estimators) in enumerate(configs, start=1):
        if cancel_callback and cancel_callback():
            stop_reason = "cancelled_by_user"
            break
        if time.monotonic() - started >= total_time_budget_seconds:
            stop_reason = "total_time_budget_exhausted"
            break
        X_fit, X_val, y_fit, y_val = splits[0]
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        error, score, mae, rmse = None, None, None, None
        candidate_started = time.monotonic()
        try:
            model = make_regression_model(X_fit, name, n_estimators)
            model.fit(X_fit, y_fit)
            pred = model.predict(X_val)
            score = float(r2_score(y_val, pred))
            mae = float(mean_absolute_error(y_val, pred))
            rmse = float(mean_squared_error(y_val, pred) ** 0.5)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        candidate_elapsed = time.monotonic() - candidate_started
        candidate_timed_out = candidate_elapsed > DEFAULT_CANDIDATE_TIMEOUT_SECONDS
        accepted = score is not None and score > best["score"] + 1e-12 and not candidate_timed_out
        item = {"iteration": idx, "model_name": name, "n_estimators": n_estimators,
                "selection_metric": "r2", "selection_score": score, "validation_r2": score,
                "validation_mae": mae, "validation_rmse": rmse, "accepted": bool(accepted),
                "status": "candidate_timeout" if candidate_timed_out else ("evaluated" if error is None else "error"), "error": error,
                "candidate_timeout_seconds": DEFAULT_CANDIDATE_TIMEOUT_SECONDS,
                "elapsed_seconds": round(candidate_elapsed, 2)}
        history.append(item)
        if accepted:
            best = {"score": score, "model": name, "n_estimators": n_estimators, "baseline": False}
            best_iteration = idx
        if progress_callback:
            progress_callback({"status": "running", "stage": "optimization", "target_metric": "r2",
                "baseline_score": baseline_score, "target_score": target_accuracy, "stretch_target": stretch_target,
                "max_iterations": cap, "planned_iterations": plan, "completed_iterations": len(history),
                "remaining_iterations": max(0, plan-len(history)), "best_iteration": idx if accepted else None,
                "best_validation_score": best["score"], "current_iteration": idx, "current_model": name,
                "elapsed_seconds": round(time.monotonic()-started, 2)})
        if score is not None and score >= target_accuracy and accepted:
            stop_reason = "target_score_reached"
            break

    train = X_train.copy()
    train[target_column] = y_train.to_numpy()
    train["__autodatagen_source"] = "original"
    augmentation = {"requested_strategy": "NoResampling", "used_strategy": "NoResampling",
        "original_training_rows": int(len(train)), "augmented_training_rows": int(len(train)),
        "generated_rows": 0, "generated_fraction": 0.0,
        "note": "No synthetic regression rows were created because the project has no target-aware regression synthesizer configured. Regressors were compared instead."}
    search = {"task_type": "regression", "selection_metric": "r2", "baseline_score": baseline_score,
        "best_validation_score": best["score"], "baseline_validation_score": baseline_score,
        "planned_iterations": plan, "iteration_budget": plan, "max_iterations": cap,
        "completed_iterations": len(history), "iterations_run": len(history),
        "remaining_iterations": max(0, plan-len(history)), "best_iteration": best_iteration,
        "target_score": target_accuracy, "stretch_target": stretch_target,
        "target_reached": best["score"] >= target_accuracy, "stretch_target_reached": best["score"] >= stretch_target,
        "selected_model": best["model"], "selected_n_estimators": best["n_estimators"],
        "baseline_fallback_selected": best["baseline"], "stop_reason": stop_reason,
        "elapsed_seconds": round(time.monotonic()-started, 2), "history": history,
        "note": "Regression is evaluated with R²/MAE/RMSE, not accuracy. Candidate selection uses validation data only; the untouched test split is used only for final reporting."}
    strategy = {**initial_strategy, "task": "regression", "strategy": "NoResampling"}
    return {"task_type": "regression", "augmented_train": train, "augmentation": augmentation,
            "strategy": strategy, "iteration_search": search, "selected_n_estimators": best["n_estimators"],
            "selected_model": best["model"], "selected_class_weight": None}


def orchestrate_iterations(
    df: pd.DataFrame,
    target_column: str,
    initial_strategy: dict[str, Any],
    target_accuracy: float = DEFAULT_TARGET_ACCURACY,
    stretch_target: float = DEFAULT_STRETCH_TARGET,
    max_iterations: int = HARD_MAX_ITERATIONS,
    high_baseline_threshold: float = 0.80,
    total_time_budget_seconds: float = DEFAULT_TOTAL_TIME_BUDGET_SECONDS,
    candidate_timeout_seconds: float = DEFAULT_CANDIDATE_TIMEOUT_SECONDS,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
    cancel_callback: Callable[[], bool] | None = None,
) -> dict[str, Any]:
    """Search up to 60 candidates and report real progress. Test data is never used for selection."""
    if is_regression_target(df, target_column):
        return _orchestrate_regression(
            df, target_column, initial_strategy, max_iterations=max_iterations,
            target_accuracy=target_accuracy, stretch_target=stretch_target,
            total_time_budget_seconds=total_time_budget_seconds,
            progress_callback=progress_callback, cancel_callback=cancel_callback,
        )
    started = time.monotonic()
    cap = clamp_iterations(max_iterations)
    X_train, _X_test, y_train, _y_test = split_original(df, target_column)
    counts = y_train.value_counts()
    imbalance_ratio = float(counts.max() / counts.min()) if counts.min() else 1.0
    is_imbalanced = imbalance_ratio >= 1.5
    objective = "balanced_accuracy" if is_imbalanced else "accuracy"
    baseline_splits = _validation_splits(X_train, y_train, seeds=(43, 44, 45))

    baseline_accuracy_scores, baseline_balanced_scores, baseline_f1_scores = [], [], []
    for X_fit, X_val, y_fit, y_val in baseline_splits:
        model = make_model(X_fit, n_estimators=120, model_name="RandomForest")
        model.fit(X_fit, y_fit)
        pred = model.predict(X_val)
        baseline_accuracy_scores.append(float(accuracy_score(y_val, pred)))
        baseline_balanced_scores.append(float(balanced_accuracy_score(y_val, pred)))
        baseline_f1_scores.append(float(f1_score(y_val, pred, average="macro", zero_division=0)))
    baseline_accuracy = sum(baseline_accuracy_scores) / len(baseline_accuracy_scores)
    baseline_balanced_accuracy = sum(baseline_balanced_scores) / len(baseline_balanced_scores)
    baseline_f1 = sum(baseline_f1_scores) / len(baseline_f1_scores)
    baseline_score = baseline_balanced_accuracy if is_imbalanced else baseline_accuracy
    validation_stability = float(pd.Series(baseline_accuracy_scores).std(ddof=0))

    numeric = [c for c in X_train.columns if pd.api.types.is_numeric_dtype(X_train[c])]
    categorical = [c for c in X_train.columns if c not in numeric]
    if is_imbalanced:
        strategies = ["SimilarityOverSampler", "ClassWeight", "NoResampling", "RandomOverSampler", "RandomUnderSampler"]
        if categorical:
            strategies.append("SMOTENC")
        else:
            strategies.extend(["SMOTE", "TomekLinks", "SMOTETomek"])
    else:
        strategies = ["SimilarityOverSampler", "NoResampling", "ClassWeight", "RandomOverSampler"]
        if categorical:
            strategies.append("SMOTENC")
        else:
            strategies.extend(["SMOTE", "TomekLinks"])
    strategies = list(dict.fromkeys(strategies))

    model_names = ["ExtraTrees", "RandomForest", "ExtraTreesLeaf2", "RandomForestLeaf2",
                   "DecisionTree", "DecisionTreeShallow", "DecisionTreeDeep", "AdaBoost",
                   "SGDClassifier", "ExtraTreesTopFeatures", "RandomForestTopFeatures"]
    if len(df) <= 5000:
        model_names.extend(["LogisticRegression", "LinearSVC", "SVC", "KNeighbors"])
    configs = _candidate_configs(strategies, model_names)

    analysis_missing = float(df.drop(columns=[target_column]).isna().to_numpy().mean()) if df.shape[1] > 1 else 0.0
    duplicate_rate = float(df.duplicated().mean()) if len(df) else 0.0
    plan = adaptive_iteration_plan(
        baseline_accuracy=baseline_accuracy, rows=len(df), features=len(X_train.columns),
        imbalance_ratio=imbalance_ratio, missing_rate=analysis_missing,
        duplicate_rate=duplicate_rate, validation_stability=validation_stability,
        max_iterations=cap,
    )
    planned_iterations = int(plan["planned_iterations"])
    llm_search_guidance = None
    if llm_enabled() and configs:
        guidance = ask_json(
            "You are the search-planning agent for tabular classification. Rank candidate strategy/model options "
            "to try first based only on the supplied dataset profile. You may only reorder candidates; do not "
            "invent options or claim a score. Return JSON with preferred_models (array of exact model names) "
            "and preferred_strategies (array of exact strategy names), plus rationale (short string).",
            {
                "rows": int(len(df)), "features": int(len(X_train.columns)),
                "numeric_features": len(numeric), "categorical_features": len(categorical),
                "class_count_values": sorted([int(v) for v in counts.values], reverse=True),
                "number_of_classes": int(y_train.nunique()),
                "imbalance_ratio": round(imbalance_ratio, 4), "baseline_accuracy": round(baseline_accuracy, 4),
                "baseline_balanced_accuracy": round(baseline_balanced_accuracy, 4),
                "missing_rate": round(analysis_missing, 4), "duplicate_rate": round(duplicate_rate, 4),
                "allowed_models": model_names, "allowed_strategies": strategies,
                "candidate_count": len(configs),
            }, num_predict=240,
        )
        if guidance:
            preferred_models = [m for m in guidance.get("preferred_models", []) if m in model_names] if isinstance(guidance.get("preferred_models", []), list) else []
            preferred_strategies = [m for m in guidance.get("preferred_strategies", []) if m in strategies] if isinstance(guidance.get("preferred_strategies", []), list) else []
            model_rank = {name: idx for idx, name in enumerate(preferred_models)}
            strategy_rank = {name: idx for idx, name in enumerate(preferred_strategies)}
            original_index = {cfg: idx for idx, cfg in enumerate(configs)}
            configs.sort(key=lambda cfg: (strategy_rank.get(cfg[0], len(strategy_rank)), model_rank.get(cfg[2], len(model_rank)), original_index[cfg]))
            llm_search_guidance = {"status": "used", "rationale": str(guidance.get("rationale", ""))[:1200],
                                   "preferred_models": preferred_models, "preferred_strategies": preferred_strategies}
    if llm_search_guidance is None:
        llm_search_guidance = {"status": "fallback", "rationale": "Ollama unavailable/disabled; deterministic candidate order used."}
    configs = configs[:planned_iterations]

    best = {
        "selection_score": baseline_score, "validation_accuracy": baseline_accuracy,
        "strategy": "NoResampling", "n_estimators": 120, "model_name": "RandomForest",
        "class_weight": None, "macro_f1": baseline_f1, "is_baseline_fallback": True,
    }
    history: list[dict[str, Any]] = []
    stop_reason = "planned_iterations_completed"
    best_iteration = 0

    def emit(status: str, current_iteration: int, current_model: str | None = None,
              current_strategy: str | None = None, current_score: float | None = None):
        if progress_callback:
            elapsed = time.monotonic() - started
            progress_callback({
                "status": status, "stage": "optimization", "baseline_score": baseline_score,
                "baseline_accuracy": baseline_accuracy, "target_score": target_accuracy,
                "stretch_target": stretch_target, "target_metric": objective,
                "max_iterations": cap, "planned_iterations": planned_iterations,
                "completed_iterations": len(history), "remaining_iterations": max(0, planned_iterations-len(history)),
                "best_iteration": best_iteration, "best_validation_score": best["selection_score"],
                "current_iteration": current_iteration, "current_model": current_model,
                "current_strategy": current_strategy, "current_validation_score": current_score,
                "elapsed_seconds": round(elapsed, 2), "stop_reason": None,
            })

    emit("running", 0, "baseline", "NoResampling", baseline_score)
    for idx, (method, n_estimators, model_name) in enumerate(configs, start=1):
        elapsed = time.monotonic() - started
        if cancel_callback and cancel_callback():
            stop_reason = "cancelled_by_user"
            break
        if elapsed >= max(1.0, float(total_time_budget_seconds)):
            stop_reason = "total_time_budget_exhausted"
            break

        candidate_accuracies, candidate_scores, candidate_f1s = [], [], []
        error = None
        candidate_started = time.monotonic()
        weight = "balanced" if method == "ClassWeight" and model_name not in {"KNeighbors"} else None
        try:
            # Screen every candidate on one identical fixed validation split. Baseline
            # stability is measured on three splits above; the final shortlist is selected
            # without touching the test set. This bounds search cost for larger iteration budgets.
            for split_idx, (X_fit, X_val, y_fit, y_val) in enumerate(baseline_splits[:1]):
                if cancel_callback and cancel_callback():
                    raise InterruptedError("Cancellation requested")
                fit_df = X_fit.copy()
                fit_df[target_column] = y_fit.to_numpy()
                config = {**initial_strategy, "strategy": method}
                candidate_train, _metadata = augment_training_data(fit_df, target_column, config)
                candidate_train = candidate_train.drop(columns=["__autodatagen_source"], errors="ignore")
                X_aug = candidate_train.drop(columns=[target_column]).reindex(columns=X_fit.columns)
                y_aug = candidate_train[target_column]
                model = make_model(X_aug, n_estimators=n_estimators, class_weight=weight, model_name=model_name)
                model.fit(X_aug, y_aug)
                pred = model.predict(X_val)
                candidate_accuracies.append(float(accuracy_score(y_val, pred)))
                candidate_scores.append(float(balanced_accuracy_score(y_val, pred) if is_imbalanced else accuracy_score(y_val, pred)))
                candidate_f1s.append(float(f1_score(y_val, pred, average="macro", zero_division=0)))
        except InterruptedError:
            stop_reason = "cancelled_by_user"
            break
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"

        accuracy = sum(candidate_accuracies) / len(candidate_accuracies) if candidate_accuracies else None
        score = sum(candidate_scores) / len(candidate_scores) if candidate_scores else None
        macro_f1 = sum(candidate_f1s) / len(candidate_f1s) if candidate_f1s else None
        accuracy_preserved = accuracy is not None and accuracy + 1e-12 >= baseline_accuracy
        objective_preserved = score is not None and score + 1e-12 >= baseline_score
        accuracy_improved = accuracy is not None and accuracy > baseline_accuracy + 1e-12
        score_improved = score is not None and score > baseline_score + 1e-12
        candidate_elapsed = time.monotonic() - candidate_started
        candidate_timed_out = candidate_elapsed > candidate_timeout_seconds
        # Do not accept an objective improvement if raw accuracy regresses; for imbalanced
        # data, balanced accuracy is the primary score, but raw accuracy remains a guardrail.
        # A candidate that exceeded its configured time budget is recorded but not selected.
        accepted = bool(score_improved and accuracy_preserved and not candidate_timed_out)
        target_reached = bool(score is not None and score >= target_accuracy and accepted)
        item = {
            "iteration": idx, "strategy": method, "n_estimators": n_estimators, "model_name": model_name,
            "validation_splits": len(candidate_accuracies), "baseline_validation_accuracy": round(baseline_accuracy, 6),
            "validation_accuracy": round(accuracy, 6) if accuracy is not None else None,
            "accuracy_delta_vs_baseline": round(accuracy-baseline_accuracy, 6) if accuracy is not None else None,
            "selection_metric": objective, "selection_score": round(score, 6) if score is not None else None,
            "validation_macro_f1": round(macro_f1, 6) if macro_f1 is not None else None,
            "accuracy_preserved": bool(accuracy_preserved), "accuracy_improved": bool(accuracy_improved),
            "selection_score_improved": bool(score_improved), "accepted": accepted,
            "target_accuracy": target_accuracy, "elapsed_seconds": round(candidate_elapsed, 2),
            "candidate_timeout_seconds": candidate_timeout_seconds,
            "status": "target_reached" if target_reached else (
                "candidate_timeout" if candidate_timed_out else
                "rejected_accuracy_regression" if accuracy is not None and not accuracy_preserved else
                "rejected_no_objective_improvement" if score is not None and not accepted else
                "error" if error else "evaluated"), "error": error,
        }
        history.append(item)
        if accepted and (score > best["selection_score"] + 1e-12 or
                         (abs(score-best["selection_score"]) <= 1e-12 and (macro_f1 or 0) > best["macro_f1"])):
            best = {"selection_score": score, "validation_accuracy": accuracy, "strategy": method,
                    "n_estimators": n_estimators, "model_name": model_name, "class_weight": weight,
                    "macro_f1": macro_f1 or 0.0, "is_baseline_fallback": False}
            best_iteration = idx
        emit("running", idx, model_name, method, score)
        if target_reached:
            stop_reason = "target_score_reached"
            break
        if time.monotonic()-started >= total_time_budget_seconds:
            stop_reason = "total_time_budget_exhausted"
            break
        # If the validation objective has not improved for a while, the remaining search
        # switches are already diverse (model family + resampling); continue only within plan.

    if cancel_callback and cancel_callback():
        stop_reason = "cancelled_by_user"

    output_strategy_name = best["strategy"] if not best["is_baseline_fallback"] else "SimilarityOverSampler"
    selected_strategy = {**initial_strategy, "strategy": output_strategy_name, "model_name": best["model_name"]}
    full_train = X_train.copy()
    full_train[target_column] = y_train.to_numpy()
    augmented, metadata = augment_training_data(full_train, target_column, selected_strategy)
    metadata["selected_strategy"] = output_strategy_name
    metadata["selected_model"] = best["model_name"]
    metadata["class_weight"] = best["class_weight"]
    metadata["model_candidate_accepted"] = not best["is_baseline_fallback"]
    if best["is_baseline_fallback"]:
        metadata["note"] = str(metadata.get("note", "")) + " No candidate improved the validation objective while preserving raw accuracy; baseline model retained and SimilarityOverSampler still generated export rows."

    completed = len(history)
    search = {
        "dataset_balance": "imbalanced" if is_imbalanced else "balanced", "imbalance_ratio": round(imbalance_ratio, 4),
        "selection_metric": objective, "validation_splits": 1, "baseline_validation_accuracy": round(baseline_accuracy, 6),
        "baseline_validation_balanced_accuracy": round(baseline_balanced_accuracy, 6),
        "high_baseline_threshold": high_baseline_threshold, "max_iterations": cap,
        "planned_iterations": planned_iterations, "iteration_budget": planned_iterations,
        "completed_iterations": completed, "remaining_iterations": max(0, planned_iterations-completed),
        "iterations_run": completed, "best_iteration": best_iteration,
        "baseline_score": round(baseline_score, 6), "target_score": target_accuracy,
        "stretch_target": stretch_target, "target_metric": objective,
        "target_reached": best["selection_score"] >= target_accuracy,
        "stretch_target_reached": best["selection_score"] >= stretch_target,
        "best_validation_accuracy": round(best["validation_accuracy"], 6),
        "best_validation_score": round(best["selection_score"], 6), "best_selection_score": round(best["selection_score"], 6),
        "selected_n_estimators": best["n_estimators"], "selected_model": best["model_name"],
        "selected_class_weight": best["class_weight"], "baseline_fallback_selected": best["is_baseline_fallback"],
        "output_augmentation_strategy": output_strategy_name, "elapsed_seconds": round(time.monotonic()-started, 2),
        "llm_search_guidance": llm_search_guidance,
        "stop_reason": stop_reason, "history": history,
        "note": (f"Adaptive search completed {completed}/{planned_iterations} candidate evaluations. "
                 f"Baseline validation accuracy was {baseline_accuracy:.1%}; primary metric is {objective}. "
                 f"80% is the target and 90% is a stretch target, neither guaranteed. Final test data was not used for model selection."),
    }
    emit("completed" if stop_reason != "cancelled_by_user" else "cancelled", completed,
         best["model_name"], output_strategy_name, best["selection_score"])
    return {"task_type": "classification", "augmented_train": augmented, "augmentation": metadata, "strategy": selected_strategy,
            "iteration_search": search, "selected_n_estimators": best["n_estimators"],
            "selected_model": best["model_name"], "selected_class_weight": best["class_weight"]}
