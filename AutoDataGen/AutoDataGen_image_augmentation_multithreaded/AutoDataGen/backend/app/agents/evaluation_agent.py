import pandas as pd
from app.agents.ollama_agent import ask_json
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier,
                              RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor,
                              HistGradientBoostingRegressor)
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.linear_model import LogisticRegression, SGDClassifier, Ridge
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score, recall_score,
    mean_absolute_error, mean_squared_error, r2_score,
    f1_score, confusion_matrix, roc_auc_score, average_precision_score,
    matthews_corrcoef, classification_report,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.base import BaseEstimator, TransformerMixin


def split_original(df, target):
    clean = df.dropna(subset=[target]).copy()
    if clean[target].nunique() < 2:
        raise ValueError("Target must have at least two classes.")
    if len(clean) < 20:
        raise ValueError("At least 20 rows are required.")
    X, y = clean.drop(columns=[target]), clean[target]
    if X.shape[1] == 0:
        raise ValueError("At least one feature column is required.")
    stratify = y if y.value_counts().min() >= 2 else None
    try:
        return train_test_split(X, y, test_size=.25, random_state=42, stratify=stratify)
    except ValueError:
        return train_test_split(X, y, test_size=.25, random_state=42)



class TopKMutualInfoSelector(BaseEstimator, TransformerMixin):
    """Select a small set of high-signal encoded features without leaking validation data."""
    def __init__(self, k=5, random_state=42):
        self.k = k
        self.random_state = random_state

    def fit(self, X, y):
        self.selector_ = SelectKBest(
            score_func=lambda features, labels: mutual_info_classif(
                features, labels, random_state=self.random_state
            ),
            k=min(self.k, X.shape[1]),
        )
        self.selector_.fit(X, y)
        return self

    def transform(self, X):
        return self.selector_.transform(X)

    def get_support(self):
        return self.selector_.get_support()

def model_feature_columns(X):
    """Exclude obvious row identifiers that add noise rather than generalizable signal."""
    kept = []
    for col in X.columns:
        name = str(col).strip().lower().replace("-", "_").replace(" ", "_")
        id_name = (
            name in {"id", "index", "row_id", "record_id", "uuid", "guid", "key"}
            or name.endswith("_id") or name.endswith("_uuid")
            or name.endswith("_index") or name.endswith("_key")
        )
        if id_name:
            continue
        kept.append(col)
    return kept or list(X.columns)


def make_model(X, n_estimators=200, class_weight=None, model_name="RandomForest"):
    # Keep identifier-like columns out of the predictive model while preserving them in exports.
    X = X[model_feature_columns(X)]
    num = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    cat = [c for c in X.columns if c not in num]
    trans = []
    if num:
        trans.append(("num", SimpleImputer(strategy="median"), num))
    if cat:
        trans.append(("cat", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]), cat))
    models = {
        "RandomForest": RandomForestClassifier(
            n_estimators=n_estimators, random_state=42, n_jobs=2, class_weight=class_weight,
            min_samples_leaf=1,
        ),
        "ExtraTrees": ExtraTreesClassifier(
            n_estimators=max(100, n_estimators), random_state=42, n_jobs=2,
            class_weight=class_weight, min_samples_leaf=1,
        ),
        "ExtraTreesTopFeatures": ExtraTreesClassifier(
            n_estimators=max(100, n_estimators), random_state=42, n_jobs=2,
            class_weight=class_weight, min_samples_leaf=1,
        ),
        "RandomForestTopFeatures": RandomForestClassifier(
            n_estimators=n_estimators, random_state=42, n_jobs=2,
            class_weight=class_weight, min_samples_leaf=1,
        ),
        "DecisionTree": DecisionTreeClassifier(random_state=42, class_weight=class_weight,
                                                min_samples_leaf=2),
        "DecisionTreeShallow": DecisionTreeClassifier(random_state=42, class_weight=class_weight,
                                                       max_depth=6, min_samples_leaf=15),
        "DecisionTreeDeep": DecisionTreeClassifier(random_state=42, class_weight=class_weight,
                                                    max_depth=10, min_samples_leaf=5),
        "ExtraTreesLeaf2": ExtraTreesClassifier(
            n_estimators=max(100, n_estimators), random_state=42, n_jobs=2,
            class_weight=class_weight, min_samples_leaf=2,
        ),
        "RandomForestLeaf2": RandomForestClassifier(
            n_estimators=n_estimators, random_state=42, n_jobs=2,
            class_weight=class_weight, min_samples_leaf=2,
        ),
        "LogisticRegression": LogisticRegression(
            max_iter=500, class_weight=class_weight, random_state=42,
        ),
        "LinearSVC": LinearSVC(class_weight=class_weight, random_state=42, max_iter=5000),
        "SGDClassifier": SGDClassifier(loss="modified_huber", max_iter=2000, tol=1e-3,
                                        class_weight=class_weight, random_state=42),
        "SVC": SVC(class_weight=class_weight, random_state=42),
        "KNeighbors": KNeighborsClassifier(n_neighbors=7, weights="distance", n_jobs=1),
        "AdaBoost": AdaBoostClassifier(
            estimator=DecisionTreeClassifier(max_depth=2, class_weight=class_weight, random_state=42),
            n_estimators=min(max(50, n_estimators), 300), random_state=42,
        ),
    }
    if model_name not in models:
        raise ValueError(f"Unsupported model: {model_name}")
    steps = [("prep", ColumnTransformer(trans))]
    if model_name.endswith("TopFeatures"):
        steps.append(("select_top_features", TopKMutualInfoSelector(k=5)))
    steps.append(("model", models[model_name]))
    return Pipeline(steps)


def metrics(model, X, y):
    pred = model.predict(X)
    labels = list(model.classes_)
    result = {
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "precision_macro": float(precision_score(y, pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y, pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y, pred, average="macro", zero_division=0)),
        "matthews_corrcoef": float(matthews_corrcoef(y, pred)),
        "labels": [str(x) for x in labels],
        "confusion_matrix": confusion_matrix(y, pred, labels=labels).tolist(),
        "classification_report": classification_report(
            y, pred, labels=labels, target_names=[str(x) for x in labels],
            output_dict=True, zero_division=0,
        ),
        "roc_auc": None,
        "average_precision": None,
    }
    try:
        proba = model.predict_proba(X)
        if len(labels) == 2:
            result["roc_auc"] = float(roc_auc_score(y, proba[:, 1]))
            result["average_precision"] = float(average_precision_score(y == labels[1], proba[:, 1]))
        else:
            result["roc_auc"] = float(roc_auc_score(y, proba, labels=labels, multi_class="ovr", average="macro"))
    except Exception:
        pass
    return result


def evaluate(df, target, augmented_train, n_estimators=200, class_weight=None, model_name="RandomForest"):
    X_train, X_test, y_train, y_test = split_original(df, target)
    baseline = make_model(X_train, n_estimators=200, model_name="RandomForest")
    baseline.fit(X_train, y_train)
    augmented = augmented_train.drop(columns=["__autodatagen_source"], errors="ignore")
    X_aug = augmented.drop(columns=[target]).reindex(columns=X_train.columns)
    y_aug = augmented[target]
    model_aug = make_model(X_aug, n_estimators=n_estimators, class_weight=class_weight, model_name=model_name)
    model_aug.fit(X_aug, y_aug)
    before, after = metrics(baseline, X_test, y_test), metrics(model_aug, X_test, y_test)
    keys = ["accuracy", "balanced_accuracy", "precision_macro", "recall_macro",
            "f1_macro", "matthews_corrcoef", "roc_auc", "average_precision"]
    delta = {key: (round(after[key] - before[key], 6)
                   if after[key] is not None and before[key] is not None else None)
             for key in keys}
    counts = y_train.value_counts()
    imbalance_ratio = float(counts.max() / counts.min()) if counts.min() else None
    class_counts = y_train.value_counts()
    n_classes = int(y_train.nunique())
    majority_accuracy = float(class_counts.max() / class_counts.sum()) if len(class_counts) else 0.0
    interpretation = (
        "Metrics use the same untouched original test rows. For imbalanced data, inspect "
        "balanced accuracy, per-class precision/recall/F1, confusion matrix and average precision; "
        "accuracy alone can hide minority-class failures. Resampling was confined to training data."
    )
    if n_classes >= 5 and max(before["accuracy"], after["accuracy"]) < 0.35:
        interpretation += (
            " Exact-match accuracy remains low across a multi-class target. This often means the "
            "available features have weak predictive signal or the labels contain substantial noise; "
            "resampling cannot manufacture a reliable relationship, so a 90% target may be unattainable."
        )
    llm_metric_keys = ["accuracy", "balanced_accuracy", "precision_macro", "recall_macro", "f1_macro",
                       "matthews_corrcoef", "roc_auc", "average_precision"]
    llm_review = ask_json(
        "You are an ML evaluation reviewer. Interpret only these aggregate test metrics. "
        "Return JSON with keys: summary (string), concerns (array of strings), next_steps (array of strings). "
        "Do not change metrics, claim causation, promise an accuracy target, or recommend using test data for tuning.",
        {"baseline_metrics": {k: before.get(k) for k in llm_metric_keys},
         "selected_metrics": {k: after.get(k) for k in llm_metric_keys}, "metric_delta": delta,
         "target_diagnostics": {"number_of_classes": n_classes, "majority_class_baseline_accuracy": majority_accuracy,
                                "imbalance_ratio": round(imbalance_ratio, 4) if imbalance_ratio is not None else None},
         "task_type": "classification"}, num_predict=240,
    )
    if llm_review:
        llm_review["source"] = "ollama"
        interpretation += " Ollama review: " + str(llm_review.get("summary", ""))[:1000]
    else:
        llm_review = {"summary": "Ollama unavailable/disabled; deterministic metric interpretation retained.", "concerns": [], "next_steps": [], "source": "fallback"}
    return {
        "llm_review": llm_review,
        "target_diagnostics": {
            "number_of_classes": n_classes,
            "majority_class_baseline_accuracy": majority_accuracy,
            "uniform_random_reference_accuracy": 1.0 / n_classes if n_classes else None,
        },
        "split": {
            "method": "stratified split when feasible; resampling only on training data",
            "train_rows_original": len(X_train), "test_rows_untouched": len(X_test),
            "random_state": 42, "test_size": .25,
        },
        "class_distribution": {str(k): int(v) for k, v in df[target].value_counts().items()},
        "imbalance_ratio": round(imbalance_ratio, 4) if imbalance_ratio is not None else None,
        "baseline": before, "augmented": after, "delta": delta,
        "baseline_model": "RandomForest", "augmented_model": model_name,
        "interpretation": interpretation,
    }



def is_regression_target(df, target):
    """Conservative task inference: numeric targets with many distinct values are regression."""
    y = df[target].dropna()
    if not pd.api.types.is_numeric_dtype(y) or y.nunique() < 2:
        return False
    threshold = max(20, int(0.05 * len(y)))
    return int(y.nunique()) > threshold


def make_regression_model(X, model_name="RandomForestRegressor", n_estimators=120):
    """Leakage-safe preprocessing plus a regression estimator."""
    X = X[model_feature_columns(X)]
    num = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    cat = [c for c in X.columns if c not in num]
    transformers = []
    if num:
        transformers.append(("num", SimpleImputer(strategy="median"), num))
    if cat:
        transformers.append(("cat", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), cat))
    models = {
        "RandomForestRegressor": RandomForestRegressor(n_estimators=n_estimators, random_state=42, n_jobs=2, min_samples_leaf=2),
        "ExtraTreesRegressor": ExtraTreesRegressor(n_estimators=n_estimators, random_state=42, n_jobs=2, min_samples_leaf=2),
        "DecisionTreeRegressor": DecisionTreeRegressor(random_state=42, min_samples_leaf=5),
        "GradientBoostingRegressor": GradientBoostingRegressor(n_estimators=min(n_estimators, 150), random_state=42),
        "HistGradientBoostingRegressor": HistGradientBoostingRegressor(max_iter=min(n_estimators, 150), random_state=42),
        "Ridge": Ridge(alpha=1.0),
    }
    if model_name not in models:
        raise ValueError(f"Unsupported regression model: {model_name}")
    return Pipeline([("prep", ColumnTransformer(transformers, sparse_threshold=0.0)), ("model", models[model_name])])


def evaluate_regression(df, target, model_name="RandomForestRegressor", n_estimators=120):
    """Evaluate a selected regression model once on an untouched test split."""
    X_train, X_test, y_train, y_test = split_original(df, target)
    baseline = make_regression_model(X_train, "RandomForestRegressor", 120)
    baseline.fit(X_train, y_train)
    selected = make_regression_model(X_train, model_name, n_estimators)
    selected.fit(X_train, y_train)
    def score(model):
        prediction = model.predict(X_test)
        mse = mean_squared_error(y_test, prediction)
        return {"accuracy": None, "mae": float(mean_absolute_error(y_test, prediction)),
                "rmse": float(mse ** 0.5), "r2": float(r2_score(y_test, prediction)),
                "labels": [], "confusion_matrix": [], "classification_report": {},
                "balanced_accuracy": None, "precision_macro": None, "recall_macro": None,
                "f1_macro": None, "matthews_corrcoef": None, "roc_auc": None,
                "average_precision": None}
    before, after = score(baseline), score(selected)
    delta = {key: (after[key] - before[key]) if after.get(key) is not None and before.get(key) is not None else None
             for key in ("mae", "rmse", "r2")}
    llm_review = ask_json(
        "You are an ML evaluation reviewer. Interpret only these aggregate regression metrics. "
        "Return JSON with summary, concerns (array), and next_steps (array). Do not call R² accuracy or suggest tuning on test data.",
        {"baseline_metrics": before, "selected_metrics": after, "metric_delta": delta, "task_type": "regression"}, num_predict=180,
    )
    if llm_review:
        llm_review["source"] = "ollama"
    else:
        llm_review = {"summary": "Ollama unavailable/disabled; deterministic interpretation retained.", "concerns": [], "next_steps": [], "source": "fallback"}
    return {
        "llm_review": llm_review,
        "task_type": "regression",
        "split": {"method": "train/test split before fitting; preprocessing fitted on training data only",
                  "train_rows_original": len(X_train), "test_rows_untouched": len(X_test),
                  "random_state": 42, "test_size": 0.25},
        "baseline": before, "augmented": after,
        "delta": delta,
        "baseline_model": "RandomForestRegressor", "augmented_model": model_name,
        "interpretation": "Regression is reported with MAE, RMSE, and R²; accuracy percentages are not applicable. Test data was not used for model selection." + (" Ollama review: " + str(llm_review.get("summary", ""))[:1000] if llm_review.get("source") == "ollama" else ""),
        "test_set_usage": {"used_for_model_selection": False},
    }
