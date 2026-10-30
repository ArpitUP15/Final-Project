import io
from typing import TypedDict, Any
import pandas as pd
from langgraph.graph import StateGraph, START, END
from app.agents.analysis_agent import analyze_dataframe
from app.agents.strategy_agent import choose_strategy
from app.agents.execution_agent import augment_training_data, augment_unlabeled_text_column
from app.agents.evaluation_agent import split_original, evaluate, evaluate_regression
from app.agents.orchestrator_agent import orchestrate_iterations
from app.config import DEFAULT_MAX_ITERATIONS, DEFAULT_TARGET_ACCURACY, DEFAULT_STRETCH_TARGET, DEFAULT_TOTAL_TIME_BUDGET_SECONDS

class State(TypedDict, total=False):
    df: Any
    target_column: str
    run_id: str
    filename: str
    output_format: str
    analysis: dict
    strategy: dict
    augmented_train: Any
    augmentation: dict
    evaluation: dict
    iteration_search: dict
    selected_n_estimators: int
    selected_model: str
    selected_class_weight: Any
    task_type: str
    progress_callback: Any
    cancel_callback: Any
    max_iterations: int
    target_accuracy: float
    stretch_target: float
    total_time_budget_seconds: float
    final_report: dict
    dataset_bytes: bytes
    download_suffix: str

def analyze_node(s):
    return {"analysis":analyze_dataframe(s["df"])}

def strategy_node(s):
    return {"strategy":choose_strategy(s["df"],s["target_column"],s["analysis"])}

def execute_node(s):
    """Delegate bounded validation-only iteration management to the Orchestrator agent."""
    result = orchestrate_iterations(
        s["df"], s["target_column"], s["strategy"],
        max_iterations=s.get("max_iterations", DEFAULT_MAX_ITERATIONS),
        target_accuracy=s.get("target_accuracy", DEFAULT_TARGET_ACCURACY),
        stretch_target=s.get("stretch_target", DEFAULT_STRETCH_TARGET),
        total_time_budget_seconds=s.get("total_time_budget_seconds", DEFAULT_TOTAL_TIME_BUDGET_SECONDS),
        progress_callback=s.get("progress_callback"),
        cancel_callback=s.get("cancel_callback"),
    )
    return {
        "task_type": result.get("task_type", "classification"),
        "augmented_train": result["augmented_train"],
        "augmentation": result["augmentation"],
        "strategy": result["strategy"],
        "iteration_search": result["iteration_search"],
        "selected_n_estimators": result["selected_n_estimators"],
        "selected_model": result["selected_model"],
        "selected_class_weight": result["selected_class_weight"],
    }

def evaluate_node(s):
    """Evaluate the selected candidate once on the untouched test split; never use test data to select it."""
    if s.get("task_type") == "regression":
        evaluation = evaluate_regression(
            s["df"], s["target_column"],
            model_name=s.get("selected_model", "RandomForestRegressor"),
            n_estimators=s.get("selected_n_estimators", 120),
        )
        return {"evaluation": evaluation}
    evaluation = evaluate(
        s["df"], s["target_column"], s["augmented_train"],
        n_estimators=s.get("selected_n_estimators", 120),
        class_weight=s.get("selected_class_weight"),
        model_name=s.get("selected_model", "RandomForest"),
    )
    evaluation["test_set_usage"] = {
        "used_for_model_selection": False,
        "note": "The untouched test set is used only for final reporting. Candidate selection and augmentation strategy selection use validation splits only.",
    }
    return {"evaluation": evaluation}

def export_node(s):
    Xtr, Xte, ytr, yte = split_original(s["df"],s["target_column"])
    test = Xte.copy()
    test[s["target_column"]] = yte.values
    test["__autodatagen_source"] = "original_test"
    train = s["augmented_train"].copy()
    combined = pd.concat([train,test],ignore_index=True,sort=False)
    fmt = s.get("output_format","original")
    original_ext = s.get("filename","data.csv").lower().rsplit(".",1)[-1]
    suffix = ("."+original_ext if original_ext in {"csv","xlsx","xlsm"} else ".csv") if fmt=="original" else "."+fmt
    buffer = io.BytesIO()
    if suffix == ".csv":
        combined.to_csv(buffer,index=False); fmt_name="csv"
    else:
        combined.to_excel(buffer,index=False,engine="openpyxl"); fmt_name="xlsx"
    report = {"run_id":s["run_id"],"filename":s["filename"],"target_column":s["target_column"],"status":"completed",
        "task_type": s.get("task_type", "classification"), "analysis":s["analysis"],"strategy":s["strategy"],"augmentation":s["augmentation"],"evaluation":s["evaluation"],
        "iteration_search":s.get("iteration_search", {}),
        "selected_model": s.get("selected_model", "RandomForest"),
        "export":{"format":fmt_name,"rows":len(combined),"columns":list(map(str,combined.columns)),
        "note":"Output combines augmented training partition with untouched original test partition. __autodatagen_source identifies row origin."},
        "warnings":["Synthetic rows are not guaranteed realistic, unique, or privacy-preserving.","Quality score is a heuristic."]}
    return {"final_report":report,"dataset_bytes":buffer.getvalue(),"download_suffix":suffix}

def build_graph():
    b=StateGraph(State)
    b.add_node("analyze",analyze_node); b.add_node("choose_strategy",strategy_node)
    b.add_node("execute",execute_node); b.add_node("evaluate",evaluate_node); b.add_node("export",export_node)
    b.add_edge(START,"analyze"); b.add_edge("analyze","choose_strategy")
    b.add_edge("choose_strategy","execute"); b.add_edge("execute","evaluate")
    b.add_edge("evaluate","export"); b.add_edge("export",END)
    return b.compile()

workflow=build_graph()

def _looks_like_unlabeled_text(df, target_column):
    """Detect a free-text column mistakenly supplied as a supervised target."""
    if target_column not in df.columns:
        raise ValueError(f"Target/text column '{target_column}' was not found in the dataset.")
    series = df[target_column].dropna().astype(str).str.strip()
    if len(series) < 2:
        return False
    unique_ratio = series.nunique() / max(1, len(series))
    avg_len = float(series.str.len().mean()) if len(series) else 0.0
    return unique_ratio >= 0.50 and avg_len >= 12

def _run_text_only_augmentation(df, text_column, run_id, filename, output_format):
    """Augment an unlabeled text column; classification accuracy is not defined without labels."""
    augmented, metadata = augment_unlabeled_text_column(
        df, text_column, method="TextBackTranslation", max_generated_fraction=0.10
    )
    fmt = output_format or "original"
    original_ext = filename.lower().rsplit(".", 1)[-1]
    suffix = ("." + original_ext if original_ext in {"txt", "csv", "xlsx", "xlsm"} else ".csv") if fmt == "original" else "." + fmt
    buffer = io.BytesIO()
    if suffix == ".txt":
        # TXT export contains one example per line, without a CSV header.
        text_lines = augmented[text_column].fillna("").astype(str).tolist()
        buffer.write(("\n".join(text_lines) + "\n").encode("utf-8"))
        fmt_name = "txt"
    elif suffix == ".csv":
        augmented.to_csv(buffer, index=False)
        fmt_name = "csv"
    else:
        augmented.to_excel(buffer, index=False, engine="openpyxl")
        fmt_name = "xlsx"
    report = {
        "run_id": run_id, "filename": filename, "target_column": text_column,
        "status": "completed", "task_type": "text_augmentation",
        "strategy": {"strategy": "TextBackTranslation", "reason": "Unlabeled free-text column detected"},
        "augmentation": metadata,
        "evaluation": None,
        "iteration_search": {"completed_iterations": 0, "max_iterations": 0,
                             "stop_reason": "Text-only augmentation does not compute classification accuracy without labels"},
        "selected_model": None,
        "export": {"format": fmt_name, "rows": len(augmented), "columns": list(map(str, augmented.columns)),
                   "note": "Text-only augmentation output. No classification accuracy is reported because no label column was provided."},
        "warnings": ["No supervised label column was provided; accuracy metrics are not applicable.",
                     "Review generated text for semantic fidelity before using it."]
    }
    return {**report, "dataset_bytes": buffer.getvalue(), "download_suffix": suffix}

def run_workflow(df, target_column, run_id, filename, output_format="original", *,
                 progress_callback=None, cancel_callback=None, max_iterations=DEFAULT_MAX_ITERATIONS,
                 target_accuracy=DEFAULT_TARGET_ACCURACY, stretch_target=DEFAULT_STRETCH_TARGET, total_time_budget_seconds=DEFAULT_TOTAL_TIME_BUDGET_SECONDS):
    # One-column text files are corpora even when their lines are short or repeated.
    is_single_text_column = len(df.columns) == 1 and not pd.api.types.is_numeric_dtype(df.iloc[:, 0])
    if str(filename).lower().endswith(".txt") or is_single_text_column or _looks_like_unlabeled_text(df, target_column):
        return _run_text_only_augmentation(df, target_column, run_id, filename, output_format)
    state=workflow.invoke({"df":df,"target_column":target_column,"run_id":run_id,
        "filename":filename,"output_format":output_format,
        "progress_callback":progress_callback,"cancel_callback":cancel_callback,
        "max_iterations":max_iterations,"target_accuracy":target_accuracy,
        "stretch_target":stretch_target,"total_time_budget_seconds":total_time_budget_seconds})
    return {**state["final_report"],"dataset_bytes":state["dataset_bytes"],"download_suffix":state["download_suffix"]}
