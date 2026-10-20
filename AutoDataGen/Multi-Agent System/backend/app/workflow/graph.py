import io
from typing import TypedDict, Any
import pandas as pd
from langgraph.graph import StateGraph, START, END
from app.agents.analysis_agent import analyze_dataframe
from app.agents.strategy_agent import choose_strategy
from app.agents.execution_agent import augment_training_data
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
def run_workflow(df, target_column, run_id, filename, output_format="original", *,
                 progress_callback=None, cancel_callback=None, max_iterations=DEFAULT_MAX_ITERATIONS,
                 target_accuracy=DEFAULT_TARGET_ACCURACY, stretch_target=DEFAULT_STRETCH_TARGET, total_time_budget_seconds=DEFAULT_TOTAL_TIME_BUDGET_SECONDS):
    state=workflow.invoke({"df":df,"target_column":target_column,"run_id":run_id,
        "filename":filename,"output_format":output_format,
        "progress_callback":progress_callback,"cancel_callback":cancel_callback,
        "max_iterations":max_iterations,"target_accuracy":target_accuracy,
        "stretch_target":stretch_target,"total_time_budget_seconds":total_time_budget_seconds})
    return {**state["final_report"],"dataset_bytes":state["dataset_bytes"],"download_suffix":state["download_suffix"]}
