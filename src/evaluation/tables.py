"""
Deterministic, single-source results tables.
Notebook MUST NOT do aggregation.
"""

from typing import Dict, List
import pandas as pd
from pathlib import Path
import numpy as np


# ---------------------------------------------------------------------
# 1) CANONICAL FLATTEN
# ---------------------------------------------------------------------

def flatten_all_results(
    all_results: Dict[str, Dict[str, Dict[str, dict]]],
    models: List[str],
    classifiers: List[str],
    tasks: List[str],
) -> pd.DataFrame:
    """
    Canonical flat results table.
    Columns:
      model | classifier | task | macro_f1 | weighted_f1 | accuracy
    """

    rows = []

    for model in models:
        if model not in all_results:
            continue

        for task in tasks:
            if task not in all_results[model]:
                continue

            for clf in classifiers:
                if clf not in all_results[model][task]:
                    continue

                result = all_results[model][task][clf]
                if "metrics" not in result:
                    continue

                m = result["metrics"]

                rows.append({
                    "model": model,
                    "classifier": clf,
                    "task": task,
                    "macro_f1": m.get("macro_f1"),
                    "weighted_f1": m.get("weighted_f1"),
                    "accuracy": m.get("accuracy"),
                })

    df = pd.DataFrame(rows)

    # HARD FAIL if schema broken
    required = {"model", "classifier", "task", "macro_f1"}
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f"Missing required columns: {missing}")

    return df


# ---------------------------------------------------------------------
# 2) MODEL-WISE: Classifier × Tasks
# ---------------------------------------------------------------------

def build_model_wise_tables(
    df: pd.DataFrame,
    metric: str = "macro_f1",
) -> Dict[str, pd.DataFrame]:
    """
    Returns:
      dict[model_name] -> pivot(Classifier × Tasks)
    """

    tables = {}

    for model in sorted(df["model"].unique()):
        sub = df[df["model"] == model]

        pivot = sub.pivot_table(
            index="classifier",
            columns="task",
            values=metric,
        )

        pivot = pivot.dropna(how="all").dropna(axis=1, how="all")

        if not pivot.empty:
            tables[model] = pivot

    return tables


# ---------------------------------------------------------------------
# 3) CLASSIFIER-WISE: Model × Tasks
# ---------------------------------------------------------------------

def build_classifier_wise_tables(
    df: pd.DataFrame,
    metric: str = "macro_f1",
) -> Dict[str, pd.DataFrame]:
    """
    Returns:
      dict[classifier_name] -> pivot(Model × Tasks)
    """

    tables = {}

    for clf in sorted(df["classifier"].unique()):
        sub = df[df["classifier"] == clf]

        pivot = sub.pivot_table(
            index="model",
            columns="task",
            values=metric,
        )

        pivot = pivot.dropna(how="all").dropna(axis=1, how="all")

        if not pivot.empty:
            tables[clf] = pivot

    return tables


# ---------------------------------------------------------------------
# 4) PAPER-READY STYLING (MINIMAL, SAFE)
# ---------------------------------------------------------------------

def style_table_paper(
    df: pd.DataFrame,
    precision: int = 4,
):
    """
    Column-wise best highlighted (bold).
    No background tricks. Deterministic.
    """

    numeric_cols = df.select_dtypes(include=[np.number]).columns

    styled = df.style

    # format numbers
    for c in numeric_cols:
        styled = styled.format({c: f"{{:.{precision}f}}"})

    # column-wise best
    def highlight_best(col):
        if not np.issubdtype(col.dtype, np.number):
            return [""] * len(col)
        max_val = col.max()
        return [
            "font-weight: bold" if pd.notna(v) and abs(v - max_val) < 1e-9 else ""
            for v in col
        ]

    styled = styled.apply(highlight_best, axis=0)

    styled = styled.set_table_styles([
        {"selector": "th", "props": [("font-weight", "bold")]},
        {"selector": "td", "props": [("padding", "6px")]},
    ])

    return styled


# ---------------------------------------------------------------------
# 5) EXPORT
# ---------------------------------------------------------------------

def export_tables(
    styled,
    out_path: Path,
    name: str,
):
    out_path.mkdir(parents=True, exist_ok=True)

    styled.to_excel(out_path / f"{name}.xlsx", engine="openpyxl")
    styled.to_html(out_path / f"{name}.html")

    # raw df exports
    styled.data.to_csv(out_path / f"{name}.csv")
    styled.data.to_markdown(out_path / f"{name}.md")
