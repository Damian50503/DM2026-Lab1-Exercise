from typing import Optional, Dict, Any
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from langchain_core.tools import tool


def make_tools(session) -> list:
    """Factory function returning tools for data checking, sampling, and inspection bound to session."""

    @tool
    def tokenize_tool() -> Dict[str, Any]:
        """Tokenize document text in session.dataframe into lowercase unigrams.

        Uses nltk.word_tokenize on lowercased text strings, falling back to basic
        whitespace/punctuation splitting if an NLTK LookupError occurs (e.g., missing punkt data).
        Stores unigram token lists in session.dataframe['unigrams'] and session.artifacts['unigrams'].

        Returns:
            Dict containing result_id, n_documents, and average_tokens_per_doc.
        """
        if session.dataframe is None:
            return {
                "status": "error",
                "message": "No dataframe loaded in session. Call load_dataset_tool first.",
            }

        df = session.dataframe
        if "text" not in df.columns:
            return {
                "status": "error",
                "message": "Dataframe missing required 'text' column.",
            }

        def tokenize_text(text: str) -> list:
            if not isinstance(text, str) or not text.strip():
                return []
            lowercased = text.lower()
            try:
                import nltk
                return nltk.word_tokenize(lowercased)
            except LookupError:
                import re
                return re.findall(r"\b\w+\b", lowercased)

        df["unigrams"] = df["text"].apply(tokenize_text)
        session.artifacts["unigrams"] = df["unigrams"].tolist()

        total_tokens = sum(len(tokens) for tokens in df["unigrams"])
        n_docs = len(df)
        avg_tokens = float(round(total_tokens / n_docs, 4)) if n_docs > 0 else 0.0

        result_id = session.next_result_id("tokenize")
        summary = {
            "result_id": result_id,
            "n_documents": n_docs,
            "average_tokens_per_doc": avg_tokens,
            "status": "success",
        }

        session.store_result("tokenize_tool", {}, summary)
        return summary

    @tool
    def inspect_data_tool(
        n_rows: int = 5,
        max_text_length: int = 200,
    ) -> dict:
        """Peeks at top rows of session.dataframe for data structure and content inspection.

        Args:
            n_rows: Number of top rows to inspect (must be >= 1, default 5).
            max_text_length: Max character length for previewing the 'text' column (must be >= 1, default 200).

        Returns:
            Dict containing result_id, total_rows, columns, and rows.
        """
        if session.dataframe is None:
            return {"error": "No dataframe loaded in session. Call load_dataset_tool first."}

        if n_rows < 1:
            return {"error": f"n_rows must be >= 1, got {n_rows}."}

        if max_text_length < 1:
            return {"error": f"max_text_length must be >= 1, got {max_text_length}."}

        df = session.dataframe
        total_rows = len(df)
        columns = df.columns.tolist()

        def clean_val(val):
            if isinstance(val, dict):
                return {str(k): clean_val(v) for k, v in val.items()}
            if isinstance(val, (list, tuple)):
                return [clean_val(x) for x in val]
            if isinstance(val, np.ndarray):
                return [clean_val(x) for x in val.tolist()]
            if isinstance(val, (pd.Timestamp, pd.Timedelta)):
                return str(val)
            if isinstance(val, np.generic):
                return val.item()
            if pd.isna(val):
                return None
            return val

        if total_rows == 0:
            result_id = session.next_result_id("inspect")
            summary = {
                "result_id": result_id,
                "total_rows": 0,
                "columns": columns,
                "rows": [],
            }
            session.store_result(
                "inspect_data_tool",
                {"n_rows": n_rows, "max_text_length": max_text_length},
                summary,
                full_report=[],
            )
            return summary

        subset = df.head(n_rows)
        raw_records = subset.to_dict(orient="records")

        full_report = [
            {str(k): clean_val(v) for k, v in record.items()}
            for record in raw_records
        ]

        preview_rows = []
        for record in full_report:
            row_copy = dict(record)
            if "text" in row_copy and isinstance(row_copy["text"], str) and len(row_copy["text"]) > max_text_length:
                row_copy["text"] = row_copy["text"][:max_text_length] + "..."
            preview_rows.append(row_copy)

        result_id = session.next_result_id("inspect")
        summary = {
            "result_id": result_id,
            "total_rows": total_rows,
            "columns": columns,
            "rows": preview_rows,
        }

        session.store_result(
            "inspect_data_tool",
            {"n_rows": n_rows, "max_text_length": max_text_length},
            summary,
            full_report=full_report,
        )
        return summary

    @tool
    def check_missing_tool() -> Dict[str, Any]:
        """Check for missing or empty text entries in session.dataframe."""
        if session.dataframe is None:
            return {
                "status": "error",
                "message": "No dataframe loaded in session. Call load_dataset_tool first.",
            }

        df = session.dataframe
        if "text" not in df.columns:
            return {
                "status": "error",
                "message": "Dataframe missing required 'text' column.",
            }

        total_rows = len(df)
        missing_mask = df["text"].isna() | (df["text"].astype(str).str.strip() == "")
        missing_count = int(missing_mask.sum())
        missing_pct = float(round(100.0 * missing_count / total_rows, 4)) if total_rows > 0 else 0.0

        result_id = session.next_result_id("check_missing")
        summary = {
            "result_id": result_id,
            "total_rows": total_rows,
            "missing_count": missing_count,
            "missing_pct": missing_pct,
            "status": "success",
        }

        session.store_result("check_missing_tool", {}, summary)
        return summary

    @tool
    def check_duplicates_tool(drop: bool = False) -> Dict[str, Any]:
        """Check for and optionally drop duplicate rows in session.dataframe."""
        if session.dataframe is None:
            return {
                "status": "error",
                "message": "No dataframe loaded in session. Call load_dataset_tool first.",
            }

        df = session.dataframe
        initial_rows = len(df)

        duplicated_mask = df.duplicated(keep=False)
        duplicate_count = int(duplicated_mask.sum())

        if drop and duplicate_count > 0:
            df.drop_duplicates(keep=False, inplace=True)
            df.reset_index(drop=True, inplace=True)

        final_rows = len(session.dataframe)

        result_id = session.next_result_id("check_duplicates")
        summary = {
            "result_id": result_id,
            "initial_rows": initial_rows,
            "duplicate_count": duplicate_count,
            "final_rows": final_rows,
            "dropped": drop,
            "status": "success",
        }

        session.store_result("check_duplicates_tool", {"drop": drop}, summary)
        return summary

    @tool
    def list_files_tool(subdirectory: str = ".") -> Dict[str, Any]:
        """List files and subdirectories in a workspace folder for dataset discovery."""
        base_dir = Path.cwd().resolve()
        target_path = (base_dir / subdirectory).resolve()

        if not str(target_path).startswith(str(base_dir)):
            return {
                "status": "error",
                "message": "Access restricted: Cannot inspect path outside workspace root.",
            }

        if not target_path.exists():
            return {
                "status": "error",
                "message": f"Subdirectory does not exist: '{subdirectory}'",
            }

        if not target_path.is_dir():
            return {
                "status": "error",
                "message": f"Target path is a file, not a directory: '{subdirectory}'",
            }

        directories = []
        files = []

        try:
            for entry in target_path.iterdir():
                if entry.name.startswith("."):
                    continue
                if entry.is_dir():
                    directories.append(entry.name)
                elif entry.is_file():
                    files.append(entry.name)
        except Exception as e:
            return {
                "status": "error",
                "message": f"Error reading directory '{subdirectory}': {str(e)}",
            }

        directories.sort()
        files.sort()

        result_id = session.next_result_id("list_files")
        summary = {
            "result_id": result_id,
            "subdirectory": subdirectory,
            "directories": directories,
            "files": files,
            "status": "success",
        }

        session.store_result("list_files_tool", {"subdirectory": subdirectory}, summary)
        return summary

    @tool
    def sample_data_tool(
        n: Optional[int] = None,
        frac: Optional[float] = None,
        random_state: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Subsamples the working DataFrame session.dataframe.

        Args:
            n: Number of items from axis to return. Cannot be used with frac.
            frac: Fraction of axis items to return. Cannot be used with n.
            random_state: Seed for the random number generator for reproducibility.

        Returns:
            Dict summary with result_id, n_samples, original_n_samples, and sampled_indices.
        """
        if session.dataframe is None:
            return {
                "status": "error",
                "message": "No dataframe loaded in session. Call load_dataset_tool first.",
            }

        original_count = len(session.dataframe)

        if n is None and frac is None:
            n = min(100, original_count)

        sampled_df = session.dataframe.sample(
            n=n, frac=frac, random_state=random_state
        )

        session.dataframe = sampled_df

        result_id = session.next_result_id("sample")
        summary = {
            "result_id": result_id,
            "n_samples": len(sampled_df),
            "original_n_samples": original_count,
            "sampled_indices": sampled_df.index.tolist(),
        }

        session.store_result(
            "sample_data_tool",
            {"n": n, "frac": frac, "random_state": random_state},
            summary,
        )
        return summary

    @tool
    def describe_data_tool() -> Dict[str, Any]:
        """Calculates text length summary statistics overall and per category, and generates a boxplot.

        Computes character length (text_length) as df['text'].apply(len). Returns .describe()
        statistics overall and grouped by category, and assigns a Seaborn boxplot figure to
        session.pending_figure.

        Returns:
            Dict containing result_id, overall_stats, per_category_stats, and status.
        """
        if session.dataframe is None:
            return {
                "status": "error",
                "message": "No dataframe loaded in session. Call load_dataset_tool first.",
            }

        df = session.dataframe
        if "text" not in df.columns:
            return {
                "status": "error",
                "message": "Dataframe missing required 'text' column.",
            }

        df["text_length"] = df["text"].fillna("").apply(len)

        overall_desc = df["text_length"].describe().to_dict()

        cat_col = "category_name" if "category_name" in df.columns else ("category" if "category" in df.columns else None)

        per_category_desc = {}
        if cat_col:
            grouped = df.groupby(cat_col)["text_length"].describe()
            for cat, row in grouped.iterrows():
                per_category_desc[str(cat)] = row.to_dict()

        fig, ax = plt.subplots(figsize=(7, 4.5))
        if cat_col:
            sns.boxplot(data=df, x=cat_col, y="text_length", ax=ax)
            ax.set_xlabel(cat_col.replace("_", " ").title())
        else:
            sns.boxplot(y=df["text_length"], ax=ax)
            ax.set_xlabel("Dataset")

        ax.set_ylabel("Text Length (Characters)")
        ax.set_title("Document Text Length Distribution by Category")
        plt.tight_layout()

        session.pending_figure = fig
        plt.close(fig)

        result_id = session.next_result_id("describe")
        summary = {
            "result_id": result_id,
            "overall_stats": overall_desc,
            "per_category_stats": per_category_desc,
            "status": "success",
        }

        session.store_result("describe_data_tool", {}, summary)
        return summary

    return [
        tokenize_tool,
        inspect_data_tool,
        check_missing_tool,
        check_duplicates_tool,
        list_files_tool,
        sample_data_tool,
        describe_data_tool,
    ]
