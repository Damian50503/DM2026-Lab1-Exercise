from typing import Dict, Any
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from langchain_core.tools import tool


def make_tools(session) -> list:
    """Factory function returning feature filtering tools bound to session."""

    @tool
    def variance_filter_tool(threshold: float = 0.15) -> Dict[str, Any]:
        """Flags near-constant terms by variance for removal based on a variance threshold.

        Args:
            threshold: Minimum variance threshold to keep a feature (default: 0.15).

        Returns:
            Dict containing result_id, threshold, initial_terms, kept_terms_count,
            removed_terms_count, and kept_terms.
        """
        if session.feature_matrix is None or session.feature_names is None:
            return {
                "status": "error",
                "message": "Document-Term Matrix not built. Call build_dtm_tool first.",
            }

        X = session.feature_matrix
        feature_names = np.array(session.feature_names)

        mean_sq = np.asarray(X.power(2).mean(axis=0)).flatten()
        sq_mean = np.square(np.asarray(X.mean(axis=0)).flatten())
        variances = mean_sq - sq_mean

        report_df = pd.DataFrame({
            "term": feature_names,
            "variance": variances,
        }).sort_values(by="variance", ascending=False)

        kept_mask = variances >= threshold
        kept_terms = feature_names[kept_mask].tolist()
        removed_terms = feature_names[~kept_mask].tolist()

        result_id = session.next_result_id("variance_filter")
        summary = {
            "result_id": result_id,
            "threshold": threshold,
            "initial_terms": len(feature_names),
            "kept_terms_count": len(kept_terms),
            "removed_terms_count": len(removed_terms),
            "kept_terms": kept_terms,
            "status": "success",
        }

        session.store_result(
            "variance_filter_tool",
            {"threshold": threshold},
            summary,
            full_report=report_df,
        )
        return summary

    @tool
    def pearson_filter_tool(target_class: str) -> Dict[str, Any]:
        """Calculates Pearson linear correlation between DTM term counts and a binary target class.

        Args:
            target_class: The category label string to calculate correlation against (one-vs-rest).

        Returns:
            Dict containing result_id, target_class, and correlations dictionary mapping terms to r values.
        """
        if session.feature_matrix is None or session.feature_names is None:
            return {
                "status": "error",
                "message": "Document-Term Matrix not built. Call build_dtm_tool first.",
            }

        if session.labels is None:
            return {
                "status": "error",
                "message": "Category labels not set in session. Ensure dataset has category labels.",
            }

        y = (np.array(session.labels) == target_class).astype(float)
        X = session.feature_matrix.toarray()

        correlations = {}
        for idx, term in enumerate(session.feature_names):
            x_col = X[:, idx]
            if np.std(x_col) == 0 or np.std(y) == 0:
                r = 0.0
            else:
                r_val, _ = pearsonr(x_col, y)
                r = 0.0 if np.isnan(r_val) else float(round(r_val, 4))
            correlations[term] = r

        report_df = pd.DataFrame({
            "term": list(correlations.keys()),
            "pearson_r": list(correlations.values()),
        }).sort_values(by="pearson_r", ascending=False)

        result_id = session.next_result_id("pearson_filter")
        summary = {
            "result_id": result_id,
            "target_class": target_class,
            "correlations": correlations,
            "status": "success",
        }

        session.store_result(
            "pearson_filter_tool",
            {"target_class": target_class},
            summary,
            full_report=report_df,
        )
        return summary

    @tool
    def spearman_filter_tool(target_class: str) -> Dict[str, Any]:
        """Calculates Spearman rank correlation between DTM term counts and a binary target class.

        Args:
            target_class: The category label string to calculate rank correlation against (one-vs-rest).

        Returns:
            Dict containing result_id, target_class, and correlations dictionary mapping terms to r values.
        """
        if session.feature_matrix is None or session.feature_names is None:
            return {
                "status": "error",
                "message": "Document-Term Matrix not built. Call build_dtm_tool first.",
            }

        if session.labels is None:
            return {
                "status": "error",
                "message": "Category labels not set in session. Ensure dataset has category labels.",
            }

        y = (np.array(session.labels) == target_class).astype(float)
        X = session.feature_matrix.toarray()

        correlations = {}
        for idx, term in enumerate(session.feature_names):
            x_col = X[:, idx]
            if np.std(x_col) == 0 or np.std(y) == 0:
                r = 0.0
            else:
                r_val, _ = spearmanr(x_col, y)
                r = 0.0 if np.isnan(r_val) else float(round(r_val, 4))
            correlations[term] = r

        report_df = pd.DataFrame({
            "term": list(correlations.keys()),
            "spearman_r": list(correlations.values()),
        }).sort_values(by="spearman_r", ascending=False)

        result_id = session.next_result_id("spearman_filter")
        summary = {
            "result_id": result_id,
            "target_class": target_class,
            "correlations": correlations,
            "status": "success",
        }

        session.store_result(
            "spearman_filter_tool",
            {"target_class": target_class},
            summary,
            full_report=report_df,
        )
        return summary

    return [variance_filter_tool, pearson_filter_tool, spearman_filter_tool]
