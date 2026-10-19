from typing import Dict, Any, List
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from langchain_core.tools import tool


def make_tools(session) -> list:
    """Factory function returning exploration and analysis tools bound to session."""

    @tool
    def cosine_similarity_tool(
        doc_idx1: int = 0,
        doc_idx2: int = 1,
    ) -> Dict[str, Any]:
        """Computes cosine similarity between two documents in the Document-Term Matrix.

        Args:
            doc_idx1: Zero-based index of the first document (default: 0).
            doc_idx2: Zero-based index of the second document (default: 1).

        Returns:
            Dict containing result_id, doc_idx1, doc_idx2, similarity score, and status.
        """
        if session.feature_matrix is None or session.feature_names is None:
            return {
                "status": "error",
                "message": "Document-Term Matrix not built. Call build_dtm_tool first.",
            }

        n_docs = session.feature_matrix.shape[0]
        if not (0 <= doc_idx1 < n_docs):
            return {
                "status": "error",
                "message": f"doc_idx1 ({doc_idx1}) is out of bounds for matrix with {n_docs} documents.",
            }

        if not (0 <= doc_idx2 < n_docs):
            return {
                "status": "error",
                "message": f"doc_idx2 ({doc_idx2}) is out of bounds for matrix with {n_docs} documents.",
            }

        row1 = session.feature_matrix[doc_idx1 : doc_idx1 + 1]
        row2 = session.feature_matrix[doc_idx2 : doc_idx2 + 1]

        # Calculate cosine similarity using scikit-learn
        sim_matrix = cosine_similarity(row1, row2)
        sim_val = float(sim_matrix[0, 0])

        # Handle zero-vector/empty text case safely (NaN check)
        if np.isnan(sim_val):
            sim_val = 0.0

        sim_val = float(round(sim_val, 4))

        result_id = session.next_result_id("cosine_sim")
        summary = {
            "result_id": result_id,
            "doc_idx1": doc_idx1,
            "doc_idx2": doc_idx2,
            "similarity": sim_val,
            "status": "success",
        }

        session.store_result(
            "cosine_similarity_tool",
            {"doc_idx1": doc_idx1, "doc_idx2": doc_idx2},
            summary,
        )
        return summary

    @tool
    def feature_correlation_matrix_tool(top_n: int = 20) -> Dict[str, Any]:
        """Computes Pearson correlation matrix across top terms by variance and generates a heatmap.

        Args:
            top_n: Number of top variance terms to include in the matrix (default: 20).

        Returns:
            Dict containing result_id, terms, correlation_matrix (nested list), and status.
        """
        if session.feature_matrix is None or session.feature_names is None:
            return {
                "status": "error",
                "message": "Document-Term Matrix not built. Call build_dtm_tool first.",
            }

        X = session.feature_matrix.toarray() if hasattr(session.feature_matrix, "toarray") else np.asarray(session.feature_matrix)
        feature_names = np.array(session.feature_names)
        n_docs, n_terms = X.shape

        if n_terms == 0:
            return {
                "status": "error",
                "message": "Document-Term Matrix has 0 terms.",
            }

        # Calculate term variances
        variances = np.var(X, axis=0)

        # Select top N terms by variance
        actual_top_n = min(top_n, n_terms)
        top_indices = np.argsort(variances)[::-1][:actual_top_n]

        top_terms = feature_names[top_indices].tolist()
        top_X = X[:, top_indices]

        # Pearson correlation matrix (np.corrcoef expects variables as rows, so transpose top_X)
        corr_matrix = np.corrcoef(top_X, rowvar=False)

        # Handle potential NaNs if a column has zero variance
        corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)
        corr_matrix_rounded = np.round(corr_matrix, 4)

        # Generate Heatmap figure
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(
            corr_matrix_rounded,
            annot=True,
            fmt=".4f",
            cmap="coolwarm",
            xticklabels=top_terms,
            yticklabels=top_terms,
            ax=ax,
            vmin=-1.0,
            vmax=1.0,
        )
        ax.set_title(f"Feature Correlation Matrix (Top {actual_top_n} Terms by Variance)")
        plt.tight_layout()

        session.pending_figure = fig
        plt.close(fig)

        corr_list = corr_matrix_rounded.tolist()

        result_id = session.next_result_id("feature_corr")
        summary = {
            "result_id": result_id,
            "terms": top_terms,
            "correlation_matrix": corr_list,
            "status": "success",
        }

        session.store_result(
            "feature_correlation_matrix_tool",
            {"top_n": top_n},
            summary,
            full_report={"terms": top_terms, "correlation_matrix": corr_list},
        )
        return summary

    return [cosine_similarity_tool, feature_correlation_matrix_tool]
