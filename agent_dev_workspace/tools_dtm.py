from typing import List, Optional, Dict, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
from langchain_core.tools import tool


def make_tools(session) -> list:
    """Factory function returning DTM construction and analysis tools bound to session."""

    @tool
    def build_dtm_tool(
        ngram_range: List[int] = [1, 1],
        min_df: int = 1,
        max_df: float = 1.0,
        stop_words: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Builds the Document-Term Matrix (DTM) into session.feature_matrix and session.feature_names.

        Args:
            ngram_range: List of two ints [min_n, max_n] for n-gram bounds (default [1, 1]).
            min_df: Minimum document frequency threshold for terms.
            max_df: Maximum document frequency proportion or count threshold.
            stop_words: Stopwords strategy ('english' or None).

        Returns:
            Dict summary with result_id, n_documents, n_terms, non_zero, total_elements, and sparsity_pct.
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

        tuple_ngram = (ngram_range[0], ngram_range[1]) if len(ngram_range) >= 2 else (1, 1)

        vectorizer = CountVectorizer(
            ngram_range=tuple_ngram,
            min_df=min_df,
            max_df=max_df,
            stop_words=stop_words,
        )

        texts = df["text"].fillna("").tolist()
        feature_matrix = vectorizer.fit_transform(texts)
        feature_names = vectorizer.get_feature_names_out().tolist()

        session.feature_matrix = feature_matrix
        session.feature_names = feature_names
        session.artifacts["count_vectorizer"] = vectorizer

        if "category_name" in df.columns:
            session.set_labels(df["category_name"].values)
        elif "category" in df.columns:
            session.set_labels(df["category"].values)

        n_docs, n_terms = feature_matrix.shape
        non_zero = int(feature_matrix.nnz)
        total_elements = n_docs * n_terms
        sparsity_pct = float(round(100.0 * (1.0 - non_zero / total_elements), 4)) if total_elements > 0 else 0.0

        result_id = session.next_result_id("build_dtm")
        summary = {
            "result_id": result_id,
            "n_documents": n_docs,
            "n_terms": n_terms,
            "non_zero": non_zero,
            "total_elements": total_elements,
            "sparsity_pct": sparsity_pct,
            "status": "success",
        }

        session.store_result(
            "build_dtm_tool",
            {"ngram_range": ngram_range, "min_df": min_df, "max_df": max_df},
            summary,
        )
        return summary

    @tool
    def term_frequency_tool(top_n: int = 20) -> Dict[str, Any]:
        """Aggregates term frequencies across all documents in the Document-Term Matrix.

        Args:
            top_n: Number of top frequent terms to return in the summary.

        Returns:
            Dict containing result_id, total_terms, total_occurrences, and top_terms.
        """
        if session.feature_matrix is None or session.feature_names is None:
            return {
                "status": "error",
                "message": "Document-Term Matrix not built. Call build_dtm_tool first.",
            }

        frequencies = np.asarray(session.feature_matrix.sum(axis=0)).flatten()
        term_freq_series = pd.Series(frequencies, index=session.feature_names)

        sorted_term_freq = term_freq_series.sort_values(ascending=False)

        full_report_df = pd.DataFrame({
            "term": sorted_term_freq.index,
            "frequency": sorted_term_freq.values,
        })

        top_terms_dict = {str(k): int(v) for k, v in sorted_term_freq.head(top_n).to_dict().items()}

        result_id = session.next_result_id("term_freq")
        summary = {
            "result_id": result_id,
            "total_terms": len(session.feature_names),
            "total_occurrences": int(frequencies.sum()),
            "top_terms": top_terms_dict,
            "status": "success",
        }

        session.store_result("term_frequency_tool", {"top_n": top_n}, summary, full_report=full_report_df)
        return summary

    @tool
    def dtm_heatmap_tool(n_terms: int = 20, n_documents: int = 20) -> Dict[str, Any]:
        """Generates a heatmap visualization for a positional slice of the Document-Term Matrix.

        Args:
            n_terms: Number of vocabulary terms (columns) to include (default: 20).
            n_documents: Number of documents (rows) to include (default: 20).

        Returns:
            Dict summary with result_id, terms, document_indices, and matrix_slice.
        """
        if session.feature_matrix is None or session.feature_names is None:
            return {
                "status": "error",
                "message": "Document-Term Matrix not built. Call build_dtm_tool first.",
            }

        max_docs, max_terms = session.feature_matrix.shape
        sliced_docs = min(n_documents, max_docs)
        sliced_terms = min(n_terms, max_terms)

        matrix_slice_sparse = session.feature_matrix[0:sliced_docs, 0:sliced_terms]
        matrix_dense = matrix_slice_sparse.toarray()
        terms_included = session.feature_names[0:sliced_terms]
        doc_indices = list(range(sliced_docs))

        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(
            matrix_dense,
            annot=True,
            fmt="d",
            cmap="YlGnBu",
            xticklabels=terms_included,
            yticklabels=doc_indices,
            ax=ax,
        )
        ax.set_title(f"DTM Heatmap ({sliced_docs} Docs x {sliced_terms} Terms)")
        ax.set_xlabel("Terms")
        ax.set_ylabel("Document Index")
        plt.tight_layout()

        session.pending_figure = fig
        plt.close(fig)

        result_id = session.next_result_id("dtm_heatmap")
        summary = {
            "result_id": result_id,
            "n_documents_shown": sliced_docs,
            "n_terms_shown": sliced_terms,
            "terms": terms_included,
            "matrix_slice": matrix_dense.tolist(),
            "status": "success",
        }

        session.store_result("dtm_heatmap_tool", {"n_terms": n_terms, "n_documents": n_documents}, summary)
        return summary

    return [build_dtm_tool, term_frequency_tool, dtm_heatmap_tool]
