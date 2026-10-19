import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, Optional, Literal
from langchain_core.tools import tool

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap

def make_tools(session) -> list:
    """Factory function returning dimension reduction and label binarization tools bound to session."""

    @tool
    def binarize_labels_tool() -> dict:
        """One-hot encodes document category labels into binary indicator columns.

        Uses session.labels if available; otherwise reads dataframe['category_name'] and calls session.set_labels.
        Generates one binary column per unique category in sorted order. Stores the resulting DataFrame in
        session.artifacts['binarized_labels'] and records full_report in session.results_store.

        Returns:
            Dict containing result_id, n_documents, categories, and shape.
        """
        labels = session.labels

        if labels is None or len(labels) == 0:
            if session.dataframe is not None and "category_name" in session.dataframe.columns:
                raw_labels = session.dataframe["category_name"].values
                if len(raw_labels) > 0:
                    session.set_labels(raw_labels)
                    labels = session.labels

        if labels is None or len(labels) == 0:
            return {"error": "No labels available in session.labels or session.dataframe['category_name']."}

        categories = sorted(list(set(labels)))
        n_documents = len(labels)

        # Create one-hot encoded binary matrix
        binarized_data = {cat: (labels == cat).astype(int) for cat in categories}
        binarized_df = pd.DataFrame(binarized_data)

        session.artifacts["binarized_labels"] = binarized_df

        result_id = session.next_result_id("binarize")
        summary = {
            "result_id": result_id,
            "n_documents": n_documents,
            "categories": categories,
            "shape": [n_documents, len(categories)],
        }

        session.store_result("binarize_labels_tool", {}, summary, full_report=binarized_df)
        return summary

    @tool
    def reduce_dimensions_tool(
        method: Literal["pca", "tsne", "umap"] = "pca",
        n_components: int = 2,
        random_state: Optional[int] = None,
        perplexity: Optional[float] = 30.0,
        n_neighbors: Optional[int] = 15,
    ) -> dict:
        """Reduces document-term matrix dimensions to 2D coordinates and plots a scatter figure.

        Supports PCA, t-SNE, and UMAP reduction methods. Generates a 2D scatter plot colored by document category,
        assigns the figure object to session.pending_figure, and stores reduced coordinates in session.artifacts.

        Args:
            method: Reduction algorithm ("pca", "tsne", or "umap", default "pca").
            n_components: Number of target dimensions (default 2).
            random_state: Seed for reproducible output.
            perplexity: Perplexity parameter for t-SNE.
            n_neighbors: Number of neighbors parameter for UMAP.

        Returns:
            Dict containing result_id, method, coordinates, and explained_variance_ratio (PCA only).
        """
        if session.feature_matrix is None:
            return {"error": "No feature matrix available in session. Call build_dtm_tool first."}

        X = session.feature_matrix.toarray() if hasattr(session.feature_matrix, "toarray") else np.asarray(session.feature_matrix)
        explained_variance_ratio = None

        if method == "pca":
            model = PCA(n_components=n_components, random_state=random_state)
            coords = model.fit_transform(X)
            explained_variance_ratio = [float(round(v, 4)) for v in model.explained_variance_ratio_]
        elif method == "tsne":
            p = perplexity if perplexity is not None else 30.0
            model = TSNE(n_components=n_components, perplexity=p, random_state=random_state)
            coords = model.fit_transform(X)
        elif method == "umap":
            nn = n_neighbors if n_neighbors is not None else 15
            model = umap.UMAP(n_components=n_components, n_neighbors=nn, random_state=random_state)
            coords = model.fit_transform(X)
        else:
            return {"error": f"Unsupported reduction method: '{method}'."}

        coords_list = [[float(round(c, 4)) for c in row] for row in coords]
        session.artifacts["reduced_coordinates"] = coords_list

        # Build mandatory 2D scatter plot
        fig, ax = plt.subplots(figsize=(7, 5))
        plot_df = pd.DataFrame({
            "Dim 1": coords[:, 0],
            "Dim 2": coords[:, 1]
        })

        if session.dataframe is not None and "category_name" in session.dataframe.columns:
            plot_df["Category"] = session.dataframe["category_name"].values
            sns.scatterplot(data=plot_df, x="Dim 1", y="Dim 2", hue="Category", ax=ax, s=70)
        elif session.labels is not None:
            plot_df["Category"] = session.labels
            sns.scatterplot(data=plot_df, x="Dim 1", y="Dim 2", hue="Category", ax=ax, s=70)
        else:
            sns.scatterplot(data=plot_df, x="Dim 1", y="Dim 2", ax=ax, s=70)

        ax.set_title(f"Document Dimensionality Reduction ({method.upper()})")
        plt.tight_layout()

        session.pending_figure = fig
        plt.close(fig)

        result_id = session.next_result_id("reduce_dim")
        summary = {
            "result_id": result_id,
            "method": method,
            "coordinates": coords_list,
            "explained_variance_ratio": explained_variance_ratio,
        }

        session.store_result(
            "reduce_dimensions_tool",
            {
                "method": method,
                "n_components": n_components,
                "random_state": random_state,
                "perplexity": perplexity,
                "n_neighbors": n_neighbors,
            },
            summary,
        )
        return summary

    return [binarize_labels_tool, reduce_dimensions_tool]
