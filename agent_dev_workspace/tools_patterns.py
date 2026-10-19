import os
import tempfile
from typing import Literal, Dict, Any
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from langchain_core.tools import tool


def make_tools(session) -> list:
    """Factory function returning pattern mining tools bound to session."""

    @tool
    def mine_patterns_tool(
        category_name: str,
        filtering_method: Literal["variance", "tfidf", "term_frequency"] = "tfidf",
        algorithm: Literal["fpgrowth", "topk", "maxfpgrowth"] = "fpgrowth",
        min_sup: int = 1,
        k: int = 10,
    ) -> Dict[str, Any]:
        """Mines frequent word patterns for a specific document category using PAMI.

        Args:
            category_name: The category string to mine (e.g. 'catA').
            filtering_method: Local vocabulary filter ('variance', 'tfidf', 'term_frequency').
            algorithm: PAMI mining algorithm ('fpgrowth', 'topk', 'maxfpgrowth').
            min_sup: Minimum support count threshold for fpgrowth / maxfpgrowth.
            k: Top-K patterns count for topk algorithm.

        Returns:
            Dict containing result_id, category_name, filtering_method, algorithm, patterns list, and status.
        """
        if session.dataframe is None:
            return {
                "status": "error",
                "message": "No dataframe loaded in session. Call load_dataset_tool first.",
            }

        df = session.dataframe
        cat_col = "category_name" if "category_name" in df.columns else ("category" if "category" in df.columns else None)
        if not cat_col:
            return {
                "status": "error",
                "message": "Dataframe missing category column.",
            }

        cat_df = df[df[cat_col].astype(str) == str(category_name)]
        if len(cat_df) == 0:
            return {
                "status": "error",
                "message": f"No documents found for category '{category_name}'.",
            }

        vectorizer = CountVectorizer(stop_words="english")
        texts = cat_df["text"].fillna("").tolist()
        try:
            X_cat = vectorizer.fit_transform(texts)
        except ValueError:
            return {
                "result_id": session.next_result_id("mine_patterns"),
                "category_name": category_name,
                "patterns": [],
                "note": "no terms survived stopword removal",
                "status": "success",
            }

        feature_names = vectorizer.get_feature_names_out()
        if len(feature_names) == 0:
            return {
                "result_id": session.next_result_id("mine_patterns"),
                "category_name": category_name,
                "patterns": [],
                "note": "no terms survived stopword removal",
                "status": "success",
            }

        dense_X = X_cat.toarray()
        if filtering_method == "variance":
            scores = np.var(dense_X, axis=0)
            low_p, high_p = np.percentile(scores, [5, 95])
            kept_mask = (scores > low_p) & (scores < high_p)
        elif filtering_method == "term_frequency":
            scores = np.sum(dense_X, axis=0)
            low_p, high_p = np.percentile(scores, [5, 95])
            kept_mask = (scores > low_p) & (scores < high_p)
        elif filtering_method == "tfidf":
            tfidf_mat = TfidfTransformer().fit_transform(X_cat).toarray()
            scores = np.mean(tfidf_mat, axis=0)
            p20 = np.percentile(scores, 20)
            kept_mask = scores >= p20
        else:
            kept_mask = np.ones(len(feature_names), dtype=bool)

        surviving_terms = feature_names[kept_mask]

        if len(surviving_terms) == 0:
            result_id = session.next_result_id("mine_patterns")
            summary = {
                "result_id": result_id,
                "category_name": category_name,
                "filtering_method": filtering_method,
                "algorithm": algorithm,
                "patterns": [],
                "note": "no terms survived filtering",
                "status": "success",
            }
            session.store_result("mine_patterns_tool", {"category_name": category_name}, summary)
            return summary

        surviving_indices = np.where(kept_mask)[0]
        filtered_X = dense_X[:, surviving_indices]

        df_trans = pd.DataFrame(filtered_X, columns=surviving_terms)

        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as tmp_db:
            tmp_db_path = tmp_db.name

        try:
            from PAMI.extras.convert import DF2DB
            converter = DF2DB.DF2DB(df_trans, "dense")
            converter.convert2TransactionalDatabase(tmp_db_path, ">=", 1)

            if algorithm == "fpgrowth":
                from PAMI.frequentPattern.basic import FPGrowth
                miner = FPGrowth.FPGrowth(tmp_db_path, minSup=min_sup, sep="\t")
            elif algorithm == "topk":
                from PAMI.frequentPattern.topk import FAE
                miner = FAE.FAE(tmp_db_path, k=k, sep="\t")
            elif algorithm == "maxfpgrowth":
                from PAMI.frequentPattern.maximal import MaxFPGrowth
                miner = MaxFPGrowth.MaxFPGrowth(tmp_db_path, minSup=min_sup, sep="\t")
            else:
                from PAMI.frequentPattern.basic import FPGrowth
                miner = FPGrowth.FPGrowth(tmp_db_path, minSup=min_sup, sep="\t")

            miner.startMine()
            patterns_df = miner.getPatternsAsDataFrame()

            patterns_list = []
            if patterns_df is not None and not patterns_df.empty:
                for _, row in patterns_df.iterrows():
                    pattern_str = str(row.iloc[0])
                    pattern_items = [item.strip() for item in pattern_str.split("\t") if item.strip()]
                    sup = int(row.iloc[1])
                    patterns_list.append({
                        "pattern": pattern_items,
                        "support": sup,
                    })

        finally:
            if os.path.exists(tmp_db_path):
                os.remove(tmp_db_path)

        artifact_key = f"patterns_{category_name}_{filtering_method}_{algorithm}"
        session.artifacts[artifact_key] = patterns_list

        result_id = session.next_result_id("mine_patterns")
        summary = {
            "result_id": result_id,
            "category_name": category_name,
            "filtering_method": filtering_method,
            "algorithm": algorithm,
            "patterns_count": len(patterns_list),
            "patterns": patterns_list,
            "status": "success",
        }

        session.store_result("mine_patterns_tool", {"category_name": category_name}, summary)
        return summary

    return [mine_patterns_tool]
