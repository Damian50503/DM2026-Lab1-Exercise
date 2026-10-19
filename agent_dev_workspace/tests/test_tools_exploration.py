import pytest
import numpy as np
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_exploration import make_tools


def setup_fixture_session():
    """Helper to initialize session with sample_fixture.csv dataset and fitted DTM."""
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label",
    })

    from sklearn.feature_extraction.text import CountVectorizer
    texts = session.dataframe["text"].fillna("").tolist()
    vectorizer = CountVectorizer()
    session.feature_matrix = vectorizer.fit_transform(texts)
    session.feature_names = vectorizer.get_feature_names_out().tolist()
    session.set_labels(session.dataframe["category_name"].values)
    return session


def test_cosine_similarity_tool_known_answer_16():
    """Tests cosine_similarity_tool against Known Answer #16 in TEST_FIXTURE.md."""
    session = setup_fixture_session()
    tools = make_tools(session)
    cosine_tool = next(t for t in tools if t.name == "cosine_similarity_tool")

    # Doc 0 vs 1 (both catA, near-identical text): 0.9847
    res01 = cosine_tool.invoke({"doc_idx1": 0, "doc_idx2": 1})
    assert res01["status"] == "success"
    assert np.isclose(res01["similarity"], 0.9847, atol=1e-3)

    # Doc 0 vs 3 (catA vs catB, share only 'always'): 0.0909
    res03 = cosine_tool.invoke({"doc_idx1": 0, "doc_idx2": 3})
    assert res03["status"] == "success"
    assert np.isclose(res03["similarity"], 0.0909, atol=1e-3)

    # Doc 3 vs 4 (both catB, near-identical text): 0.9847
    res34 = cosine_tool.invoke({"doc_idx1": 3, "doc_idx2": 4})
    assert res34["status"] == "success"
    assert np.isclose(res34["similarity"], 0.9847, atol=1e-3)

    # Doc 0 vs 7 (catA vs empty text row): 0.0
    res07 = cosine_tool.invoke({"doc_idx1": 0, "doc_idx2": 7})
    assert res07["status"] == "success"
    assert np.isclose(res07["similarity"], 0.0, atol=1e-3)


def test_feature_correlation_matrix_tool_known_answer_19():
    """Tests feature_correlation_matrix_tool against Known Answer #19 in TEST_FIXTURE.md."""
    session = setup_fixture_session()
    tools = make_tools(session)
    corr_tool = next(t for t in tools if t.name == "feature_correlation_matrix_tool")

    res = corr_tool.invoke({"top_n": 20})

    assert res["status"] == "success"
    terms = res["terms"]
    # Check terms sorted by variance descending: alpha, gamma, delta, beta, always
    assert terms == ["alpha", "gamma", "delta", "beta", "always"]

    corr_mat = np.array(res["correlation_matrix"])

    # Diagonal elements must be 1.0
    for i in range(len(terms)):
        assert np.isclose(corr_mat[i, i], 1.0000, atol=1e-3)

    # Check known pairwise correlations from TEST_FIXTURE.md #19
    alpha_idx = terms.index("alpha")
    beta_idx = terms.index("beta")
    gamma_idx = terms.index("gamma")
    delta_idx = terms.index("delta")
    always_idx = terms.index("always")

    assert np.isclose(corr_mat[alpha_idx, beta_idx], 0.8885, atol=1e-3)
    assert np.isclose(corr_mat[gamma_idx, delta_idx], 0.8307, atol=1e-3)
    assert np.isclose(corr_mat[alpha_idx, delta_idx], -0.6882, atol=1e-3)
    assert np.isclose(corr_mat[always_idx, delta_idx], 0.3780, atol=1e-3)

    # Plot figure should be set on session
    assert session.pending_figure is not None
