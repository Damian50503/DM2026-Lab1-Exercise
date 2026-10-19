import pytest
import numpy as np
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_filtering import make_tools


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

    # Build global DTM using CountVectorizer defaults (matching Known Answer #4)
    from sklearn.feature_extraction.text import CountVectorizer
    texts = session.dataframe["text"].fillna("").tolist()
    vectorizer = CountVectorizer()
    session.feature_matrix = vectorizer.fit_transform(texts)
    session.feature_names = vectorizer.get_feature_names_out().tolist()
    session.set_labels(session.dataframe["category_name"].values)
    return session


def test_variance_filter_tool_known_answer_5():
    """Tests variance_filter_tool against Known Answer #5 in TEST_FIXTURE.md."""
    session = setup_fixture_session()
    tools = make_tools(session)
    variance_tool = next(t for t in tools if t.name == "variance_filter_tool")

    # Call with threshold=0.15
    res = variance_tool.invoke({"threshold": 0.15})

    assert res["status"] == "success"
    assert res["initial_terms"] == 5
    assert res["kept_terms_count"] == 4
    assert res["removed_terms_count"] == 1

    # Threshold 0.15 keeps alpha, beta, delta, gamma and removes 'always' (variance 0.109375)
    assert set(res["kept_terms"]) == {"alpha", "beta", "delta", "gamma"}


def test_pearson_filter_tool_known_answer_6():
    """Tests pearson_filter_tool against Known Answer #6 in TEST_FIXTURE.md."""
    session = setup_fixture_session()
    tools = make_tools(session)
    pearson_tool = next(t for t in tools if t.name == "pearson_filter_tool")

    res = pearson_tool.invoke({"target_class": "catB"})

    assert res["status"] == "success"
    assert res["target_class"] == "catB"
    corrs = res["correlations"]

    # Assert known Pearson r values from TEST_FIXTURE.md #6
    assert np.isclose(corrs["alpha"], -0.6882, atol=1e-3)
    assert np.isclose(corrs["always"], 0.3780, atol=1e-3)
    assert np.isclose(corrs["beta"], -0.7746, atol=1e-3)
    assert np.isclose(corrs["delta"], 1.0000, atol=1e-3)
    assert np.isclose(corrs["gamma"], 0.8307, atol=1e-3)


def test_spearman_filter_tool_known_answer_6():
    """Tests spearman_filter_tool against Known Answer #6 in TEST_FIXTURE.md."""
    session = setup_fixture_session()
    tools = make_tools(session)
    spearman_tool = next(t for t in tools if t.name == "spearman_filter_tool")

    res = spearman_tool.invoke({"target_class": "catB"})

    assert res["status"] == "success"
    assert res["target_class"] == "catB"
    corrs = res["correlations"]

    # Assert known Spearman r values from TEST_FIXTURE.md #6
    assert np.isclose(corrs["alpha"], -0.7500, atol=1e-3)
    assert np.isclose(corrs["always"], 0.3780, atol=1e-3)
    assert np.isclose(corrs["beta"], -0.7746, atol=1e-3)
    assert np.isclose(corrs["delta"], 1.0000, atol=1e-3)
    assert np.isclose(corrs["gamma"], 0.9363, atol=1e-3)
