import pytest
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_dtm import make_tools


def test_build_dtm_tool_known_answer_4():
    """Test build_dtm_tool against Known Answer 4 in TEST_FIXTURE.md."""
    session = SessionState()

    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    dtm_tools = make_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")

    res = build_dtm_tool.invoke({})

    assert res["status"] == "success"
    assert res["n_documents"] == 8
    assert res["n_terms"] == 5
    assert res["non_zero"] == 21
    assert res["total_elements"] == 40
    assert res["sparsity_pct"] == 47.5

    assert session.feature_names == ["alpha", "always", "beta", "delta", "gamma"]
    assert session.feature_matrix is not None
    assert session.feature_matrix.shape == (8, 5)
    assert session.artifacts["count_vectorizer"] is not None


def test_term_frequency_tool_known_answer_14():
    """Test term_frequency_tool against Known Answer 14 in TEST_FIXTURE.md."""
    session = SessionState()

    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    dtm_tools = make_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    term_frequency_tool = next(t for t in dtm_tools if t.name == "term_frequency_tool")

    build_dtm_tool.invoke({})
    res = term_frequency_tool.invoke({"top_n": 20})

    assert res["status"] == "success"
    assert res["total_terms"] == 5
    assert res["total_occurrences"] == 27

    top_terms = res["top_terms"]
    assert top_terms["always"] == 7
    assert top_terms["gamma"] == 7
    assert top_terms["alpha"] == 6
    assert top_terms["delta"] == 4
    assert top_terms["beta"] == 3


def test_dtm_heatmap_tool_known_answer_21():
    """Test dtm_heatmap_tool against Known Answer 21 in TEST_FIXTURE.md."""
    session = SessionState()

    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    dtm_tools = make_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    dtm_heatmap_tool = next(t for t in dtm_tools if t.name == "dtm_heatmap_tool")

    # Build DTM first
    build_dtm_tool.invoke({})

    # 1. Test full matrix default slice
    res_full = dtm_heatmap_tool.invoke({"n_terms": 20, "n_documents": 20})

    assert res_full["status"] == "success"
    assert res_full["n_documents_shown"] == 8
    assert res_full["n_terms_shown"] == 5
    assert res_full["terms"] == ["alpha", "always", "beta", "delta", "gamma"]
    assert session.pending_figure is not None

    # 2. Test corner slice matching Known Answer #21 (n_terms=2, n_documents=3)
    res_corner = dtm_heatmap_tool.invoke({"n_terms": 2, "n_documents": 3})

    assert res_corner["status"] == "success"
    assert res_corner["n_documents_shown"] == 3
    assert res_corner["n_terms_shown"] == 2
    assert res_corner["terms"] == ["alpha", "always"]
    assert res_corner["matrix_slice"] == [[3, 1], [2, 1], [1, 1]]
