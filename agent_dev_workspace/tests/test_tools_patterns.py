import pytest
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_patterns import make_tools as make_patterns_tools


def test_mine_patterns_zero_surviving_terms():
    """Test that variance or term_frequency filtering handles zero surviving terms cleanly."""
    session = SessionState()

    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    patterns_tools = make_patterns_tools(session)
    mine_patterns_tool = next(t for t in patterns_tools if t.name == "mine_patterns_tool")

    res_var = mine_patterns_tool.invoke({"category_name": "catA", "filtering_method": "variance"})
    assert res_var["status"] == "success"
    assert res_var["patterns"] == []
    assert "no terms survived filtering" in res_var["note"]


def test_mine_patterns_fpgrowth_known_answer_17():
    """Test mine_patterns_tool with FPGrowth against Known Answer 17 in TEST_FIXTURE.md."""
    session = SessionState()

    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    patterns_tools = make_patterns_tools(session)
    mine_patterns_tool = next(t for t in patterns_tools if t.name == "mine_patterns_tool")

    # catA mining
    res_catA = mine_patterns_tool.invoke({
        "category_name": "catA",
        "filtering_method": "tfidf",
        "algorithm": "fpgrowth",
        "min_sup": 1
    })

    assert res_catA["status"] == "success"
    patterns_catA = res_catA["patterns"]
    assert len(patterns_catA) == 1
    assert patterns_catA[0]["pattern"] == ["alpha"]
    assert patterns_catA[0]["support"] == 3

    # catB mining
    res_catB = mine_patterns_tool.invoke({
        "category_name": "catB",
        "filtering_method": "tfidf",
        "algorithm": "fpgrowth",
        "min_sup": 1
    })

    assert res_catB["status"] == "success"
    patterns_catB = res_catB["patterns"]
    assert len(patterns_catB) == 1
    assert patterns_catB[0]["pattern"] == ["gamma"]
    assert patterns_catB[0]["support"] == 4


def test_mine_patterns_topk_and_maxfpgrowth_known_answer_20():
    """Test mine_patterns_tool topk and maxfpgrowth against Known Answer 20 in TEST_FIXTURE.md."""
    session = SessionState()

    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    patterns_tools = make_patterns_tools(session)
    mine_patterns_tool = next(t for t in patterns_tools if t.name == "mine_patterns_tool")

    # Top-K test
    res_topk = mine_patterns_tool.invoke({
        "category_name": "catA",
        "filtering_method": "tfidf",
        "algorithm": "topk",
        "k": 1
    })
    assert res_topk["status"] == "success"
    assert len(res_topk["patterns"]) == 1
    assert res_topk["patterns"][0]["pattern"] == ["alpha"]

    # MaxFPGrowth test (Known Answer #20 expects 0 patterns)
    res_max = mine_patterns_tool.invoke({
        "category_name": "catA",
        "filtering_method": "tfidf",
        "algorithm": "maxfpgrowth",
        "min_sup": 1
    })
    assert res_max["status"] == "success"
    assert res_max["patterns"] == []
