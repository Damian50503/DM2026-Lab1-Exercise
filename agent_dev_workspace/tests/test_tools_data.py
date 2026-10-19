import pytest
import pandas as pd
import numpy as np
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_data import make_tools

def test_tokenize_tool_known_answer_10():
    session = SessionState()

    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    load_res = load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    assert load_res["n_documents"] == 8

    tools = make_tools(session)
    tokenize_tool = next(t for t in tools if t.name == "tokenize_tool")

    result = tokenize_tool.invoke({})

    assert "result_id" in result
    assert result["n_documents"] == 8
    assert result["status"] == "success"

    expected_tokens = [
        ["always", "alpha", "alpha", "alpha", "beta"],
        ["always", "alpha", "alpha", "beta"],
        ["always", "alpha", "beta"],
        ["always", "gamma", "gamma", "gamma", "delta"],
        ["always", "gamma", "gamma", "delta"],
        ["always", "gamma", "delta"],
        ["always", "gamma", "delta"],
        []
    ]

    actual_tokens = session.dataframe["unigrams"].tolist()
    assert actual_tokens == expected_tokens


def test_list_files_tool_known_answer_11():
    session = SessionState()

    tools = make_tools(session)
    list_files_tool = next(t for t in tools if t.name == "list_files_tool")

    result = list_files_tool.invoke({"subdirectory": "newdataset"})

    assert result["status"] == "success"
    assert result["subdirectory"] == "newdataset"
    assert result["directories"] == []
    assert result["files"] == ["Reddit-stock-sentiment.csv"]


def test_inspect_data_tool_custom_design():
    session_unloaded = SessionState()
    tools_unloaded = make_tools(session_unloaded)
    inspect_unloaded = next(t for t in tools_unloaded if t.name == "inspect_data_tool")
    res_err = inspect_unloaded.invoke({})
    assert "error" in res_err

    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    tools = make_tools(session)
    inspect_tool = next(t for t in tools if t.name == "inspect_data_tool")

    res_inv1 = inspect_tool.invoke({"n_rows": 0})
    assert "error" in res_inv1
    res_inv2 = inspect_tool.invoke({"max_text_length": -5})
    assert "error" in res_inv2

    res_sliced = inspect_tool.invoke({"n_rows": 3, "max_text_length": 10})
    assert res_sliced["total_rows"] == 8
    assert len(res_sliced["rows"]) == 3
    assert res_sliced["rows"][0]["text"] == "always alp..."
    assert res_sliced["rows"][0]["category_name"] == "catA"

    result_id = res_sliced["result_id"]
    stored_result = session.results_store[result_id]
    assert stored_result["full_report"][0]["text"] == "always alpha alpha alpha beta"

    res_cap = inspect_tool.invoke({"n_rows": 20})
    assert res_cap["total_rows"] == 8
    assert len(res_cap["rows"]) == 8

    session_empty = SessionState()
    session_empty.dataframe = pd.DataFrame(columns=["text", "label"])
    tools_empty = make_tools(session_empty)
    inspect_empty = next(t for t in tools_empty if t.name == "inspect_data_tool")
    res_emp = inspect_empty.invoke({})
    assert res_emp["total_rows"] == 0
    assert res_emp["rows"] == []
    empty_result_id = res_emp["result_id"]
    assert session_empty.results_store[empty_result_id]["full_report"] == []


def test_check_missing_tool_known_answer_2():
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    tools = make_tools(session)
    check_missing_tool = next(t for t in tools if t.name == "check_missing_tool")

    res = check_missing_tool.invoke({})
    assert res["status"] == "success"
    assert res["total_rows"] == 8
    assert res["missing_count"] == 1
    assert pytest.approx(res["missing_pct"], abs=1e-3) == 12.5


def test_check_duplicates_tool_known_answer_3():
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    tools = make_tools(session)
    check_duplicates_tool = next(t for t in tools if t.name == "check_duplicates_tool")

    # Check without dropping
    res_inspect = check_duplicates_tool.invoke({"drop": False})
    assert res_inspect["status"] == "success"
    assert res_inspect["initial_rows"] == 8
    assert res_inspect["duplicate_count"] == 2
    assert res_inspect["final_rows"] == 8

    # Check with dropping (keep=False drops every copy of duplicated rows)
    res_drop = check_duplicates_tool.invoke({"drop": True})
    assert res_drop["status"] == "success"
    assert res_drop["final_rows"] == 6
    assert len(session.dataframe) == 6


def test_sample_data_tool_known_answer_13():
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    tools = make_tools(session)
    sample_data_tool = next(t for t in tools if t.name == "sample_data_tool")

    res = sample_data_tool.invoke({"n": 4, "random_state": 42})
    assert res["n_samples"] == 4
    assert res["original_n_samples"] == 8
    assert res["sampled_indices"] == [1, 5, 0, 7]
    assert len(session.dataframe) == 4


def test_describe_data_tool_known_answer_18():
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    tools = make_tools(session)
    describe_data_tool = next(t for t in tools if t.name == "describe_data_tool")

    res = describe_data_tool.invoke({})
    assert res["status"] == "success"

    # Verify overall text length describe statistics
    overall = res["overall_stats"]
    assert overall["count"] == 8
    assert pytest.approx(overall["mean"], abs=1e-3) == 19.875
    assert pytest.approx(overall["std"], abs=1e-3) == 9.4330
    assert overall["min"] == 0
    assert overall["max"] == 30

    # Verify figure set on pending_figure
    assert session.pending_figure is not None
