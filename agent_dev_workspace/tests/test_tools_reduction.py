import pytest
import pandas as pd
import numpy as np
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_reduction import make_tools

def test_binarize_labels_tool_custom_design():
    # 1. Unloaded / empty session error test
    session_unloaded = SessionState()
    tools_unloaded = make_tools(session_unloaded)
    binarize_unloaded = next(t for t in tools_unloaded if t.name == "binarize_labels_tool")
    res_err = binarize_unloaded.invoke({})
    assert "error" in res_err

    # 2. Setup session with sample fixture dataset
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    load_res = load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })
    assert load_res["n_documents"] == 8

    # Snapshot session.dataframe before tool execution to verify invariance
    df_before = session.dataframe.copy()

    tools = make_tools(session)
    binarize_tool = next(t for t in tools if t.name == "binarize_labels_tool")

    # 3. Invoke binarize_labels_tool
    result = binarize_tool.invoke({})

    # Assert summary payload structure
    assert "result_id" in result
    assert result["n_documents"] == 8
    assert result["categories"] == ["catA", "catB"]
    assert result["shape"] == [8, 2]

    # 4. Assert exact binarized values for catA and catB
    expected_catA = [1, 1, 1, 0, 0, 0, 0, 1]
    expected_catB = [0, 0, 0, 1, 1, 1, 1, 0]

    binarized_df = session.artifacts["binarized_labels"]
    assert isinstance(binarized_df, pd.DataFrame)
    assert binarized_df.shape == (8, 2)
    assert list(binarized_df.columns) == ["catA", "catB"]
    assert binarized_df["catA"].tolist() == expected_catA
    assert binarized_df["catB"].tolist() == expected_catB

    # Row sums across classes must equal 1
    assert (binarized_df.sum(axis=1) == 1).all()

    # 5. Assert full_report in results_store
    result_id = result["result_id"]
    stored_result = session.results_store[result_id]
    assert stored_result["full_report"] is binarized_df

    # 6. Assert session.dataframe remains unchanged
    pd.testing.assert_frame_equal(session.dataframe, df_before)
