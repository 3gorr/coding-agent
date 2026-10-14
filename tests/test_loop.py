import json

from conftest import FIXED_ORDER, UNITTEST, call, final, reply, tool_messages, tool_result

from agent.tools import default_registry


def control_script():
    return [
        reply(call("c1", "list_files", path=".")),
        reply(call("c2", "read_file", path="order.py"), call("c3", "read_file", path="test_order.py")),
        reply(call("c4", "write_file", path="order.py", content=FIXED_ORDER)),
        reply(call("c5", "Bash", command=UNITTEST, cwd=".")),
        final("Исправил calculate_total, тесты проходят."),
    ]


def test_control_task_fixed_and_tests_run_through_bash(project):
    test_hash_before = project.hashes()["test_order.py"]
    agent, run_id = project.start(control_script(), mode="auto")

    assert agent.run() == "finished"

    store = project.store()
    bash = tool_result(store, run_id, "c5")
    assert bash["exit_code"] == 0
    assert "OK" in bash["stderr"]
    assert "test_discount" in bash["stderr"]
    assert project.hashes()["test_order.py"] == test_hash_before
    assert (project.ws / "order.py").read_text() == FIXED_ORDER
    assert not (project.ws / "__pycache__").exists()


def test_tool_results_use_role_tool_and_original_ids(project):
    agent, run_id = project.start(control_script(), mode="auto")
    agent.run()
    msgs = tool_messages(project.store(), run_id)
    assert [m["tool_call_id"] for m in msgs] == ["c1", "c2", "c3", "c4", "c5"]
    assert all(m["role"] == "tool" for m in msgs)
    assert "def calculate_total" in tool_result(project.store(), run_id, "c2")["content"]
    assert "OrderTest" in tool_result(project.store(), run_id, "c3")["content"]


def test_tools_are_sent_with_name_description_and_json_schema(project):
    agent, _ = project.start([final()], mode="auto")
    agent.run()
    specs = default_registry().specs("execution", with_tasks=False)
    for spec in specs:
        fn = spec["function"]
        assert fn["name"] and fn["description"]
        assert fn["parameters"]["type"] == "object"
    names = {s["function"]["name"] for s in specs}
    assert {"Bash", "list_files", "read_file", "write_file"} <= names
    assert agent.llm.requests[0]["tools"]


def test_tool_error_does_not_crash_and_model_can_choose_next_step(project):
    script = [
        reply(call("e1", "read_file", path="missing.py")),
        reply(call("e2", "read_file", path="order.py")),
        final(),
    ]
    agent, run_id = project.start(script)
    assert agent.run() == "finished"
    err = tool_result(project.store(), run_id, "e1")
    assert err["ok"] is False and "не найден" in err["error"]
    assert tool_result(project.store(), run_id, "e2")["ok"] is True


def test_unknown_tool_bad_json_and_schema_errors_are_returned_to_model(project):
    bad_json = {"id": "b2", "type": "function", "function": {"name": "read_file", "arguments": "{oops"}}
    script = [
        reply(call("b1", "rm_everything", path="."), bad_json,
              call("b3", "read_file"),
              call("b4", "write_file", path="x.py", content=1)),
        final(),
    ]
    agent, run_id = project.start(script)
    assert agent.run() == "finished"
    store = project.store()
    assert "неизвестный инструмент" in tool_result(store, run_id, "b1")["error"]
    assert "JSON" in tool_result(store, run_id, "b2")["error"]
    assert "JSON Schema" in tool_result(store, run_id, "b3")["error"]
    assert "JSON Schema" in tool_result(store, run_id, "b4")["error"]
    assert not (project.ws / "x.py").exists()
    assert store.get_run(run_id)["budget_used"]["tool_calls"] == 0


def test_bash_timeout(project):
    script = [reply(call("t1", "Bash", command="sleep 5", cwd=".")), final()]
    agent, run_id = project.start(script, bash_timeout=0.5)
    assert agent.run() == "finished"
    result = tool_result(project.store(), run_id, "t1")
    assert result["timed_out"] is True and result["ok"] is False


def test_bash_output_is_limited_and_marked(project):
    script = [reply(call("o1", "Bash", command="seq 100000", cwd=".")), final()]
    budget = project.budget(per_call_output_bytes=2000)
    agent, run_id = project.start(script, budget=budget)
    agent.run()
    store = project.store()
    msg = [m for m in tool_messages(store, run_id) if m["tool_call_id"] == "o1"][0]
    assert len(msg["content"].encode()) <= 2000
    assert "ОБРЕЗАНО RUNTIME" in msg["content"] or json.loads(msg["content"])["stdout_truncated"]
    row = store.latest_call(run_id, "o1")
    assert row["result"]["stdout_truncated"] is True


def test_list_files_hides_denied_directory(project):
    script = [reply(call("l1", "list_files", path=".")), final()]
    agent, run_id = project.start(script)
    agent.run()
    entries = tool_result(project.store(), run_id, "l1")["entries"]
    assert "order.py" in entries
    assert not any(e.startswith("private") for e in entries)
