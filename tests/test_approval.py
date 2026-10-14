import pytest
from conftest import FIXED_ORDER, UNITTEST, call, final, reply, tool_result

from agent.loop import UserError


def script_write_then_test():
    return [
        reply(call("r1", "read_file", path="order.py")),
        reply(call("w1", "write_file", path="order.py", content=FIXED_ORDER)),
        reply(call("b1", "Bash", command=UNITTEST, cwd=".")),
        final(),
    ]


def test_safe_mode_read_immediately_write_and_bash_after_approval(project):
    original = (project.ws / "order.py").read_text()
    agent, run_id = project.start(script_write_then_test(), mode="safe")

    assert agent.run() == "awaiting_approval"
    store = project.store()
    assert store.latest_call(run_id, "r1")["status"] == "done"
    assert store.latest_call(run_id, "w1")["status"] == "awaiting_approval"
    assert (project.ws / "order.py").read_text() == original

    agent2 = project.agent(run_id, script_write_then_test())
    agent2.approve()
    assert agent2.run() == "awaiting_approval"
    assert (project.ws / "order.py").read_text() == FIXED_ORDER

    agent3 = project.agent(run_id, script_write_then_test())
    agent3.approve()
    assert agent3.run() == "finished"
    assert tool_result(project.store(), run_id, "b1")["exit_code"] == 0
    decisions = [e["kind"] for e in project.store().events(run_id)]
    assert decisions.count("user_approved") == 2


def test_deny_returns_reason_to_model_which_changes_plan(project):
    script = [
        reply(call("w1", "write_file", path="order.py", content="broken")),
        reply(call("w2", "write_file", path="order.py", content=FIXED_ORDER)),
        final(),
    ]
    agent, run_id = project.start(script, mode="safe")
    assert agent.run() == "awaiting_approval"

    agent2 = project.agent(run_id, script)
    agent2.deny(reason="это сломает код")
    assert agent2.run() == "awaiting_approval"
    result = tool_result(project.store(), run_id, "w1")
    assert result["ok"] is False and "это сломает код" in result["error"]
    assert (project.ws / "order.py").read_text() != "broken"

    seen = agent2.llm.requests[-1]["messages"]
    assert any(m["role"] == "tool" and "это сломает код" in m["content"] for m in seen)


def test_delete_requires_approval_even_in_auto(project):
    (project.ws / "tmp.txt").write_text("x")
    script = [reply(call("d1", "delete_file", path="tmp.txt")), final()]
    agent, run_id = project.start(script, mode="auto")
    assert agent.run() == "awaiting_approval"
    assert (project.ws / "tmp.txt").exists()
    agent2 = project.agent(run_id, script)
    agent2.approve()
    assert agent2.run() == "finished"
    assert not (project.ws / "tmp.txt").exists()


def test_auto_mode_writes_and_runs_bash_without_approval(project):
    agent, run_id = project.start(script_write_then_test(), mode="auto")
    assert agent.run() == "finished"
    assert not [e for e in project.store().events(run_id) if e["kind"] == "approval_requested"]


def test_approve_when_nothing_is_waiting_is_an_error(project):
    agent, run_id = project.start([final()], mode="safe")
    agent.run()
    with pytest.raises(UserError):
        project.agent(run_id, []).approve()


def test_batch_with_several_actions_asks_one_by_one(project):
    script = [
        reply(call("a1", "write_file", path="a.txt", content="a"),
              call("a2", "write_file", path="b.txt", content="b")),
        final(),
    ]
    agent, run_id = project.start(script, mode="safe")
    assert agent.run() == "awaiting_approval"
    project.agent(run_id, script).approve()
    agent2 = project.agent(run_id, script)
    assert agent2.run() == "awaiting_approval"
    assert (project.ws / "a.txt").exists() and not (project.ws / "b.txt").exists()
    agent2.deny(reason="не нужно")
    assert agent2.run() == "finished"
    assert not (project.ws / "b.txt").exists()
