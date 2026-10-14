
import pytest
from conftest import (FIXED_ORDER, UNITTEST, call, execution_script, final, planning_script, reply,
                      tool_result)

from agent.crash import SimulatedCrash


def used(project, run_id) -> dict:
    return project.store().get_run(run_id)["budget_used"]


def test_model_requests_limit(project):
    script = [reply(call("r1", "read_file", path="order.py")),
              reply(call("r2", "read_file", path="order.py")), final()]
    agent, run_id = project.start(script, budget=project.budget(model_requests=2))
    assert agent.run() == "budget_exhausted"
    assert len(agent.llm.requests) == 2
    assert used(project, run_id)["model_requests"] == 2
    assert "model_requests" in project.store().get_run(run_id)["status_detail"]


def test_tool_calls_limit_whole_batch_is_rejected(project):
    script = [reply(call("a", "read_file", path="order.py"),
                    call("b", "read_file", path="test_order.py"),
                    call("c", "list_files", path=".")),
              final()]
    agent, run_id = project.start(script, budget=project.budget(tool_calls=2))
    assert agent.run() == "finished"
    store = project.store()
    assert [store.latest_call(run_id, c)["status"] for c in "abc"] == ["rejected_budget"] * 3
    assert all(store.latest_call(run_id, c)["started_at"] is None for c in "abc")
    assert used(project, run_id)["tool_calls"] == 0
    assert "Ни один вызов из этой пачки" in tool_result(store, run_id, "a")["error"]


def test_tool_calls_limit_exhausted_stops_run(project):
    script = [reply(call("a", "read_file", path="order.py")),
              reply(call("b", "read_file", path="order.py")), final()]
    agent, run_id = project.start(script, budget=project.budget(tool_calls=1))
    assert agent.run() == "budget_exhausted"
    store = project.store()
    assert store.latest_call(run_id, "a")["status"] == "done"
    assert store.latest_call(run_id, "b")["status"] == "rejected_budget"
    assert used(project, run_id)["tool_calls"] == 1


def test_side_effects_limit(project):
    script = [reply(call("w1", "write_file", path="order.py", content=FIXED_ORDER)),
              reply(call("b1", "Bash", command=UNITTEST, cwd=".")),
              reply(call("r1", "read_file", path="order.py")),
              final()]
    agent, run_id = project.start(script, budget=project.budget(side_effects=1))
    assert agent.run() == "finished"
    store = project.store()
    assert store.latest_call(run_id, "w1")["status"] == "done"
    assert store.latest_call(run_id, "b1")["status"] == "rejected_budget"
    assert store.latest_call(run_id, "b1")["started_at"] is None
    assert store.latest_call(run_id, "r1")["status"] == "done"
    assert used(project, run_id)["side_effects"] == 1


def test_side_effects_batch_does_not_partially_execute(project):
    script = [reply(call("w1", "write_file", path="a.txt", content="a"),
                    call("w2", "write_file", path="b.txt", content="b")), final()]
    agent, run_id = project.start(script, budget=project.budget(side_effects=1))
    agent.run()
    assert not (project.ws / "a.txt").exists() and not (project.ws / "b.txt").exists()


def test_output_bytes_limit_truncates_and_then_blocks(project):
    script = [reply(call("r1", "read_file", path="test_order.py")),
              reply(call("r2", "read_file", path="order.py")), final()]
    agent, run_id = project.start(script, budget=project.budget(output_bytes=300))
    assert agent.run() == "budget_exhausted"
    store = project.store()
    content = [m for _, m in store.messages(run_id)
               if m["role"] == "tool" and m["tool_call_id"] == "r1"][0]["content"]
    assert len(content.encode()) <= 300
    assert "ОБРЕЗАНО RUNTIME" in content and "вывод неполный" in content
    assert store.latest_call(run_id, "r1")["truncated"] == 1
    assert used(project, run_id)["output_bytes"] == 300
    assert store.latest_call(run_id, "r2")["status"] == "rejected_budget"


def test_counters_survive_resume_and_plan_approval(project):
    script = planning_script() + execution_script()
    agent, run_id = project.start(script, mode="safe", use_plan=True)
    agent.run()
    after_planning = used(project, run_id)
    assert after_planning["model_requests"] == 3

    agent2 = project.agent(run_id, script)
    agent2.approve_plan()
    assert used(project, run_id) == after_planning
    agent2.run()
    mid = used(project, run_id)
    assert mid["model_requests"] > after_planning["model_requests"]

    agent3 = project.agent(run_id, script)
    agent3.approve()
    agent3.run()
    later = used(project, run_id)
    assert later["model_requests"] > mid["model_requests"]
    assert later["tool_calls"] > mid["tool_calls"]


def test_crash_after_charge_does_not_refund_budget(project):
    script = [reply(call("w1", "write_file", path="order.py", content=FIXED_ORDER)), final()]
    budget = project.budget(per_call_output_bytes=5000)
    agent, run_id = project.start(script, budget=budget, crash_at="after_exec")
    with pytest.raises(SimulatedCrash):
        agent.run()
    charged = used(project, run_id)
    assert charged["tool_calls"] == 1 and charged["side_effects"] == 1
    assert charged["output_bytes"] == 5000

    agent2 = project.agent(run_id, script)
    agent2.run()
    assert used(project, run_id) == charged
    agent2.recover("retry")
    agent2.run()
    after = used(project, run_id)
    assert after["tool_calls"] == 2 and after["side_effects"] == 2


def test_retry_is_blocked_when_budget_is_exhausted(project):
    script = [reply(call("w1", "write_file", path="order.py", content=FIXED_ORDER)), final()]
    agent, run_id = project.start(script, budget=project.budget(side_effects=1), crash_at="after_exec")
    with pytest.raises(SimulatedCrash):
        agent.run()
    agent2 = project.agent(run_id, script)
    agent2.run()
    agent2.recover("retry")
    agent2.run()
    store = project.store()
    attempt2 = store.calls(run_id, call_id="w1")[-1]
    assert attempt2["status"] == "rejected_budget" and attempt2["started_at"] is None
