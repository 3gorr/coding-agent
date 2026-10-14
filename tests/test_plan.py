import copy

from conftest import PLAN, UNITTEST, call, execution_script, final, planning_script, reply, tool_result

from agent.tools.base import SIDE_EFFECT_KINDS


def test_no_side_effects_before_plan_approval(project):
    before = project.hashes()
    agent, run_id = project.start(planning_script(), mode="auto", use_plan=True)

    assert agent.run() == "awaiting_plan_approval"

    store = project.store()
    assert project.hashes() == before
    side = [r for r in store.calls(run_id) if r["kind"] in SIDE_EFFECT_KINDS]
    assert {r["call_id"] for r in side} == {"p3", "p4", "p5"}
    assert all(r["status"] == "denied" and r["started_at"] is None for r in side)
    for cid in ("p3", "p4", "p5"):
        assert "Plan Mode" in tool_result(store, run_id, cid)["error"]
    assert store.get_run(run_id)["budget_used"]["side_effects"] == 0
    for request in agent.llm.requests:
        assert set(request["tools"]) == {"list_files", "read_file", "submit_plan"}


def test_plan_is_structured_and_saved_with_run_id(project):
    agent, run_id = project.start(planning_script(), use_plan=True)
    agent.run()
    run = project.store().get_run(run_id)
    assert run["plan_status"] == "pending_approval"
    plan = run["plan"]
    assert plan == PLAN
    assert {"goal", "steps", "risks", "assumptions", "done_criteria"} <= set(plan)
    assert all(step["tools"] is not None for step in plan["steps"])
    for row in project.store().calls(run_id):
        assert row["phase"] == "planning" and row["task_id"] is None
    assert "Цель:" in project.output and "Критерий готовности" in project.output


def test_execution_starts_only_after_approval_and_safe_still_asks(project):
    script = planning_script() + execution_script()
    agent, run_id = project.start(script, mode="safe", use_plan=True)
    assert agent.run() == "awaiting_plan_approval"
    assert agent.run() == "awaiting_plan_approval"

    agent2 = project.agent(run_id, script)
    agent2.approve_plan()
    assert agent2.run() == "awaiting_approval"
    row = project.store().calls(run_id, status="awaiting_approval")[0]
    assert row["name"] == "write_file" and row["task_id"] == "T2"


def test_reject_plan_sends_reason_and_model_replans(project):
    script = planning_script() + [reply(call("p7", "submit_plan", **PLAN))]
    agent, run_id = project.start(script, use_plan=True)
    agent.run()
    agent2 = project.agent(run_id, script)
    agent2.reject_plan("добавь шаг проверки тестов")
    assert agent2.run() == "awaiting_plan_approval"
    last_request = agent2.llm.requests[-1]["messages"]
    assert any("добавь шаг проверки тестов" in (m.get("content") or "") for m in last_request)
    assert project.store().get_run(run_id)["phase"] == "planning"


def test_invalid_plan_is_rejected_by_runtime(project):
    bad = copy.deepcopy(PLAN)
    bad["steps"][0]["depends_on"] = ["T4"]
    bad["steps"][1]["tools"] = ["rm"]
    bad["steps"][2]["verification"]["command"] = f"{UNITTEST} | cat"
    script = [reply(call("x1", "submit_plan", **bad)), reply(call("x2", "submit_plan", **PLAN))]
    agent, run_id = project.start(script, use_plan=True)
    assert agent.run() == "awaiting_plan_approval"
    result = tool_result(project.store(), run_id, "x1")
    assert result["ok"] is False
    problems = " ".join(result["problems"])
    assert "rm" in problems and "policy" in problems
    assert project.store().get_run(run_id)["plan"] == PLAN


def test_cycle_is_detected(project):
    bad = copy.deepcopy(PLAN)
    bad["steps"][0]["depends_on"] = ["T4"]
    script = [reply(call("x1", "submit_plan", **bad)), final()]
    agent, run_id = project.start(script, use_plan=True)
    agent.run()
    assert "циклическая" in " ".join(tool_result(project.store(), run_id, "x1")["problems"])


def test_text_answer_during_planning_does_not_finish_run(project):
    script = [final("вот мой план словами"), reply(call("x1", "submit_plan", **PLAN))]
    agent, run_id = project.start(script, use_plan=True)
    assert agent.run() == "awaiting_plan_approval"
