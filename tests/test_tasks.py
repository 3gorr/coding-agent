import pytest
from conftest import (FIXED_ORDER, PLAN, UNITTEST, call, execution_script, final, planning_script,
                      reply, tool_result)

from agent.crash import SimulatedCrash
from agent.tasks import checklist


def approved_run(project, exec_script, mode="auto", crash_at=None, crash_nth=1):
    """Запуск с уже утверждённым планом. Возвращает (agent, run_id, полный сценарий)."""
    script = planning_script() + exec_script
    agent, run_id = project.start(script, mode=mode, use_plan=True)
    assert agent.run() == "awaiting_plan_approval"
    agent2 = project.agent(run_id, script, crash_at=crash_at, crash_nth=crash_nth)
    agent2.approve_plan()
    return agent2, run_id, script


def test_full_run_with_four_tasks_and_dependencies(project):
    agent, run_id, _ = approved_run(project, execution_script())
    assert agent.run() == "finished"
    store = project.store()
    tasks = store.tasks(run_id)
    assert [t["task_id"] for t in tasks] == ["T1", "T2", "T3", "T4"]
    assert all(t["status"] == "done" for t in tasks)
    assert tasks[1]["depends_on"] == ["T1"] and tasks[2]["depends_on"] == ["T2"]
    assert set(tasks[0]["evidence"]["read_calls"]) == {"order.py", "test_order.py"}
    assert tasks[1]["evidence"]["sha_before"] != tasks[1]["evidence"]["sha_after"]
    assert tasks[2]["evidence"]["call_id"] == "e6" and tasks[2]["evidence"]["exit_code"] == 0
    assert tasks[3]["evidence"]["sha_before"] == tasks[3]["evidence"]["sha_after"]
    assert (project.ws / "order.py").read_text() == FIXED_ORDER


def test_every_execution_call_is_linked_to_current_task(project):
    agent, run_id, _ = approved_run(project, execution_script())
    agent.run()
    rows = {r["call_id"]: r for r in project.store().calls(run_id)}
    expected = {"e1": "T1", "e2": "T1", "e3": "T1", "e4": "T2", "e5": "T2",
                "e6": "T3", "e7": "T3", "e8": "T4"}
    for cid, task_id in expected.items():
        assert rows[cid]["phase"] == "execution" and rows[cid]["task_id"] == task_id, cid
    assert all(rows[c]["phase"] == "planning" and rows[c]["task_id"] is None
               for c in ("p1", "p2", "p3", "p4", "p5", "p6"))


def test_checklist_is_printed_to_stdout(project):
    agent, run_id, _ = approved_run(project, execution_script())
    agent.run()
    out = project.output
    assert "- [ ] T1: Прочитать order.py и test_order.py  <- в работе" in out
    assert "- [x] T1: Прочитать order.py и test_order.py" in out
    assert "- [x] T4: Убедиться, что тесты не изменены (после: T3)" in out


def test_model_claim_is_not_enough_to_complete_task(project):
    script = [
        reply(call("f1", "finish_task", task_id="T1", summary="всё прочитал")),
        final("Задача T1 выполнена!"),
        reply(call("r1", "read_file", path="order.py"), call("r2", "read_file", path="test_order.py")),
        reply(call("f2", "finish_task", task_id="T1")),
        final(),
    ]
    agent, run_id, _ = approved_run(project, script)
    agent.run()
    store = project.store()
    assert "не подтвердил" in tool_result(store, run_id, "f1")["error"]
    assert tool_result(store, run_id, "f2")["ok"] is True
    assert store.get_task(run_id, "T1")["status"] == "done"
    assert store.get_task(run_id, "T2")["status"] == "in_progress"
    assert "задачи не подтверждены" in project.output


def test_task_does_not_start_before_dependencies(project):
    script = [reply(call("f1", "finish_task", task_id="T3")), final()]
    agent, run_id, _ = approved_run(project, script)
    agent.run()
    store = project.store()
    assert "сейчас выполняется задача T1" in tool_result(store, run_id, "f1")["error"]
    assert store.get_task(run_id, "T3")["status"] == "pending"
    assert store.get_task(run_id, "T2")["started_at"] is None


def test_command_check_fails_if_files_changed_after_tests(project):
    script = execution_script()
    script.insert(5, reply(call("late", "write_file", path="order.py", content=FIXED_ORDER + "\n")))
    script.insert(7, reply(call("b2", "Bash", command=UNITTEST, cwd=".")))
    script.insert(8, reply(call("f3", "finish_task", task_id="T3")))
    agent, run_id, _ = approved_run(project, script)
    agent.run()
    store = project.store()
    assert "файлы менялись" in tool_result(store, run_id, "e7")["error"]
    assert store.get_task(run_id, "T3")["evidence"]["call_id"] == "b2"


def test_completed_tasks_are_not_rerun_after_resume(project):
    agent, run_id, script = approved_run(project, execution_script(), crash_at="after_exec", crash_nth=6)
    with pytest.raises(SimulatedCrash):
        agent.run()
    store = project.store()
    done_before = {t["task_id"]: t["finished_at"] for t in store.tasks(run_id) if t["status"] == "done"}
    assert set(done_before) == {"T1", "T2"}
    calls_before = len(store.calls(run_id))

    agent2 = project.agent(run_id, script)
    assert agent2.run() == "awaiting_recovery"
    agent2.recover("retry")
    assert agent2.run() == "finished"

    store = project.store()
    for task_id, finished_at in done_before.items():
        assert store.get_task(run_id, task_id)["finished_at"] == finished_at
    new_calls = store.calls(run_id)[calls_before:]
    assert all(r["task_id"] not in ("T1", "T2") for r in new_calls)
    assert store.get_task(run_id, "T3")["evidence"]["attempt"] == 2


def test_finish_task_is_idempotent_for_done_task(project):
    script = execution_script()[:2] + [reply(call("again", "finish_task", task_id="T1")), final()]
    agent, run_id, _ = approved_run(project, script)
    agent.run()
    result = tool_result(project.store(), run_id, "again")
    assert result["ok"] is True and result["note"] == "задача уже подтверждена"


def test_checklist_format():
    tasks = [
        {"task_id": "T1", "description": "a", "depends_on": [], "status": "done"},
        {"task_id": "T2", "description": "b", "depends_on": ["T1"], "status": "in_progress"},
        {"task_id": "T3", "description": "c", "depends_on": ["T2"], "status": "pending"},
    ]
    assert checklist(tasks) == ("- [x] T1: a\n"
                                "- [ ] T2: b (после: T1)  <- в работе\n"
                                "- [ ] T3: c (после: T2)")
