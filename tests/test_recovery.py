import json
import os
import subprocess
import sys

import pytest
from conftest import FIXED_ORDER, REPO, UNITTEST, call, final, reply, tool_result

from agent.crash import SimulatedCrash
from agent.loop import UserError


def fix_script():
    return [
        reply(call("w1", "write_file", path="order.py", content=FIXED_ORDER)),
        reply(call("b1", "Bash", command=UNITTEST, cwd=".")),
        final(),
    ]


def crash_on_write(project, mode="auto"):
    agent, run_id = project.start(fix_script(), mode=mode, crash_at="after_exec")
    with pytest.raises(SimulatedCrash):
        agent.run()
    return run_id


def test_crash_between_execution_and_saving_result_makes_call_uncertain(project):
    run_id = crash_on_write(project)
    store = project.store()
    row = store.latest_call(run_id, "w1")
    assert (project.ws / "order.py").read_text() == FIXED_ORDER
    assert row["status"] == "started" and row["result"] is None

    agent = project.agent(run_id, fix_script())
    assert agent.run() == "awaiting_recovery"
    assert project.store().latest_call(run_id, "w1")["status"] == "uncertain"
    assert "Неопределённый вызов w1" in project.output


def test_uncertain_call_is_not_silently_repeated(project):
    run_id = crash_on_write(project)
    (project.ws / "order.py").write_text("# изменено после сбоя\n")
    agent = project.agent(run_id, fix_script())
    agent.run()
    agent.run()
    assert (project.ws / "order.py").read_text() == "# изменено после сбоя\n"
    assert len(project.store().calls(run_id, call_id="w1")) == 1


def test_recover_retry_is_new_attempt_of_same_tool_call(project):
    run_id = crash_on_write(project)
    agent = project.agent(run_id, fix_script())
    agent.run()
    agent.recover("retry")
    assert agent.run() == "finished"
    attempts = project.store().calls(run_id, call_id="w1")
    assert [(a["attempt"], a["status"]) for a in attempts] == [(1, "superseded"), (2, "done")]
    assert attempts[1]["user_decision"] == "retry"
    assert tool_result(project.store(), run_id, "b1")["exit_code"] == 0


def test_recover_fail_marks_step_failed_and_continues(project):
    run_id = crash_on_write(project)
    agent = project.agent(run_id, fix_script())
    agent.run()
    agent.recover("fail")
    assert agent.run() == "finished"
    result = tool_result(project.store(), run_id, "w1")
    assert result["ok"] is False and "неуспешным" in result["error"]
    assert project.store().latest_call(run_id, "w1")["status"] == "failed"


def test_recover_abort_finishes_run_with_error(project):
    run_id = crash_on_write(project)
    agent = project.agent(run_id, fix_script())
    agent.run()
    agent.recover("abort")
    assert agent.run() == "failed"


def test_retry_is_refused_for_non_retryable_tool(project):
    (project.ws / "tmp.txt").write_text("x")
    script = [reply(call("d1", "delete_file", path="tmp.txt")), final()]
    agent, run_id = project.start(script, mode="auto", crash_at="after_exec")
    assert agent.run() == "awaiting_approval"
    crashing = project.agent(run_id, script, crash_at="after_exec")
    crashing.approve()
    with pytest.raises(SimulatedCrash):
        crashing.run()
    agent2 = project.agent(run_id, script)
    agent2.run()
    with pytest.raises(UserError):
        agent2.recover("retry")
    agent2.recover("fail")
    assert agent2.run() == "finished"


def test_completed_action_is_not_repeated_after_restart(project):
    agent, run_id = project.start(fix_script(), crash_at="after_start", crash_nth=2)
    with pytest.raises(SimulatedCrash):
        agent.run()
    assert project.store().latest_call(run_id, "w1")["status"] == "done"
    (project.ws / "order.py").write_text("# внешний файл\n")
    agent2 = project.agent(run_id, fix_script())
    assert agent2.run() == "awaiting_recovery"
    assert [r["call_id"] for r in project.store().calls(run_id, status="uncertain")] == ["b1"]
    agent2.recover("retry")
    assert agent2.run() == "finished"
    assert len(project.store().calls(run_id, call_id="w1")) == 1
    assert (project.ws / "order.py").read_text() == "# внешний файл\n"


def test_resume_after_crash_before_execution(project):
    """Упали после записи 'started', но ДО запуска инструмента - вызов всё равно неопределённый."""
    agent, run_id = project.start(fix_script(), crash_at="after_start")
    with pytest.raises(SimulatedCrash):
        agent.run()
    assert (project.ws / "order.py").read_text() != FIXED_ORDER
    agent2 = project.agent(run_id, fix_script())
    assert agent2.run() == "awaiting_recovery"
    agent2.recover("retry")
    assert agent2.run() == "finished"
    assert (project.ws / "order.py").read_text() == FIXED_ORDER


def test_repeated_tool_call_id_with_same_args_does_not_modify_files(project):
    script = [
        reply(call("w1", "write_file", path="order.py", content=FIXED_ORDER)),
        reply(call("w1", "write_file", path="order.py", content=FIXED_ORDER)),
        final(),
    ]
    agent, run_id = project.start(script, crash_at=None)
    original_chat = agent.llm.chat

    def chat(messages, tools):
        if sum(1 for m in messages if m["role"] == "assistant") == 1:
            (project.ws / "order.py").write_text("# правка пользователя\n")
        return original_chat(messages, tools)

    agent.llm.chat = chat
    assert agent.run() == "finished"
    assert (project.ws / "order.py").read_text() == "# правка пользователя\n"
    store = project.store()
    assert len(store.calls(run_id, call_id="w1")) == 1
    assert [e for e in store.events(run_id) if e["kind"] == "tool_call_replay"][0]["data"]["same"] is True


def test_repeated_tool_call_id_with_other_args_is_protocol_error(project):
    script = [
        reply(call("w1", "write_file", path="order.py", content=FIXED_ORDER)),
        reply(call("w1", "write_file", path="order.py", content="другое")),
        final(),
    ]
    agent, run_id = project.start(script)
    assert agent.run() == "finished"
    assert (project.ws / "order.py").read_text() == FIXED_ORDER
    result = tool_result(project.store(), run_id, "w1")
    assert "Ошибка протокола" in result["error"]


def test_real_process_crash_via_cli(project, tmp_path):
    """Настоящее падение процесса (os._exit) и продолжение через CLI."""
    script_file = tmp_path / "script.json"
    script_file.write_text(json.dumps(fix_script(), ensure_ascii=False))
    base = [sys.executable, "-m", "agent", "--db", str(project.db), "--scripted", str(script_file)]
    env = {**os.environ, "AGENT_CRASH_AT": "after_exec"}

    first = subprocess.run(base + ["run", "fix", "--mode", "auto", "--no-plan",
                                   "--workspace", str(project.ws), "--policy", str(project.policy)],
                           cwd=REPO, env=env, capture_output=True, text=True)
    assert first.returncode == 137, first.stderr
    run_id = first.stdout.split("run_id:")[1].split()[0]

    env.pop("AGENT_CRASH_AT")
    resumed = subprocess.run(base + ["resume", run_id], cwd=REPO, env=env, capture_output=True, text=True)
    assert "awaiting_recovery" in resumed.stdout

    done = subprocess.run(base + ["recover", run_id, "--retry"], cwd=REPO, env=env,
                          capture_output=True, text=True)
    assert "статус: finished" in done.stdout, done.stdout + done.stderr
