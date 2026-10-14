import json
import os

import pytest
from conftest import FIXED_ORDER, UNITTEST, call, final, reply, tool_result

from agent.policy import Policy, PolicyConfigError, PolicyViolation
from agent.tools import default_registry

REG = default_registry()


def policy(project) -> Policy:
    return Policy(project.policy, project.ws)


def decide(project, tool, mode="auto", phase="execution", **args):
    return policy(project).check(REG.get(tool), args, mode=mode, phase=phase)



def test_allowed_command(project):
    assert decide(project, "Bash", command=UNITTEST, cwd=".").action == "allow"
    assert decide(project, "Bash", mode="safe", command=UNITTEST, cwd=".").action == "ask"


def test_denied_command_deny_wins_over_allow(project):
    d = decide(project, "Bash", command="rm -rf .", cwd=".")
    assert d.action == "deny" and "commands.deny" in d.reason


def test_command_not_in_allow_is_denied(project):
    d = decide(project, "Bash", command="python3 -m unittest", cwd=".")
    assert d.action == "deny" and "commands.allow" in d.reason


@pytest.mark.parametrize("command", [
    f"{UNITTEST} | cat",
    f"{UNITTEST}; rm -rf /",
    f"{UNITTEST} > out.txt",
    "echo $(cat /etc/passwd)",
    "bash -c 'rm -rf .'",
    "ls && rm order.py",
    "ls *",
])
def test_shell_syntax_is_rejected_as_a_whole(project, command):
    d = decide(project, "Bash", command=command, cwd=".")
    assert d.action == "deny"


def test_bash_c_is_not_allowed_even_without_metachars(project):
    d = decide(project, "Bash", command="bash -c ls", cwd=".")
    assert d.action == "deny"



def test_allowed_command_in_denied_directory(project):
    d = decide(project, "Bash", command=UNITTEST, cwd="private")
    assert d.action == "deny" and "directories.deny" in d.reason


def test_file_tool_in_denied_subdirectory(project):
    for tool, args in [("read_file", {"path": "private/secret.txt"}),
                       ("write_file", {"path": "private/new.txt", "content": "x"}),
                       ("list_files", {"path": "private"})]:
        assert decide(project, tool, **args).action == "deny", tool


def test_path_traversal_with_dotdot_is_denied(project):
    for p in ["../outside/secret.txt", "../config/policy.json", "sub/../../outside/secret.txt"]:
        d = decide(project, "read_file", path=p)
        assert d.action == "deny" and "за пределы workspace" in d.reason


def test_absolute_path_is_denied(project):
    d = decide(project, "read_file", path=str(project.root / "outside" / "secret.txt"))
    assert d.action == "deny" and "абсолютный" in d.reason


def test_symlink_escape_is_denied(project):
    os.symlink(project.root / "outside", project.ws / "link_dir")
    os.symlink(project.policy, project.ws / "policy_link.json")
    assert decide(project, "read_file", path="link_dir/secret.txt").action == "deny"
    assert decide(project, "write_file", path="policy_link.json", content="{}").action == "deny"
    assert decide(project, "Bash", command=UNITTEST, cwd="link_dir").action == "deny"


def test_symlink_inside_workspace_to_denied_dir_is_denied(project):
    os.symlink(project.ws / "private", project.ws / "public")
    assert decide(project, "read_file", path="public/secret.txt").action == "deny"


def test_policy_file_cannot_be_modified_by_agent(project):
    """Policy лежит вне workspace: изменить её через инструменты нельзя."""
    before = project.policy.read_text()
    script = [
        reply(call("p1", "write_file", path="../config/policy.json", content="{}"),
              call("p2", "delete_file", path="../config/policy.json")),
        final(),
    ]
    agent, run_id = project.start(script, mode="auto")
    agent.run()
    assert project.policy.read_text() == before
    assert "Отклонено policy" in tool_result(project.store(), run_id, "p1")["error"]



def test_modes(project):
    assert decide(project, "read_file", mode="safe", path="order.py").action == "allow"
    assert decide(project, "write_file", mode="safe", path="order.py", content="").action == "ask"
    assert decide(project, "write_file", mode="auto", path="order.py", content="").action == "allow"
    assert decide(project, "delete_file", mode="auto", path="order.py").action == "ask"
    assert decide(project, "delete_file", mode="safe", path="order.py").action == "ask"



def test_policy_inside_workspace_is_rejected(project):
    inside = project.ws / "policy.json"
    inside.write_text(project.policy.read_text())
    with pytest.raises(PolicyConfigError):
        Policy(inside, project.ws)


def test_allowed_directory_outside_workspace_is_rejected(project):
    data = json.loads(project.policy.read_text())
    data["directories"]["allow"] = ["../outside"]
    project.policy.write_text(json.dumps(data))
    with pytest.raises(PolicyConfigError):
        Policy(project.policy, project.ws)


def test_tool_rechecks_policy_itself(project):
    """Даже если вызвать инструмент в обход runtime, он не выйдет за workspace."""
    with pytest.raises(PolicyViolation):
        policy(project).check_path("../outside/secret.txt")


def test_denied_action_reaches_model_and_does_not_execute(project):
    script = [
        reply(call("d1", "Bash", command=f"{UNITTEST} | cat", cwd=".")),
        reply(call("d2", "write_file", path="order.py", content=FIXED_ORDER)),
        final(),
    ]
    agent, run_id = project.start(script, mode="auto")
    assert agent.run() == "finished"
    store = project.store()
    assert "запрещённый синтаксис" in tool_result(store, run_id, "d1")["error"]
    assert store.latest_call(run_id, "d1")["status"] == "denied"
    assert store.get_run(run_id)["budget_used"]["side_effects"] == 1