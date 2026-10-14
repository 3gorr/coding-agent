import json
import shutil
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from agent.budget import load_budget_config
from agent.crash import CrashPoints
from agent.llm import ScriptedLLM
from agent.loop import Agent
from agent.snapshot import hash_tree
from agent.store import Store

UNITTEST = "python3 -m unittest -v"
TASK = ("Исправь расчёт итоговой стоимости заказа, чтобы тесты проходили. Тесты не меняй. "
        "Сам найди нужные файлы, внеси исправление и запусти тесты.")

FIXED_ORDER = (
    "def calculate_total(prices: list[float], discount_percent: float) -> float:\n"
    "    if not 0 <= discount_percent <= 100:\n"
    "        raise ValueError(\"discount_percent must be between 0 and 100\")\n"
    "    return sum(prices) * (1 - discount_percent / 100)\n"
)

TEST_POLICY = {
    "commands": {
        "allow": [UNITTEST, "ls", "pwd", "seq 100000", "sleep 5", "rm -rf ."],
        "deny": ["rm -rf ."],  # есть и в allow: проверяем, что запрет важнее
    },
    "directories": {"allow": ["."], "deny": ["private"]},
}



def call(cid: str, name: str, **args) -> dict:
    return {"id": cid, "type": "function",
            "function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False)}}


def reply(*calls, text: str = "") -> dict:
    return {"content": text, "tool_calls": list(calls)}


def final(text: str = "Готово") -> dict:
    return {"content": text, "tool_calls": []}


class Project:
    def __init__(self, root: Path):
        self.root = root
        self.ws = root / "ws"
        shutil.copytree(REPO / "workspace", self.ws, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        (self.ws / "private" / "secret.txt").write_text("секрет")
        (root / "outside").mkdir()
        (root / "outside" / "secret.txt").write_text("файл вне workspace")
        (root / "config").mkdir()
        self.policy = root / "config" / "policy.json"
        self.policy.write_text(json.dumps(TEST_POLICY), encoding="utf-8")
        self.db = root / "state" / "runs.db"
        self.lines: list[str] = []

    @property
    def output(self) -> str:
        return "\n".join(self.lines)

    def budget(self, **limits) -> dict:
        cfg = load_budget_config(None)
        per_call = limits.pop("per_call_output_bytes", None)
        cfg["limits"].update(limits)
        if per_call is not None:
            cfg["per_call_output_bytes"] = per_call
        return cfg

    def start(self, script, *, mode="auto", use_plan=False, budget=None, crash_at=None,
              crash_nth=1, task=TASK, bash_timeout=30.0):
        store = Store(self.db)
        run_id = Agent.create_run(store, task, workspace=self.ws, policy_path=self.policy,
                                  mode=mode, use_plan=use_plan, budget=budget,
                                  bash_timeout=bash_timeout)
        return self.agent(run_id, script, crash_at=crash_at, crash_nth=crash_nth), run_id

    def agent(self, run_id, script, crash_at=None, crash_nth=1) -> Agent:
        """Новый экземпляр агента = "новый процесс" поверх той же БД."""
        return Agent(Store(self.db), run_id, ScriptedLLM(script),
                     crash=CrashPoints(crash_at, nth=crash_nth), out=self.lines.append)

    def store(self) -> Store:
        return Store(self.db)

    def hashes(self) -> dict:
        return hash_tree(self.ws)


@pytest.fixture
def project(tmp_path) -> Project:
    return Project(tmp_path)


def tool_messages(store, run_id) -> list[dict]:
    return [m for _, m in store.messages(run_id) if m["role"] == "tool"]


def tool_result(store, run_id, call_id) -> dict:
    """Последнее сообщение role=tool для данного call_id, распарсенное из JSON."""
    found = [m for m in tool_messages(store, run_id) if m["tool_call_id"] == call_id]
    assert found, f"нет результата для {call_id}"
    return json.loads(found[-1]["content"])



PLAN = {
    "goal": "Исправить calculate_total так, чтобы тесты проходили, не меняя тесты",
    "steps": [
        {"id": "T1", "description": "Прочитать order.py и test_order.py", "tools": ["read_file"],
         "depends_on": [], "verification": {"type": "files_read", "paths": ["order.py", "test_order.py"]}},
        {"id": "T2", "description": "Исправить расчёт скидки в order.py", "tools": ["write_file"],
         "depends_on": ["T1"], "verification": {"type": "file_changed", "path": "order.py"}},
        {"id": "T3", "description": "Запустить тесты", "tools": ["Bash"],
         "depends_on": ["T2"], "verification": {"type": "command_succeeds", "command": UNITTEST, "cwd": "."}},
        {"id": "T4", "description": "Убедиться, что тесты не изменены", "tools": [],
         "depends_on": ["T3"], "verification": {"type": "file_unchanged", "path": "test_order.py"}},
    ],
    "risks": ["Сравнение float: 150 * 0.9 должно дать ровно 135.0"],
    "assumptions": ["discount_percent - процент от 0 до 100 включительно"],
    "done_criteria": "python3 -m unittest -v завершается с кодом 0, test_order.py не изменён",
}


def planning_script(plan=None) -> list:
    """Фаза планирования: чтение, попытки изменить файлы (должны быть отклонены), план."""
    return [
        reply(call("p1", "list_files", path=".")),
        reply(call("p2", "read_file", path="order.py"),
              call("p3", "write_file", path="order.py", content=FIXED_ORDER),
              call("p4", "Bash", command=UNITTEST, cwd="."),
              call("p5", "delete_file", path="test_order.py")),
        reply(call("p6", "submit_plan", **(plan or PLAN))),
    ]


def execution_script() -> list:
    """Фаза исполнения по задачам T1..T4."""
    return [
        reply(call("e1", "read_file", path="order.py"), call("e2", "read_file", path="test_order.py")),
        reply(call("e3", "finish_task", task_id="T1", summary="прочитал")),
        reply(call("e4", "write_file", path="order.py", content=FIXED_ORDER)),
        reply(call("e5", "finish_task", task_id="T2", summary="исправил")),
        reply(call("e6", "Bash", command=UNITTEST, cwd=".")),
        reply(call("e7", "finish_task", task_id="T3", summary="тесты прошли")),
        reply(call("e8", "finish_task", task_id="T4", summary="тесты не тронуты")),
        final("Готово: расчёт исправлен, 2 теста проходят."),
    ]
