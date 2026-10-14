import json
import sqlite3
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    created_at TEXT, updated_at TEXT,
    task TEXT, mode TEXT, phase TEXT, status TEXT, status_detail TEXT,
    workspace TEXT, policy_path TEXT, use_plan INTEGER, settings TEXT,
    plan TEXT, plan_status TEXT, current_task TEXT,
    budget_limits TEXT, budget_used TEXT,
    start_hashes TEXT, final_answer TEXT
);
CREATE TABLE IF NOT EXISTS messages (
    run_id TEXT, seq INTEGER, phase TEXT, body TEXT,
    PRIMARY KEY (run_id, seq)
);
CREATE TABLE IF NOT EXISTS tool_calls (
    run_id TEXT, call_id TEXT, attempt INTEGER,
    msg_seq INTEGER, phase TEXT, task_id TEXT,
    name TEXT, args TEXT, args_raw TEXT, kind TEXT,
    status TEXT, decision TEXT, decision_reason TEXT, user_decision TEXT,
    result TEXT, content TEXT, truncated INTEGER,
    output_reserved INTEGER, done_seq INTEGER,
    created_at TEXT, started_at TEXT, finished_at TEXT,
    PRIMARY KEY (run_id, call_id, attempt)
);
CREATE TABLE IF NOT EXISTS tasks (
    run_id TEXT, task_id TEXT, position INTEGER,
    description TEXT, tools TEXT, depends_on TEXT, verification TEXT,
    status TEXT, evidence TEXT, summary TEXT, start_hashes TEXT,
    started_at TEXT, finished_at TEXT,
    PRIMARY KEY (run_id, task_id)
);
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT, at TEXT, kind TEXT, data TEXT
);
"""

RUN_JSON = {"plan", "budget_limits", "budget_used", "start_hashes", "settings"}
CALL_JSON = {"args", "result"}
TASK_JSON = {"tools", "depends_on", "verification", "evidence", "start_hashes"}


def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _encode(fields: dict, json_keys: set) -> dict:
    return {
        k: (json.dumps(v, ensure_ascii=False) if k in json_keys and v is not None else v)
        for k, v in fields.items()
    }


def _decode(row, json_keys: set) -> dict:
    data = dict(row)
    for key in json_keys:
        if data.get(key) is not None:
            data[key] = json.loads(data[key])
    return data


class Store:
    def __init__(self, db_path):
        path = Path(db_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path.resolve()
        # isolation_level=None: транзакциями управляем сами через tx().
        self.conn = sqlite3.connect(str(path), isolation_level=None)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA synchronous=FULL")
        self.conn.executescript(SCHEMA)
        self._depth = 0

    def close(self) -> None:
        self.conn.close()

    @contextmanager
    def tx(self):
        """Атомарная транзакция. Вложенные tx() становятся частью внешней."""
        if self._depth:
            self._depth += 1
            try:
                yield
            finally:
                self._depth -= 1
            return
        self.conn.execute("BEGIN IMMEDIATE")
        self._depth = 1
        try:
            yield
        except BaseException:
            self.conn.execute("ROLLBACK")
            raise
        else:
            self.conn.execute("COMMIT")
        finally:
            self._depth = 0


    def _insert(self, table: str, fields: dict) -> None:
        cols = ", ".join(fields)
        marks = ", ".join(["?"] * len(fields))
        self.conn.execute(f"INSERT INTO {table} ({cols}) VALUES ({marks})", list(fields.values()))

    def _update(self, table: str, where: dict, fields: dict) -> None:
        sets = ", ".join(f"{k} = ?" for k in fields)
        cond = " AND ".join(f"{k} = ?" for k in where)
        cur = self.conn.execute(
            f"UPDATE {table} SET {sets} WHERE {cond}", [*fields.values(), *where.values()]
        )
        if cur.rowcount != 1:
            raise KeyError(f"{table}: запись {where} не найдена")

    def _select(self, table: str, where: dict, order: str, json_keys: set) -> list[dict]:
        cond = " AND ".join(f"{k} = ?" for k in where) or "1"
        rows = self.conn.execute(
            f"SELECT * FROM {table} WHERE {cond} ORDER BY {order}", list(where.values())
        ).fetchall()
        return [_decode(r, json_keys) for r in rows]


    def create_run(self, **fields) -> str:
        run_id = uuid.uuid4().hex[:12]
        stamp = now()
        self._insert("runs", _encode({"run_id": run_id, "created_at": stamp, "updated_at": stamp, **fields}, RUN_JSON))
        return run_id

    def get_run(self, run_id: str) -> dict:
        rows = self._select("runs", {"run_id": run_id}, "run_id", RUN_JSON)
        if not rows:
            raise KeyError(f"запуск '{run_id}' не найден")
        return rows[0]

    def update_run(self, run_id: str, **fields) -> None:
        self._update("runs", {"run_id": run_id}, _encode({**fields, "updated_at": now()}, RUN_JSON))

    def list_runs(self) -> list[dict]:
        return self._select("runs", {}, "created_at", RUN_JSON)


    def append_message(self, run_id: str, message: dict, phase: str) -> int:
        seq = self.conn.execute(
            "SELECT COALESCE(MAX(seq), 0) + 1 FROM messages WHERE run_id = ?", (run_id,)
        ).fetchone()[0]
        self._insert("messages", {
            "run_id": run_id, "seq": seq, "phase": phase,
            "body": json.dumps(message, ensure_ascii=False),
        })
        return seq

    def messages(self, run_id: str) -> list[tuple[int, dict]]:
        rows = self.conn.execute(
            "SELECT seq, body FROM messages WHERE run_id = ? ORDER BY seq", (run_id,)
        ).fetchall()
        return [(r["seq"], json.loads(r["body"])) for r in rows]


    def insert_call(self, **fields) -> None:
        self._insert("tool_calls", _encode({"created_at": now(), **fields}, CALL_JSON))

    def update_call(self, run_id: str, call_id: str, attempt: int, **fields) -> None:
        self._update(
            "tool_calls",
            {"run_id": run_id, "call_id": call_id, "attempt": attempt},
            _encode(fields, CALL_JSON),
        )

    def latest_call(self, run_id: str, call_id: str) -> dict | None:
        rows = self._select(
            "tool_calls", {"run_id": run_id, "call_id": call_id}, "attempt DESC", CALL_JSON
        )
        return rows[0] if rows else None

    def calls(self, run_id: str, **where) -> list[dict]:
        return self._select("tool_calls", {"run_id": run_id, **where}, "rowid", CALL_JSON)

    def next_done_seq(self, run_id: str) -> int:
        return self.conn.execute(
            "SELECT COALESCE(MAX(done_seq), 0) + 1 FROM tool_calls WHERE run_id = ?", (run_id,)
        ).fetchone()[0]


    def insert_task(self, **fields) -> None:
        self._insert("tasks", _encode(fields, TASK_JSON))

    def update_task(self, run_id: str, task_id: str, **fields) -> None:
        self._update("tasks", {"run_id": run_id, "task_id": task_id}, _encode(fields, TASK_JSON))

    def tasks(self, run_id: str) -> list[dict]:
        return self._select("tasks", {"run_id": run_id}, "position", TASK_JSON)

    def get_task(self, run_id: str, task_id: str) -> dict | None:
        rows = self._select("tasks", {"run_id": run_id, "task_id": task_id}, "position", TASK_JSON)
        return rows[0] if rows else None


    def add_event(self, run_id: str, kind: str, **data) -> None:
        self._insert("events", {
            "run_id": run_id, "at": now(), "kind": kind,
            "data": json.dumps(data, ensure_ascii=False),
        })

    def events(self, run_id: str) -> list[dict]:
        rows = self.conn.execute(
            "SELECT * FROM events WHERE run_id = ? ORDER BY id", (run_id,)
        ).fetchall()
        return [{**dict(r), "data": json.loads(r["data"])} for r in rows]
