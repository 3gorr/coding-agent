import json
from pathlib import Path

from .tools.base import SIDE_EFFECT_KINDS

LIMIT_KEYS = ("model_requests", "tool_calls", "side_effects", "output_bytes")
DEFAULT_CONFIG = {
    "limits": {"model_requests": 40, "tool_calls": 80, "side_effects": 20, "output_bytes": 300_000},
    "per_call_output_bytes": 20_000,
}


class BudgetExceeded(Exception):
    """Нужный лимит исчерпан - действие не начинается."""


def load_budget_config(path=None) -> dict:
    config = json.loads(json.dumps(DEFAULT_CONFIG))
    if path is not None and Path(path).exists():
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        config["limits"].update(data.get("limits", {}))
        if "per_call_output_bytes" in data:
            config["per_call_output_bytes"] = data["per_call_output_bytes"]
    for key in LIMIT_KEYS:
        if not isinstance(config["limits"][key], int) or config["limits"][key] < 0:
            raise ValueError(f"лимит {key} должен быть неотрицательным целым")
    return config


def zero_used() -> dict:
    return {key: 0 for key in LIMIT_KEYS}


def remaining(run: dict) -> dict:
    limits, used = run["budget_limits"]["limits"], run["budget_used"]
    return {key: limits[key] - used[key] for key in LIMIT_KEYS}


def check_batch(run: dict, kinds: list[str]) -> str | None:
    """Хватит ли остатка на ВСЮ пачку вызовов. None - хватает, иначе причина."""
    if not kinds:
        return None
    rem = remaining(run)
    side = sum(1 for k in kinds if k in SIDE_EFFECT_KINDS)
    problems = []
    if len(kinds) > rem["tool_calls"]:
        problems.append(f"нужно tool_calls={len(kinds)}, осталось {rem['tool_calls']}")
    if side > rem["side_effects"]:
        problems.append(f"нужно side_effects={side}, осталось {rem['side_effects']}")
    if rem["output_bytes"] <= 0:
        problems.append("бюджет output_bytes исчерпан")
    return "; ".join(problems) or None


def charge_model_request(store, run_id: str) -> None:
    """Списать один запрос к модели (атомарно) или бросить BudgetExceeded."""
    with store.tx():
        run = store.get_run(run_id)
        used = dict(run["budget_used"])
        limit = run["budget_limits"]["limits"]["model_requests"]
        if used["model_requests"] >= limit:
            raise BudgetExceeded(f"исчерпан лимит model_requests ({limit})")
        used["model_requests"] += 1
        store.update_run(run_id, budget_used=used)


def charge_call_start(run: dict, kind: str) -> tuple[dict, int]:
    """Новые значения used и размер резерва вывода для одного вызова.

    Ничего не сохраняет: вызывающий код пишет результат в БД в одной
    транзакции со статусом вызова.
    """
    rem = remaining(run)
    if rem["tool_calls"] < 1:
        raise BudgetExceeded("исчерпан лимит tool_calls")
    if kind in SIDE_EFFECT_KINDS and rem["side_effects"] < 1:
        raise BudgetExceeded("исчерпан лимит side_effects")
    if rem["output_bytes"] < 1:
        raise BudgetExceeded("исчерпан лимит output_bytes")
    reserve = min(run["budget_limits"]["per_call_output_bytes"], rem["output_bytes"])
    used = dict(run["budget_used"])
    used["tool_calls"] += 1
    if kind in SIDE_EFFECT_KINDS:
        used["side_effects"] += 1
    used["output_bytes"] += reserve
    return used, reserve


def settle_output(run: dict, reserved: int, actual: int) -> dict:
    """После выполнения: заменить резерв на фактический размер результата."""
    used = dict(run["budget_used"])
    used["output_bytes"] += actual - reserved
    return used


def format_budget(run: dict) -> str:
    limits, used = run["budget_limits"]["limits"], run["budget_used"]
    return ", ".join(f"{k}: {used[k]}/{limits[k]}" for k in LIMIT_KEYS)
