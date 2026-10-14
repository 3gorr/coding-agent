import argparse
import json
import sys

from .budget import format_budget, load_budget_config
from .crash import CrashPoints
from .llm import LLMError, OpenAIChatClient, ScriptedLLM
from .loop import Agent, UserError
from .plan import render_plan
from .policy import PolicyConfigError
from .store import Store
from .tasks import checklist


def _out(text: str) -> None:
    print(text, flush=True)


def _llm(args):
    if args.scripted:
        return ScriptedLLM.from_file(args.scripted)
    return OpenAIChatClient()


def _agent(store, args, run_id: str) -> Agent:
    return Agent(store, run_id, _llm(args), crash=CrashPoints.from_env(), out=_out)


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m agent", description="Учебный coding agent с tool calling")
    p.add_argument("--db", default=".agent/runs.db", help="файл SQLite с состоянием запусков")
    p.add_argument("--scripted", metavar="FILE",
                   help="демо без LLM: ответы модели берутся из JSON-сценария")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="новый запуск агента")
    r.add_argument("task", help="задача для агента")
    r.add_argument("--mode", choices=["safe", "auto"], default="safe")
    r.add_argument("--no-plan", action="store_true", help="без Plan Mode и задач")
    r.add_argument("--workspace", default="workspace")
    r.add_argument("--policy", default="config/policy.json")
    r.add_argument("--budget", default="config/budget.json")
    r.add_argument("--bash-timeout", type=float, default=30.0)

    for name, help_text in [("resume", "продолжить запуск"), ("status", "состояние запуска"),
                            ("tasks", "чеклист задач"), ("log", "журнал tool calls")]:
        sub.add_parser(name, help=help_text).add_argument("run_id")

    a = sub.add_parser("approve", help="разрешить ожидающее действие и продолжить")
    a.add_argument("run_id")
    a.add_argument("--call", help="id вызова (по умолчанию первый ожидающий)")

    d = sub.add_parser("deny", help="отклонить ожидающее действие и продолжить")
    d.add_argument("run_id")
    d.add_argument("--call")
    d.add_argument("--reason", default="без объяснения")

    pl = sub.add_parser("plan", help="показать, утвердить или отклонить план")
    pl.add_argument("run_id")
    group = pl.add_mutually_exclusive_group()
    group.add_argument("--approve", action="store_true")
    group.add_argument("--reject", action="store_true")
    pl.add_argument("--reason", default="без объяснения")

    rc = sub.add_parser("recover", help="решение по вызову, прерванному сбоем")
    rc.add_argument("run_id")
    act = rc.add_mutually_exclusive_group(required=True)
    act.add_argument("--retry", action="store_true", help="повторить (если инструмент повторяем)")
    act.add_argument("--fail", action="store_true", help="считать шаг неуспешным и продолжить")
    act.add_argument("--abort", action="store_true", help="завершить запуск с ошибкой")
    rc.add_argument("--call")

    sub.add_parser("list", help="все запуски")
    return p


def main(argv=None) -> int:
    args = _parser().parse_args(argv)
    store = Store(args.db)
    try:
        return _dispatch(store, args)
    except (UserError, PolicyConfigError, LLMError, KeyError, ValueError) as exc:
        print(f"Ошибка: {exc}", file=sys.stderr)
        return 1
    finally:
        store.close()


def _dispatch(store, args) -> int:
    if args.cmd == "run":
        llm = _llm(args)
        run_id = Agent.create_run(
            store, args.task, workspace=args.workspace, policy_path=args.policy,
            mode=args.mode, use_plan=not args.no_plan,
            budget=load_budget_config(args.budget), bash_timeout=args.bash_timeout,
        )
        _out(f"run_id: {run_id}")
        Agent(store, run_id, llm, crash=CrashPoints.from_env(), out=_out).run()
        return 0

    if args.cmd == "list":
        for run in store.list_runs():
            _out(f"{run['run_id']}  {run['created_at']}  {run['status']:<24} {run['mode']:<5} "
                 f"{run['task'][:60]}")
        return 0

    run = store.get_run(args.run_id)

    if args.cmd == "status":
        _out(f"run_id: {run['run_id']}\nзадача: {run['task']}\nрежим: {run['mode']}  фаза: {run['phase']}"
             f"  статус: {run['status']}" + (f" ({run['status_detail']})" if run["status_detail"] else ""))
        _out(f"бюджет: {format_budget(run)}")
        tasks = store.tasks(run["run_id"])
        if tasks:
            _out("Задачи:\n" + checklist(tasks))
        if run["status"] in ("awaiting_approval", "awaiting_plan_approval", "awaiting_recovery"):
            Agent(store, run["run_id"], None, out=_out)._announce(run)
        return 0

    if args.cmd == "tasks":
        tasks = store.tasks(run["run_id"])
        if not tasks:
            _out("У запуска нет задач (план ещё не утверждён или запуск без Plan Mode).")
            return 0
        _out(checklist(tasks))
        for task in tasks:
            if task["evidence"]:
                _out(f"  {task['task_id']} подтверждено: {json.dumps(task['evidence'], ensure_ascii=False)}")
        return 0

    if args.cmd == "log":
        for row in store.calls(run["run_id"]):
            _out(f"{row['call_id']}#{row['attempt']}  {row['phase']:<9} task={row['task_id'] or '-':<4} "
                 f"{row['name']:<12} {row['status']:<16} policy={row['decision'] or '-'} "
                 f"user={row['user_decision'] or '-'}  args={(row['args_raw'] or '')[:70]}")
        return 0

    if args.cmd == "plan" and not (args.approve or args.reject):
        _out(render_plan(run["plan"]) if run["plan"] else "План ещё не сдан.")
        _out(f"статус плана: {run['plan_status'] or '-'}")
        return 0

    agent = _agent(store, args, run["run_id"])
    if args.cmd == "approve":
        agent.approve(args.call)
    elif args.cmd == "deny":
        agent.deny(args.call, args.reason)
    elif args.cmd == "plan":
        agent.approve_plan() if args.approve else agent.reject_plan(args.reason)
    elif args.cmd == "recover":
        agent.recover("retry" if args.retry else "fail" if args.fail else "abort", args.call)
    agent.resume()
    return 0
