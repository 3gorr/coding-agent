import os
import sys

CRASH_POINTS = ("after_model_charge", "after_start", "after_exec")


class SimulatedCrash(BaseException):
    """Имитация падения процесса в заданной точке."""


class CrashPoints:
    def __init__(self, at: str | None = None, mode: str = "raise", nth: int = 1):
        if at is not None and at not in CRASH_POINTS:
            raise ValueError(f"неизвестная точка отказа {at!r}, доступны: {CRASH_POINTS}")
        self.at = at
        self.mode = mode
        self.nth = nth
        self._hits = 0

    @classmethod
    def from_env(cls) -> "CrashPoints":
        return cls(os.environ.get("AGENT_CRASH_AT") or None, mode="exit",
                   nth=int(os.environ.get("AGENT_CRASH_NTH", "1")))

    def __call__(self, name: str) -> None:
        if self.at != name:
            return
        self._hits += 1
        if self._hits < self.nth:
            return
        if self.mode == "exit":
            print(f"[crash] имитация падения процесса в точке {name}", flush=True)
            sys.stderr.flush()
            os._exit(137)
        raise SimulatedCrash(name)
