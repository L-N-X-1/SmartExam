"""Agents package helpers.

Provides a canonical `STAGES` list, a `load_agents()` runtime loader that attempts
to import each stage module and extract a `process` callable, and `mock_agents()`
which returns lightweight mock callables useful for local testing.
"""
from typing import Callable, Dict
import importlib

STAGES = ["remember", "understand", "apply", "analyze", "evaluate", "create"]


def _make_stub(name: str) -> Callable:
    def stub(payload, quota=None):
        raise NotImplementedError(
            f"Agent '{name}' has no implementation. "
            f"Provide a `process(payload, quota)` in agents/{name}_agent.py"
        )

    return stub


def load_agents() -> Dict[str, Callable]:
    """Dynamically import stage agent modules and return mapping stage->callable.

    If a module or `process` callable isn't found, a stub raising
    `NotImplementedError` is returned for that stage.
    """
    agents: Dict[str, Callable] = {}
    for stage in STAGES:
        module_name = f"{__name__}.{stage}_agent"
        try:
            mod = importlib.import_module(module_name)
            if hasattr(mod, "process") and callable(getattr(mod, "process")):
                agents[stage] = getattr(mod, "process")
            else:
                # try to find any object with a `process` method as a fallback
                proc = None
                for attr in dir(mod):
                    obj = getattr(mod, attr)
                    if hasattr(obj, "process") and callable(getattr(obj, "process")):
                        proc = getattr(obj, "process")
                        break
                agents[stage] = proc or _make_stub(stage)
        except Exception:
            agents[stage] = _make_stub(stage)
    return agents


def mock_agents() -> Dict[str, Callable]:
    """Return lightweight mock agents for quick coordinator testing.

    Each mock returns `quota` placeholder strings when called.
    """

    def make_mock(stage: str) -> Callable:
        def mock(payload, quota=None):
            q = int(quota or payload.get("quota") or 1)
            return [f"{stage}_mock_{i+1}" for i in range(q)]

        return mock

    return {stage: make_mock(stage) for stage in STAGES}


__all__ = ["STAGES", "load_agents", "mock_agents"]
