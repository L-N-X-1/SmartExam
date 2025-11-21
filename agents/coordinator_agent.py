import json
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple


class CoordinatorAgent:
    """Orchestrates Bloom-taxonomy agents in a simple sequential workflow.

    The coordinator expects `agents` to be one of:
      - a dict mapping agent name -> callable (callable(payload, quota) or callable(payload))
      - a sequence of objects that either implement `.process(payload, quota)` or are callables

    The coordinator provides:
      - `define_workflow()` — returns ordered stages and default distribution
      - `define_json_format()` — canonical input/output schema
      - `call_agents(input_data)` — runs agents in order distributing quotas
    """

    def __init__(self, agents: Any):
        self.agents = agents

    def define_workflow(self) -> Dict[str, Any]:
        """Return a simple workflow definition and default distribution across Bloom levels.

        The workflow is an ordered list of stages (remember -> create). Each stage has a
        recommended fraction of the total `num_items` (questions, tasks, etc.).
        """
        stages = [
            "remember",
            "understand",
            "apply",
            "analyze",
            "evaluate",
            "create",
        ]
        # Default distribution (sums to 1.0)
        distribution = {
            "remember": 0.15,
            "understand": 0.20,
            "apply": 0.25,
            "analyze": 0.15,
            "evaluate": 0.15,
            "create": 0.10,
        }
        return {"stages": stages, "distribution": distribution}

    def define_json_format(self) -> Dict[str, Any]:
        """Return canonical input/output JSON formats for the coordinator.

        Input keys:
          - `topic` (str)
          - `learning_objectives` (list[str])
          - `num_items` (int): total number of questions/tasks to generate
          - `question_types` (optional list[str])
          - `difficulty_levels` (optional list[str])
          - `metadata` (optional dict)

        Output:
          - `by_stage`: mapping stage -> list of generated items
          - `aggregated`: flattened list of all items
          - `errors`: any stage-level errors
        """
        return {
            "input": {
                "topic": "",
                "learning_objectives": [],
                "num_items": 10,
                "question_types": [],
                "difficulty_levels": [],
                "metadata": {},
            },
            "output": {"by_stage": {}, "aggregated": [], "errors": {}},
        }

    def _resolve_agents(self) -> List[Tuple[str, Callable]]:
        """Normalize `self.agents` into a list of (name, callable) pairs.

        The callable is expected to accept either `(payload, quota)` or `(payload,)`.
        """
        resolved: List[Tuple[str, Callable]] = []
        if isinstance(self.agents, dict):
            for name, obj in self.agents.items():
                if hasattr(obj, "process") and callable(getattr(obj, "process")):
                    resolved.append((name, getattr(obj, "process")))
                elif callable(obj):
                    resolved.append((name, obj))
                else:
                    raise TypeError(f"Agent '{name}' is not callable nor has .process()")
        elif isinstance(self.agents, Sequence):
            for idx, obj in enumerate(self.agents):
                name = getattr(obj, "name", f"agent_{idx}")
                if hasattr(obj, "process") and callable(getattr(obj, "process")):
                    resolved.append((name, getattr(obj, "process")))
                elif callable(obj):
                    resolved.append((name, obj))
                else:
                    raise TypeError(f"Agent at index {idx} is not callable nor has .process()")
        else:
            raise TypeError("`agents` must be a dict or a sequence of callables/objects")
        return resolved

    def _distribute_quota(self, total: int, distribution: Dict[str, float], stages: List[str]) -> Dict[str, int]:
        """Convert fractional distribution to integer quotas that sum to `total`.

        Strategy: floor each fractional allocation, then distribute remainder left-to-right.
        """
        quotas: Dict[str, int] = {}
        floats = {s: distribution.get(s, 0.0) * total for s in stages}
        for s, f in floats.items():
            quotas[s] = int(f)
        assigned = sum(quotas.values())
        remainder = total - assigned
        # Distribute remainder by largest fractional parts
        fractional_parts = sorted(((s, floats[s] - quotas[s]) for s in stages), key=lambda x: -x[1])
        i = 0
        while remainder > 0 and i < len(fractional_parts):
            quotas[fractional_parts[i][0]] += 1
            remainder -= 1
            i += 1
            if i == len(fractional_parts):
                i = 0
        return quotas

    def call_agents(self, input_data: Dict[str, Any], override_distribution: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """Call Bloom agents sequentially according to the workflow.

        `input_data` must include `num_items` (int). Returns the aggregated results and errors.
        """
        workflow = self.define_workflow()
        stages: List[str] = workflow["stages"]
        distribution = override_distribution or workflow["distribution"]
        num_items = int(input_data.get("num_items", 0))

        by_stage: Dict[str, List[Any]] = {s: [] for s in stages}
        errors: Dict[str, str] = {}

        if num_items <= 0:
            raise ValueError("`num_items` must be a positive integer in input_data")

        quotas = self._distribute_quota(num_items, distribution, stages)

        resolved_agents = self._resolve_agents()
        # Map agent names to callables for quick lookup
        agent_map = {name: func for name, func in resolved_agents}

        for stage in stages:
            quota = quotas.get(stage, 0)
            if quota <= 0:
                continue
            agent_callable = agent_map.get(stage)
            if agent_callable is None:
                errors[stage] = "No agent available for this stage"
                continue

            payload = dict(input_data)  # shallow copy
            payload.update({"stage": stage, "quota": quota})

            try:
                # Try calling with (payload, quota) signature, else fallback to (payload,)
                try:
                    result = agent_callable(payload, quota)
                except TypeError:
                    result = agent_callable(payload)

                # Expecting result to be iterable (list of items) or dict with `items` key
                if isinstance(result, dict) and "items" in result:
                    items = result["items"]
                elif isinstance(result, (list, tuple)):
                    items = list(result)
                else:
                    items = [result]

                by_stage[stage].extend(items)
            except Exception as exc:  # keep broad to avoid crash across stages
                errors[stage] = str(exc)

        # Aggregate and return
        aggregated = []
        for s in stages:
            aggregated.extend(by_stage.get(s, []))

        output = {"by_stage": by_stage, "aggregated": aggregated, "errors": errors}
        return output


# Example usage with mock agents:
# def mock_agent(payload, quota):
#     return [f"{payload['stage']}_q{i+1}" for i in range(payload['quota'])]
# agents = { 'remember': mock_agent, 'understand': mock_agent, 'apply': mock_agent,
#            'analyze': mock_agent, 'evaluate': mock_agent, 'create': mock_agent }
# coord = CoordinatorAgent(agents)
# input_data = coord.define_json_format()['input']
# input_data.update({'topic': 'Photosynthesis', 'num_items': 12})
# results = coord.call_agents(input_data)