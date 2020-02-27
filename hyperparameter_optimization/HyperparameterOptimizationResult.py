from typing import NamedTuple, Dict, Any


class HyperparameterOptimizationResult(NamedTuple):  # inherit from typing.NamedTuple
    parameters: Dict[str, Any]
    score: float