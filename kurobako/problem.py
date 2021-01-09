import abc
import copy
import enum
import json
from lupa import LuaRuntime
import numpy as np
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Union

Self = Any


class Range(object, metaclass=abc.ABCMeta):
    @property
    @abc.abstractmethod
    def low(self) -> float:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def high(self) -> float:
        raise NotImplementedError

    @abc.abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        raise NotImplementedError

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> Any:
        if d["type"] == "CONTINUOUS":
            return ContinuousRange.from_dict(d)
        elif d["type"] == "DISCRETE":
            return DiscreteRange.from_dict(d)
        elif d["type"] == "CATEGORICAL":
            return CategoricalRange.from_dict(d)
        else:
            raise ValueError("Unknown range: {}".format(d))


class ContinuousRange(Range):
    def __init__(self, low: float, high: float):
        self._low = low
        self._high = high

    @property
    def low(self) -> float:
        return self._low

    @property
    def high(self) -> float:
        return self._high

    def to_dict(self) -> Dict[str, Any]:
        d = {"type": "CONTINUOUS"}  # type: Dict[str, Any]
        if np.isfinite(self._low):
            d["low"] = self._low
        if np.isfinite(self._high):
            d["high"] = self._high
        return d

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> Any:
        if "low" not in d:
            d["low"] = float("-inf")
        if "high" not in d:
            d["high"] = float("inf")
        return ContinuousRange(low=d["low"], high=d["high"])


class DiscreteRange(Range):
    def __init__(self, low: int, high: int):
        self._low = int(low)
        self._high = int(high)

    @property
    def low(self) -> float:
        return self._low

    @property
    def high(self) -> float:
        return self._high

    def to_dict(self) -> Dict[str, Any]:
        return {"type": "DISCRETE", "low": self._low, "high": self._high}

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> Any:
        return DiscreteRange(low=d["low"], high=d["high"])


class CategoricalRange(Range):
    def __init__(self, choices: List[str]):
        self.choices = choices

    @property
    def low(self) -> float:
        return 0

    @property
    def high(self) -> float:
        return len(self.choices)

    def to_dict(self) -> Dict[str, Any]:
        return {"type": "CATEGORICAL", "choices": self.choices}

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> Any:
        return CategoricalRange(choices=d["choices"])


class Distribution(enum.Enum):
    UNIFORM = 0
    LOG_UNIFORM = 1

    def to_str(self) -> str:
        if self == Distribution.UNIFORM:
            return "UNIFORM"
        else:
            return "LOG_UNIFORM"

    @staticmethod
    def from_str(s: str) -> Any:
        if s == "UNIFORM":
            return Distribution.UNIFORM
        elif s == "LOG_UNIFORM":
            return Distribution.LOG_UNIFORM
        else:
            raise ValueError


class Var(object):
    """A variable in a domain."""

    def __init__(
        self,
        name: str,
        range: Range = ContinuousRange(float("-inf"), float("inf")),
        distribution: Distribution = Distribution.UNIFORM,
        constraint: Optional[str] = None,
    ):
        self.name = name
        self.range = range
        self.distribution = distribution
        self.constraint = constraint

    def is_constraint_satisfied(self, vars: List[Self], vals: List[Optional[float]]) -> bool:
        if self.constraint is None:
            return True

        lua = LuaRuntime()
        for var, val in zip(vars, vals):
            if val is None:
                continue

            if isinstance(var.range, CategoricalRange):
                val = var.range.choices[int(val)]  # type: ignore

            lua.execute("{} = {}".format(var.name, repr(val)))

        return lua.eval(self.constraint)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "range": self.range.to_dict(),
            "distribution": self.distribution.to_str(),
            "constraint": self.constraint,
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> Any:
        return Var(
            name=d["name"],
            range=Range.from_dict(d["range"]),
            distribution=Distribution.from_str(d["distribution"]),
            constraint=d["constraint"] if "constraint" in d else None,
        )


class ProblemSpec(object):
    """Problem specification."""

    def __init__(
        self,
        name: str,
        params: List[Var],
        values: List[Var],
        attrs: Dict[str, str] = {},
        steps: Union[int, List[int]] = 1,
        reference_point: Optional[List[float]] = None,
    ):
        self.name = name
        self.attrs = copy.deepcopy(attrs)
        self.params = params
        self.values = values
        self.steps = steps
        self.reference_point = reference_point

    @property
    def last_step(self) -> int:
        if isinstance(self.steps, int):
            return self.steps
        else:
            return self.steps[-1]

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> Any:
        """Creates a `ProblemSpec` instance from the given dictionary."""

        return ProblemSpec(
            name=d["name"],
            attrs=d["attrs"],
            params=[Var.from_dict(v) for v in d["params_domain"]],
            values=[Var.from_dict(v) for v in d["values_domain"]],
            steps=d["steps"],
            reference_point=d["reference_point"],
        )

    def to_dict(self) -> Dict[str, Any]:
        """Converts this instance to dictionary format."""

        return {
            "name": self.name,
            "attrs": self.attrs,
            "params_domain": [v.to_dict() for v in self.params],
            "values_domain": [v.to_dict() for v in self.values],
            "steps": self.steps,
            "reference_point": self.reference_point,
        }


class Evaluator(object):
    @abc.abstractmethod
    def evaluate(self, next_step: int) -> List[float]:
        raise NotImplementedError

    def current_step(self) -> int:
        raise NotImplementedError


class Problem(object):
    @abc.abstractmethod
    def create_evaluator(self, params: List[Optional[float]]) -> Optional[Evaluator]:
        raise NotImplementedError


class ProblemFactory(object):
    @abc.abstractmethod
    def specification(self) -> ProblemSpec:
        raise NotImplementedError

    @abc.abstractmethod
    def create_problem(self, seed: int) -> Problem:
        raise NotImplementedError


class ProblemRunner(object):
    def __init__(self, factory: ProblemFactory):
        self._factory = factory
        self._problems = {}  # type: Dict[int, Problem]
        self._evaluators = {}  # type: Dict[int, Evaluator]

    def run(self):
        self._cast_problem_spec()

        while self._run_once():
            pass

    def _run_once(self) -> bool:
        message = self._recv_message()
        if message is None:
            return False

        message_type = message["type"]
        if message_type == "CREATE_PROBLEM_CAST":
            self._handle_create_problem_cast(message)
        elif message_type == "DROP_PROBLEM_CAST":
            self._handle_drop_problem_cast(message)
        elif message_type == "CREATE_EVALUATOR_CALL":
            self._handle_create_evaluator_call(message)
        elif message_type == "DROP_EVALUATOR_CAST":
            self._handle_drop_evaluator_cast(message)
        elif message_type == "EVALUATE_CALL":
            self._handle_evaluate_call(message)
        else:
            raise ValueError("Unexpected message: {}".format(message))

        return True

    def _handle_create_problem_cast(self, message):
        problem_id = message["problem_id"]
        random_seed = message["random_seed"]
        assert problem_id not in self._problems

        problem = self._factory.create_problem(random_seed)
        self._problems[problem_id] = problem

    def _handle_drop_problem_cast(self, message):
        problem_id = message["problem_id"]
        del self._problems[problem_id]

    def _handle_create_evaluator_call(self, message):
        problem_id = message["problem_id"]
        evaluator_id = message["evaluator_id"]
        params = message["params"]
        assert evaluator_id not in self._evaluators

        problem = self._problems[problem_id]
        evaluator = problem.create_evaluator(params)
        if evaluator is None:
            self._send_message({"type": "ERROR_REPLY", "kind": "UNEVALABLE_PARAMS"})
        else:
            self._evaluators[evaluator_id] = evaluator
            self._send_message({"type": "CREATE_EVALUATOR_REPLY"})

    def _handle_drop_evaluator_cast(self, message):
        evaluator_id = message["evaluator_id"]
        del self._evaluators[evaluator_id]

    def _handle_evaluate_call(self, message):
        evaluator_id = message["evaluator_id"]
        next_step = message["next_step"]

        evaluator = self._evaluators[evaluator_id]
        values = evaluator.evaluate(next_step)
        current_step = evaluator.current_step()

        self._send_message(
            {"type": "EVALUATE_REPLY", "current_step": current_step, "values": values}
        )

    def _cast_problem_spec(self):
        spec = self._factory.specification()
        self._send_message({"type": "PROBLEM_SPEC_CAST", "spec": spec.to_dict()})

    def _send_message(self, message: Dict[str, Any]):
        print(json.dumps(message))

    def _recv_message(self) -> Optional[Dict[str, Any]]:
        try:
            message = input()
            return json.loads(message)
        except Exception:
            return None
