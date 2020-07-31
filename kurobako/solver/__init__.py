import abc
import copy
import enum
import json
import numpy as np
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Set

from kurobako.problem import ProblemSpec


class Capability(enum.Enum):
    UNIFORM_CONTINUOUS = 0
    UNIFORM_DISCRETE = 1
    LOG_UNIFORM_CONTINUOUS = 2
    LOG_UNIFORM_DISCRETE = 3
    CATEGORICAL = 4
    CONDITIONAL = 5
    MULTI_OBJECTIVE = 6
    CONCURRENT = 7

    def to_str(self) -> str:
        if self == Capability.UNIFORM_CONTINUOUS:
            return "UNIFORM_CONTINUOUS"
        elif self == Capability.UNIFORM_DISCRETE:
            return "UNIFORM_DISCRETE"
        elif self == Capability.LOG_UNIFORM_CONTINUOUS:
            return "LOG_UNIFORM_CONTINUOUS"
        elif self == Capability.LOG_UNIFORM_DISCRETE:
            return "LOG_UNIFORM_DISCRETE"
        elif self == Capability.CATEGORICAL:
            return "CATEGORICAL"
        elif self == Capability.CONDITIONAL:
            return "CONDITIONAL"
        elif self == Capability.MULTI_OBJECTIVE:
            return "MULTI_OBJECTIVE"
        else:
            assert self == Capability.CONCURRENT
            return "CONCURRENT"

    @staticmethod
    def from_str(s: str) -> Any:
        if s == "UNIFORM_CONTINUOUS":
            return Capability.UNIFORM_CONTINUOUS
        elif s == "UNIFORM_DISCRETE":
            return Capability.UNIFORM_DISCRETE
        elif s == "LOG_UNIFORM_CONTINUOUS":
            return Capability.LOG_UNIFORM_CONTINUOUS
        elif s == "LOG_UNIFORM_DISCRETE":
            return Capability.LOG_UNIFORM_DISCRETE
        elif s == "CATEGORICAL":
            return Capability.CATEGORICAL
        elif s == "CONDITIONAL":
            return Capability.CONDITIONAL
        elif s == "MULTI_OBJECTIVE":
            return Capability.MULTI_OBJECTIVE
        elif s == "CONCURRENT":
            return Capability.CONCURRENT
        else:
            raise ValueError("Unknown capability: {}".format(s))


def all_capabilities() -> Set[Capability]:
    return {
        Capability.UNIFORM_CONTINUOUS,
        Capability.UNIFORM_DISCRETE,
        Capability.LOG_UNIFORM_CONTINUOUS,
        Capability.LOG_UNIFORM_DISCRETE,
        Capability.CATEGORICAL,
        Capability.CONDITIONAL,
        Capability.MULTI_OBJECTIVE,
        Capability.CONCURRENT,
    }


class SolverSpec(object):
    def __init__(
        self,
        name: str,
        capabilities: Optional[Set[Capability]] = None,
        attrs: Optional[Dict[str, str]] = None,
    ):
        self.name = name

        if capabilities is None:
            self.capabilities = all_capabilities()
        else:
            self.capabilities = copy.deepcopy(capabilities)

        if attrs is None:
            self.attrs = {}
        else:
            self.attrs = copy.deepcopy(attrs)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "attrs": self.attrs,
            "capabilities": [c.to_str() for c in self.capabilities],
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> Any:
        return SolverSpec(
            name=d["name"],
            attrs=d["attrs"],
            capabilities={Capability.from_str(c) for c in d["capabilities"]},
        )


class NextTrial(object):
    def __init__(self, trial_id: int, params: List[Optional[float]], next_step: Optional[int]):
        self.trial_id = trial_id
        self.params = params
        self.next_step = next_step

    def to_dict(self) -> Dict[str, Any]:
        return {"id": self.trial_id, "params": self.params, "next_step": self.next_step}

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> Any:
        return NextTrial(trial_id=d["id"], params=d["params"], next_step=d["next_step"])


class EvaluatedTrial(object):
    def __init__(self, trial_id: int, values: List[float], current_step: int):
        self.trial_id = trial_id
        self.values = values
        self.current_step = current_step

    def to_dict(self) -> Dict[str, Any]:
        return {"id": self.trial_id, "values": self.values, "current_step": self.current_step}

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> Any:
        return EvaluatedTrial(trial_id=d["id"], values=d["values"], current_step=d["current_step"])


class TrialIdGenerator(object):
    def __init__(self, next_id: int):
        self.next_id = next_id

    def generate(self) -> int:
        trial_id = self.next_id
        self.next_id += 1
        return trial_id


class Solver(object):
    @abc.abstractmethod
    def ask(self, idg: TrialIdGenerator) -> NextTrial:
        raise NotImplementedError

    @abc.abstractmethod
    def tell(self, trial: EvaluatedTrial):
        raise NotImplementedError


class SolverFactory(object):
    @abc.abstractmethod
    def specification(self) -> SolverSpec:
        raise NotImplementedError

    @abc.abstractmethod
    def create_solver(self, seed: int, problem: ProblemSpec) -> Solver:
        raise NotImplementedError


class SolverRunner(object):
    def __init__(self, factory: SolverFactory):
        self._factory = factory
        self._solvers = {}  # type: Dict[int, Solver]

    def run(self):
        self._cast_solver_spec()

        while self._run_once():
            pass

    def _run_once(self) -> bool:
        message = self._recv_message()
        if message is None:
            return False

        message_type = message["type"]
        if message_type == "CREATE_SOLVER_CAST":
            self._handle_create_solver_cast(message)
        elif message_type == "DROP_SOLVER_CAST":
            self._handle_drop_solver_cast(message)
        elif message_type == "ASK_CALL":
            self._handle_ask_call(message)
        elif message_type == "TELL_CALL":
            self._handle_tell_call(message)
        else:
            raise ValueError("Unexpected message: {}".format(message))

        return True

    def _handle_create_solver_cast(self, message: Dict[str, Any]):
        solver_id = message["solver_id"]
        random_seed = message["random_seed"]
        problem = ProblemSpec.from_dict(message["problem"])
        assert solver_id not in self._solvers

        random_seed = random_seed % np.iinfo(np.uint32).max
        solver = self._factory.create_solver(random_seed, problem)
        self._solvers[solver_id] = solver

    def _handle_drop_solver_cast(self, message: Dict[str, Any]):
        solver_id = message["solver_id"]
        del self._solvers[solver_id]

    def _handle_ask_call(self, message: Dict[str, Any]):
        solver_id = message["solver_id"]
        next_trial_id = message["next_trial_id"]

        idg = TrialIdGenerator(next_trial_id)
        solver = self._solvers[solver_id]
        trial = solver.ask(idg)

        message = {"type": "ASK_REPLY", "trial": trial.to_dict(), "next_trial_id": idg.next_id}
        self._send_message(message)

    def _handle_tell_call(self, message: Dict[str, Any]):
        solver_id = message["solver_id"]
        trial = EvaluatedTrial.from_dict(message["trial"])

        solver = self._solvers[solver_id]
        solver.tell(trial)

        message = {"type": "TELL_REPLY"}
        self._send_message(message)

    def _cast_solver_spec(self):
        spec = self._factory.specification()
        self._send_message({"type": "SOLVER_SPEC_CAST", "spec": spec.to_dict()})

    def _send_message(self, message: Dict[str, Any]):
        print(json.dumps(message))

    def _recv_message(self) -> Optional[Dict[str, Any]]:
        try:
            message = input()
            return json.loads(message)
        except Exception:
            return None
