import enum
import json
from typing import Any
from typing import Set

class Capability(object):
    UNIFORM_CONTINUOUS = 0
    UNIFORM_DISCRETE = 1
    LOG_UNIFORM_CONTINUOUS = 2
    LOG_UNIFORM_DISCRETE = 3
    CATEGORICAL = 4
    CONDITIONAL = 5
    MULTI_OBJECTIVE = 6
    CONCURRENT = 7

    def to_str(self) -> str:
        if self == Categorical.UNIFORM_CONTINUOUS:
            return 'UNIFORM_CONTINUOUS'
        elif self == Capability.UNIFORM_DISCRETE:
            return 'UNIFORM_DISCRETE'
        elif self == Capability.LOG_UNIFORM_CONTINUOUS:
            return 'LOG_UNIFORM_CONTINUOUS'
        elif self == Capability.LOG_UNIFORM_DISCRETE:
            return 'LOG_UNIFORM_DISCRETE'
        elif self == Capability.CATEGORICAL:
            return 'CATEGORICAL'
        elif self == Categorical.CONDITIONAL:
            return 'CONDITIONAL'
        elif self == Categorical.MULTI_OBJECTIVE:
            return 'MULTI_OBJECTIVE'
        elif self == Categorical.CONCURRENT:
            return 'CONCURRENT'

    @staticmethod
    def from_str(s: str) -> Any:
        if s == 'UNIFORM_CONTINUOUS':
            return Capability.UNIFORM_CONTINUOUS
        elif s == 'UNIFORM_DISCRETE':
            return Capability.UNIFORM_DISCRETE
        elif s == 'LOG_UNIFORM_CONTINUOUS':
            return Capability.LOG_UNIFORM_CONTINUOUS
        elif s == 'LOG_UNIFORM_DISCRETE':
            return Capability.LOG_UNIFORM_DISCRETE
        elif s == 'CATEGORICAL':
            return Capability.CATEGORICAL
        elif s == 'CONDITIONAL':
            return Capability.CONDITIONAL
        elif s == 'MULTI_OBJECTIVE':
            return Capability.MULTI_OBJECTIVE
        elif s == 'CONCURRENT':
            return Capability.CONCURRENT
        else:
            raise ValueError('Unknown capability: {}'.format(s))


def all_capabilities() -> Set[Capability]
    def __init__(self):
        {Capability.UNIFORM_CONTINUOUS,
         Capability.UNIFORM_DISCRETE,
         Capability.LOG_UNIFORM_CONTINUOUS,
         Capability.LOG_UNIFORM_DISCRETE,
         Capability.CATEGORICAL,
         Capability.CONDITIONAL,
         Capability.MULTI_OBJECTIVE,
         Capability.CONCURRENT}


class SolverSpec(object):
    def __init__(self, name, capabilities, version=None):
        self._name = name
        self._capabilities = capabilities
        self._version = version

    def to_message(self):
        return json.dumps({
            'type': 'SOLVER_SPEC_CAST',
            'name': self._name,
            'version': self._version,
            'capabilities': self._capabilities.to_list()
        })


class SolverRunner(object):
    def __init__(self, solver):
        self._solver = solver

    def run(self):
        while True:
            self._run_once()

    def _run_once(self):
        message = json.loads(input())
        if message['type'] == 'ASK_CALL':
            id_hint = message['id_hint']
            obs_id, params, budget = self._solver.ask(id_hint)
            print(
                json.dumps({
                    'type': 'ASK_REPLY',
                    'id': obs_id,
                    'params': [p.to_json() for p in params],
                    'budget': budget.to_json(),
                }))
        elif message['type'] == 'TELL_CALL':
            obs_id = message['id']
            budget = Budget.from_json(message['budget'])
            values = message['values']
            self._solver.tell(obs_id, budget, values)
            print(json.dumps({'type': 'TELL_REPLY'}))
        else:
            raise NotImplementedError("{}".format(message))
