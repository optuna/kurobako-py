import abc
import copy
import json
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import List
from typing import Union

from kurobako.parameter import ParamDomain
from kurobako.domain import Domain
from kurobako.domain import Var


class ProblemSpec(object):
    """Problem specification."""
    def __init__(self,
                 name: str,
                 params: List[Var],
                 values: List[Var],
                 attrs: Dict[str, str] = {},
                 steps: Union[int, List[int]] = 1):
        self.name = name
        self.attrs = copy.deepcopy(attrs)
        self.params_domain = Domain(params)
        self.values_domain = Domain(values)
        self.steps = steps

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> Any:
        """Creates a `ProblemSpec` instance from the given dictionary."""

        return ProblemSpec(name=d['name'],
                           attrs=d['attrs'],
                           params=Domain.from_list(d['params_domain']).variables,
                           values=Domain.from_list(d['values_domain']).variables,
                           steps=d['steps'])

    def to_dict(self) -> Dict[str, Any]:
        """Converts this instance to dictionary format."""

        return {
            'name': self.name,
            'attrs': self.attrs,
            'params_domain': self.params_domain.to_list(),
            'values_domain': self.values_domain.to_list(),
            'steps': self.steps
        }


class Evaluator(object):
    @abc.abstractmethod
    def evaluate(self, next_step: int) -> List[float]:
        raise NotImplementedError

    def current_step(self) -> int:
        raise NotImplementedError


class Problem(object):
    @abc.abstractmethod
    def create_evaluator(self, params: List[float]) -> Optional[Evaluator]:
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

        while True:
            message = self._recv_message()
            if message is None:
                break;

            message_type = message['type']
            if message_type == 'CREATE_PROBLEM_CAST':
                self._handle_create_problem_cast(message)
            elif message_type == 'DROP_PROBLEM_CAST':
                self._handle_drop_problem_cast(message)
            elif message_type == 'CREATE_EVALUATOR_CALL':
                self._handle_create_evaluator_call(message)
            elif message_type == 'DROP_EVALUATOR_CAST':
                self._handle_drop_evaluator_cast(message)
            elif message_type == 'EVALUATE_CALL':
                self._handle_evaluate_call(message)
            else:
                raise ValueError('Unexpected message: {}'.format(message))

    def _handle_create_problem_cast(self, message):
        problem_id = message['problem_id']
        random_seed = message['random_seed']
        assert problem_id not in self._problems

        problem = self._factory.create_problem(random_seed)
        self._problems[problem_id] = problem

    def _handle_drop_problem_cast(self, message):
        problem_id = message['problem_id']
        del self._problems[problem_id]

    def _handle_create_evaluator_call(self, message):
        problem_id = message['problem_id']
        evaluator_id = message['evaluator_id']
        params = message['params']
        assert evaluator_id not in self._evaluators

        problem = self._problems[problem_id]
        evaluator = problem.create_evaluator(params)
        if evaluator is None:
            self._send_message({'type': 'ERROR_REPLY', 'kind': 'UNEVALABLE_PARAMS'})
        else:
            self._evaluators[evaluator_id] = evaluator
            self._send_message({'type': 'CREATE_EVALUATOR_REPLY'})

    def _handle_drop_evaluator_cast(self, message):
        evaluator_id = message['evaluator_id']
        del self._evaluators[evaluator_id]

    def _handle_evaluate_call(self, message):
        evaluator_id = message['evaluator_id']
        next_step = message['next_step']

        evaluator = self._evaluators[evaluator_id]
        values = evaluator.evaluate(next_step)
        current_step = evaluator.current_step()

        self._send_message({
            'type': 'EVALUATE_REPLY',
            'current_step': current_step,
            'values': values
        })

    def _cast_problem_spec(self):
        spec = self._factory.specification()
        self._send_message({'type': 'PROBLEM_SPEC_CAST', 'spec': spec.to_dict()})

    def _send_message(self, message: Dict[str, Any]):
        print(json.dumps(message))

    def _recv_message(self) -> Optional[Dict[str, Any]]:
        try:
            message = input()
            return json.loads(message)
        except:
            return None
