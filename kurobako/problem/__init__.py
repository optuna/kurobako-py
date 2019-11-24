import copy
from typing import Any
from typing import Dict
from typing import List
from typing import Union

from kurobako.parameter import ParamDomain
from kurobako.domain import Domain


class ProblemSpec(object):
    """Problem specification."""
    def __init__(self,
                 name: str,
                 params_domain: Domain,
                 values_domain: Domain,
                 attrs: Dict[str, str] = {},
                 steps: Union[int, List[int]] = 1):
        self.name = name
        self.attrs = copy.deepcopy(attrs)
        self.params_domain = params_domain
        self.values_domain = values_domain
        self.steps = steps

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> Any:
        """Creates a `ProblemSpec` instance from the given dictionary."""

        return ProblemSpec(name=d['name'],
                           attrs=d['attrs'],
                           params_domain=Domain.from_list(d['params_domain']),
                           values_domain=Domain.from_list(d['values_domain']),
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
