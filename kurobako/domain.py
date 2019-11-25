# TODO: merge with problem module
import abc
import enum
import numpy as np
from typing import Any
from typing import Dict
from typing import List


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
    @abc.abstractmethod
    def from_dict(d: Dict[str, Any]) -> Any:
        raise NotImplementedError


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
        d = {'type': 'CONTINUOUS'}
        if np.isfinite(self._low):
            d['low'] = self._low
        if np.isfinite(self._high):
            d['high'] = self._high
        return d

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> Any:
        return ContinuousRange(low=d['low'], high=d['high'])


class DiscreteRange(Range):
    def __init__(self, low: int, high: int):
        self._low = low
        self._high = high

    @property
    def low(self) -> float:
        return self._low

    @property
    def high(self) -> float:
        return self._high

    def to_dict(self) -> Dict[str, Any]:
        return {'type': 'DISCRETE', 'low': self._low, 'high': self._high}

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> Any:
        return DiscreteRange(low=d['low'], high=d['high'])


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
        return {'type': 'CATEGORICAL', 'choices': self.choices}

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> Any:
        return CategoricalRange(choices=d['choices'])


class Distribution(enum.Enum):
    UNIFORM = 0
    LOG_UNIFORM = 1

    def to_str(self) -> str:
        if self == Distribution.UNIFORM:
            return 'UNIFORM'
        else:
            return 'LOG_UNIFORM'

    @staticmethod
    def from_str(s: str) -> Any:
        if s == 'UNIFORM':
            return Distribution.UNIFORM
        elif s == 'LOG_UNIFORM':
            return Distribution.LOG_UNIFORM
        else:
            raise ValueError


class Var(object):
    """A variable in a domain."""
    def __init__(self,
                 name: str,
                 range: Range = ContinuousRange(low=float('-inf'), high=float('inf')),
                 distribution: Distribution = Distribution.UNIFORM):
        self.name = name
        self.range = range
        self.distribution = distribution
        # TODO: add `condition` field

    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'range': self.range.to_dict(),
            'distribution': self.distribution.to_str()
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> Any:
        return Var(name=d['name'],
                   range=Range.from_dict(d['range']),
                   distribution=Distribution.from_str(d['distribution']))


class Domain(object):
    """Domain of input parameters or objective values."""
    def __init__(self, variables: List[Var]):
        self.variables = variables

    def to_list(self) -> List[Any]:
        return [v.to_dict() for v in self.variables]

    @staticmethod
    def from_list(xs: List[Any]) -> Any:
        return Domain(variables=[Var.from_dict(x) for x in xs])
