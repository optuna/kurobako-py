kurobako-py
===========

[![pypi](https://img.shields.io/pypi/v/kurobako.svg)](https://pypi.python.org/pypi/kurobako)
[![GitHub license](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/sile/kurobako-py)
[![Actions Status](https://github.com/sile/kurobako-py/workflows/CI/badge.svg)](https://github.com/sile/kurobako-py/actions)

A Python library to help implement [kurobako]'s solvers and problems.

[kurobako]: https://github.com/sile/kurobako


Installation
------------

```console
$ pip install kurobako
```

Usage Examples
--------------

### Define a solver based on random search

```python
# filename: random_solver.py
import numpy as np

from kurobako import problem
from kurobako import solver


class RandomSolverFactory(solver.SolverFactory):
    def specification(self):
        return solver.SolverSpec(name='Random Search')

    def create_solver(self, seed, problem):
        return RandomSolver(seed, problem)


class RandomSolver(solver.Solver):
    def __init__(self, seed, problem):
        self._rng = np.random.RandomState(seed)
        self._problem = problem

    def ask(self, idg):
        params = []
        for p in self._problem.params:
            if p.distribution == problem.Distribution.UNIFORM:
                params.append(self._rng.uniform(p.range.low, p.range.high))
            else:
                low = np.log(p.range.low)
                high = np.log(p.range.high)
                params.append(float(np.exp(self._rng.uniform(low, high))))

        trial_id = idg.generate()
        next_step = self._problem.last_step
        return solver.NextTrial(trial_id, params, next_step)

    def tell(self, trial):
        pass


if __name__ == '__main__':
    runner = solver.SolverRunner(RandomSolverFactory())
    runner.run()
```

### Define a problem that represents a quadratic function `x**2 + y`

```python
# filename: quadratic_problem.py
from kurobako import problem


class QuadraticProblemFactory(problem.ProblemFactory):
    def specification(self):
        params = [
            problem.Var('x', problem.ContinuousRange(-10, 10)),
            problem.Var('y', problem.DiscreteRange(-3, 3))
        ]
        return problem.ProblemSpec(name='Quadratic Function',
                                   params=params,
                                   values=[problem.Var('x**2 + y')])

    def create_problem(self, seed):
        return QuadraticProblem()


class QuadraticProblem(problem.Problem):
    def create_evaluator(self, params):
        return QuadraticEvaluator(params)


class QuadraticEvaluator(problem.Evaluator):
    def __init__(self, params):
        self._x, self._y = params
        self._current_step = 0

    def current_step(self):
        return self._current_step

    def evaluate(self, next_step):
        self._current_step = 1
        return [self._x**2 + self._y]


if __name__ == '__main__':
    runner = problem.ProblemRunner(QuadraticProblemFactory())
    runner.run()
```

### Run a benchmark that uses the above solver and problem

```console
$ SOLVER=$(kurobako solver command python3 random_solver.py)
$ PROBLEM=$(kurobako problem command python3 quadratic_problem.py)
$ kurobako studies --solvers $SOLVER --problems $PROBLEM | kurobako run > result.json
```
