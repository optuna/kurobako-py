from typing import List
from typing import Optional

from kurobako import problem


class QuadraticProblemFactory(problem.ProblemFactory):
    def specification(self) -> problem.ProblemSpec:
        params = [
            problem.Var("x", problem.ContinuousRange(-10, 10)),
            problem.Var("y", problem.DiscreteRange(-3, 3)),
        ]
        return problem.ProblemSpec(
            name="Quadratic Function", params=params, values=[problem.Var("x**2 + y")]
        )

    def create_problem(self, seed: int) -> problem.Problem:
        return QuadraticProblem()


class QuadraticProblem(problem.Problem):
    def create_evaluator(self, params: List[Optional[float]]) -> problem.Evaluator:
        return QuadraticEvaluator(params)


class QuadraticEvaluator(problem.Evaluator):
    def __init__(self, params: List[Optional[float]]):
        x, y = params
        assert x is not None
        assert y is not None

        self._x = x
        self._y = y
        self._current_step = 0

    def current_step(self) -> int:
        return self._current_step

    def evaluate(self, next_step: int) -> List[float]:
        self._current_step = 1
        return [self._x**2 + self._y]


if __name__ == "__main__":
    runner = problem.ProblemRunner(QuadraticProblemFactory())
    runner.run()
