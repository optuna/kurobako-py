import numpy as np

from kurobako import problem
from kurobako import solver


class RandomSolverFactory(solver.SolverFactory):
    def specification(self) -> solver.SolverSpec:
        return solver.SolverSpec(name="Random Search")

    def create_solver(self, seed: int, problem: problem.ProblemSpec) -> solver.Solver:
        return RandomSolver(seed, problem)


class RandomSolver(solver.Solver):
    def __init__(self, seed: int, problem: problem.ProblemSpec):
        self._rng = np.random.RandomState(seed)
        self._problem = problem

    def ask(self, idg: solver.TrialIdGenerator) -> solver.NextTrial:
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

    def tell(self, trial: solver.EvaluatedTrial):
        pass


if __name__ == "__main__":
    runner = solver.SolverRunner(RandomSolverFactory())
    runner.run()
