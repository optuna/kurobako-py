import optuna
import queue
from typing import Callable
from typing import Dict
from typing import List
from typing import Tuple

from kurobako import problem
from kurobako import solver


class OptunaSolverFactory(solver.SolverFactory):
    def __init__(self, create_study: Callable[[int], optuna.Study]):
        self._create_study = create_study

    def specification(self) -> solver.SolverSpec:
        return solver.SolverSpec(
            name='Optuna',
            attrs={
                'version': optuna.__version__,
                'github': 'https://github.com/optuna/optuna',
                'paper': 'Optuna: A Next-generation Hyperparameter Optimization Framework'
            },
            capabilities={
                solver.Capability.UNIFORM_CONTINUOUS, solver.Capability.LOG_UNIFORM_CONTINUOUS,
                solver.Capability.UNIFORM_DISCRETE, solver.Capability.CATEGORICAL,
                solver.Capability.CONDITIONAL, solver.Capability.CONCURRENT
            })

    def create_solver(self, seed: int, problem: problem.ProblemSpec) -> solver.Solver:
        study = self._create_study(seed)
        return OptunaSolver(study, problem)


class OptunaSolver(solver.Solver):
    def __init__(self, study: optuna.Study, problem: problem.ProblemSpec):
        self._study = study
        self._problem = problem
        self._waitings = queue.Queue()  # type: queue.Queue[Tuple[int, optuna.Trial]]
        self._pruned = queue.Queue()  # type: queue.Queue[Tuple[int, optuna.Trial]]
        self._runnings = {}  # type: Dict[int, optuna.Trial]

    def ask(self, idg: solver.TrialIdGenerator) -> solver.Trial:
        if not self._pruned.empty():
            kurobako_trial_id, trial = self._pruned.get()
            next_step = None
        elif self._waitings.empty():
            kurobako_trial_id = idg.generate()
            trial = self._create_new_trial()
            if isinstance(self._study.pruner, optuna.pruners.NopPruner):
                next_step = self._problem.last_step
            else:
                next_step = 1
        else:
            kurobako_trial_id, trial = self._waitings.get()
            current_step = self._study._storage.get_trial(trial._trial_id).last_step
            next_step = current_step + 1

        params = []  # type: List[float]
        for p in self._problem.params:
            params.append(self._suggest(trial, p))

        self._runnings[kurobako_trial_id] = trial
        return solver.Trial(trial_id=kurobako_trial_id, params=params, next_step=next_step)

    def _suggest(self, trial: optuna.Trial, v: problem.Var) -> float:
        if v.name in trial.params:
            if isinstance(trial.params[v.name], str):
                assert isinstance(v.range, problem.CategoricalRange)
                return v.range.choices.index(trial.params[v.name])
            else:
                return trial.params[v.name]

        if isinstance(v.range, problem.ContinuousRange):
            if v.distribution == problem.Distribution.UNIFORM:
                return trial.suggest_uniform(v.name, v.range.low, v.range.high)
            elif v.distribution == problem.Distribution.LOG_UNIFORM:
                return trial.suggest_log_uniform(v.name, v.range.low, v.range.high)
        elif isinstance(v.range, problem.DiscreteRange):
            return trial.suggest_int(v.name, v.range.low, v.range.high - 1)
        elif isinstance(v.range, problem.CategoricalRange):
            category = trial.suggest_categorical(v.name, v.range.choices)
            return v.range.choices.index(category)

        raise ValueError('Unsupported parameter: {}'.format(v))

    def tell(self, kurobako_trial_id: int, values: List[float], current_step: int):
        trial = self._runnings[kurobako_trial_id]
        del self._runnings[kurobako_trial_id]

        if len(values) == 0:
            message = 'Unevaluable trial#{}: step={}'.format(trial.number, current_step)
            self._study.logger.info(message)
            self._study._storage.set_trial_state(trial._trial_id, optuna.structs.TrialState.PRUNED)
            return

        value = values[0]

        if self._study.direction == optuna.structs.StudyDirection.MAXIMIZE:
            value = -value

        assert current_step <= self._problem.last_step
        if self._problem.last_step == current_step:
            trial.report(value)
            self._study._storage.set_trial_state(trial._trial_id,
                                                 optuna.structs.TrialState.COMPLETE)
            self._study._log_completed_trial(trial.number, value)
        else:
            trial.report(value, current_step)
            if trial.should_prune(current_step):
                message = 'Pruned trial#{}: step={}, value={}'.format(
                    trial.number, current_step, value)
                self._study.logger.info(message)
                self._study._storage.set_trial_state(trial._trial_id,
                                                     optuna.structs.TrialState.PRUNED)
                self._pruned.put((kurobako_trial_id, trial))
            else:
                self._waitings.put((kurobako_trial_id, trial))

    def _create_new_trial(self):
        trial_id = self._study._storage.create_new_trial(self._study.study_id)
        return optuna.trial.Trial(self._study, trial_id)
