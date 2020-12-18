import optuna
from pkg_resources import get_distribution
import queue
from typing import Callable
from typing import Dict  # NOQA
from typing import List  # NOQA
from typing import Optional  # NOQA
from typing import Tuple  # NOQA

from kurobako import problem
from kurobako import solver

_optuna_logger = optuna.logging.get_logger(__name__)


class OptunaSolverFactory(solver.SolverFactory):
    def __init__(
        self, create_study: Callable[[int], optuna.Study], use_discrete_uniform: bool = False
    ):
        self._create_study = create_study
        self._use_discrete_uniform = use_discrete_uniform

    def specification(self) -> solver.SolverSpec:
        return solver.SolverSpec(
            name="Optuna",
            attrs={
                "version": "optuna={}, kurobako-py={}".format(
                    get_distribution("optuna").version, get_distribution("kurobako").version
                ),
                "github": "https://github.com/optuna/optuna",
                "paper": 'Akiba, Takuya, et al. "Optuna: A next-generation hyperparameter '
                'optimization framework." Proceedings of the 25th ACM SIGKDD International '
                "Conference on Knowledge Discovery & Data Mining. ACM, 2019.",
            },
            capabilities={
                solver.Capability.UNIFORM_CONTINUOUS,
                solver.Capability.LOG_UNIFORM_CONTINUOUS,
                solver.Capability.UNIFORM_DISCRETE,
                solver.Capability.CATEGORICAL,
                solver.Capability.CONDITIONAL,
                solver.Capability.CONCURRENT,
                solver.Capability.MULTI_OBJECTIVE,
            },
        )

    def create_solver(self, seed: int, problem: problem.ProblemSpec) -> solver.Solver:
        study = self._create_study(seed)
        return OptunaSolver(study, problem, use_discrete_uniform=self._use_discrete_uniform)


class OptunaSolver(solver.Solver):
    def __init__(
        self, study: optuna.Study, problem: problem.ProblemSpec, use_discrete_uniform: bool = False
    ):
        self._study = study
        self._problem = problem
        self._use_discrete_uniform = use_discrete_uniform
        self._waitings = queue.Queue()  # type: queue.Queue[Tuple[int, optuna.Trial]]
        self._pruned = queue.Queue()  # type: queue.Queue[Tuple[int, optuna.Trial]]
        self._runnings = {}  # type: Dict[int, optuna.Trial]

    def _next_step(self, current_step: int) -> int:
        pruner = self._study.pruner
        if isinstance(pruner, optuna.pruners.NopPruner):
            return self._problem.last_step
        elif isinstance(pruner, optuna.pruners.SuccessiveHalvingPruner):
            try:
                rung = 0
                while True:
                    n = pruner._min_early_stopping_rate + rung
                    step = pruner._min_resource * (pruner._reduction_factor ** n)
                    if step > current_step:
                        return min(step, self._problem.last_step)
                    rung += 1
            except Exception:
                # For compatibility.
                return current_step + 1
        else:
            return current_step + 1

    def ask(self, idg: solver.TrialIdGenerator) -> solver.NextTrial:
        if not self._pruned.empty():
            kurobako_trial_id, trial = self._pruned.get()
            next_step = None
        elif self._waitings.empty():
            kurobako_trial_id = idg.generate()
            trial = self._create_new_trial()
            next_step = self._next_step(0)
        else:
            kurobako_trial_id, trial = self._waitings.get()
            current_step = self._study._storage.get_trial(trial._trial_id).last_step
            next_step = self._next_step(current_step)

        params = []  # type: List[Optional[float]]
        for p in self._problem.params:
            if p.is_constraint_satisfied(self._problem.params, params):
                params.append(self._suggest(trial, p))
            else:
                params.append(None)

        self._runnings[kurobako_trial_id] = trial
        return solver.NextTrial(trial_id=kurobako_trial_id, params=params, next_step=next_step)

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
                return trial.suggest_loguniform(v.name, v.range.low, v.range.high)
        elif isinstance(v.range, problem.DiscreteRange):
            if self._use_discrete_uniform:
                return trial.suggest_discrete_uniform(v.name, v.range.low, v.range.high - 1, q=1)
            else:
                return trial.suggest_int(v.name, v.range.low, v.range.high - 1)
        elif isinstance(v.range, problem.CategoricalRange):
            category = trial.suggest_categorical(v.name, v.range.choices)
            return v.range.choices.index(category)

        raise ValueError("Unsupported parameter: {}".format(v))

    def tell(self, evaluated_trial: solver.EvaluatedTrial):
        kurobako_trial_id = evaluated_trial.trial_id
        values = evaluated_trial.values
        current_step = evaluated_trial.current_step

        trial = self._runnings[kurobako_trial_id]
        del self._runnings[kurobako_trial_id]

        if len(values) == 0:
            message = "Unevaluable trial#{}: step={}".format(trial.number, current_step)
            _optuna_logger.info(message)
            self._study._storage.set_trial_state(trial._trial_id, optuna.trial.TrialState.PRUNED)
            return

        assert len(values) == len(self._study.directions)
        for i in range(len(values)):
            if self._study.directions[i] == optuna.study.StudyDirection.MAXIMIZE:
                values[i] = -values[i]

        assert current_step <= self._problem.last_step
        if self._problem.last_step == current_step:
            self._study._storage.set_trial_values(trial._trial_id, values)
            self._study._storage.set_trial_state(trial._trial_id, optuna.trial.TrialState.COMPLETE)
            self._study._log_completed_trial(trial, values)
        else:
            if len(values) > 1:
                raise NotImplementedError(
                    "Pruning is not yet supported for optimization with more than one objective."
                )

            value = values[0]
            trial.report(value, current_step)
            if trial.should_prune():
                message = "Pruned trial#{}: step={}, value={}".format(
                    trial.number, current_step, value
                )
                _optuna_logger.info(message)
                self._study._storage.set_trial_state(
                    trial._trial_id, optuna.trial.TrialState.PRUNED
                )
                self._pruned.put((kurobako_trial_id, trial))
            else:
                self._waitings.put((kurobako_trial_id, trial))

    def _create_new_trial(self):
        try:
            trial_id = self._study._storage.create_new_trial(self._study._study_id)
        except Exception:
            # For compatibility.
            trial_id = self._study._storage.create_new_trial(self._study.study_id)

        return optuna.trial.Trial(self._study, trial_id)
