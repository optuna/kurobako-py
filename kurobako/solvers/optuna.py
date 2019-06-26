import queue

import optuna

from kurobako.budget import Budget
from kurobako.parameter import CategoricalParam
from kurobako.parameter import ContinuousParam
from kurobako.parameter import ContinuousValue
from kurobako.parameter import DiscreteParam
from kurobako.parameter import Distribution
from kurobako.parameter import ParamDomain
from kurobako.solver import SolverCapabilities
from kurobako.solver import SolverSpec


class OptunaSolver(object):
    def __init__(self, problem, sampler=None, pruner=None, direction='minimize'):
        self._study = optuna.create_study(sampler=sampler, pruner=pruner, direction=direction)
        self._problem = problem
        self._waitings = queue.Queue()
        self._runnings = {}

    @staticmethod
    def specification():
        return SolverSpec(
            name='optuna',
            version=optuna.__version__,
            capabilities=SolverCapabilities().categorical().conditional().discrete().log_uniform())

    def ask(self, id_hint):
        if self._waitings.empty():
            trial = self._create_new_trial()
            budget = self._create_new_budget()
        else:
            trial = self._waitings.get()
            consumption = max(
                self._study.storage.get_trial(trial._trial_id).intermediate_values.keys())
            budget = Budget(consumption + 1, consumption=consumption)

        params = ParamDomain.ask_independent_values(
            self._problem.params_domain, lambda p: self._suggest(p, trial))

        self._runnings[trial.number] = trial
        return trial.number, params, budget

    def tell(self, obs_id, budget, values):
        assert len(values) == 1
        value = values[0]
        if self._study.direction == optuna.structs.StudyDirection.MAXIMIZE:
            value = -value

        trial = self._runnings[obs_id]
        del self._runnings[obs_id]

        if self._problem.is_completed(budget):
            trial.report(value)
            self._study.storage.set_trial_state(trial._trial_id,
                                                optuna.structs.TrialState.COMPLETE)
            self._study._log_completed_trial(trial.number, value)
        else:
            trial.report(value, budget.consumption)
            if trial.should_prune(budget.consumption):
                message = 'Pruned trial#{}: step={}, value={}'.format(
                    trial.number, budget.consumption, value)
                self._study.logger.info(message)
                self._study.storage.set_trial_state(trial._trial_id,
                                                    optuna.structs.TrialState.PRUNED)
            else:
                self._waitings.put(trial)

    def _suggest(self, param, trial):
        if isinstance(param, ContinuousParam):
            if param.distribution == Distribution.UNIFORM:
                v = trial.suggest_uniform(param.name, param.low, param.high)
                return ContinuousValue(v)
            else:
                v = trial.suggest_loguniform(param.name, param.low, param.high)
                return ContinuousValue(v)
        elif isinstance(param, DiscreteParam):
            v = trial.suggest_int(param.name, param.low, param.high - 1)
            return param.make_value(v)
        elif isinstance(param, CategoricalParam):
            v = trial.suggest_categorical(param.name, param.choices)
            return param.make_value(v)
        else:
            raise NotImplementedError('{}'.format(param))

    def _create_new_trial(self):
        trial_id = self._study.storage.create_new_trial_id(self._study.study_id)
        return optuna.trial.Trial(self._study, trial_id)

    def _create_new_budget(self):
        return Budget(1)
