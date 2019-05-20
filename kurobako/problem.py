import json

from kurobako.parameter import ParamDomain


class ProblemCapabilities(object):
    def __init__(self):
        self._capabilities = set()

    def concurrent(self):
        self._capabilities.add('CONCURRENT')
        return self

    def dynamic_param_change(self):
        self._capabilities.add('DYNAMIC_PARAM_CHANGE')
        return self

    @staticmethod
    def from_json(values):
        this = ProblemCapabilities()
        for v in values:
            if v == 'CONCURRENT':
                this.concurrent()
            elif v == 'DYNAMIC_PARAM_CHANGE':
                this.dynamic_param_change()
            else:
                raise ValueError('Unknown capability: {}'.format(v))
        return this


class ProblemSpec(object):
    def __init__(self,
                 name,
                 params_domain,
                 values_domain,
                 capabilities,
                 evaluation_expense=1,
                 version=None):
        self.name = name
        self.params_domain = params_domain
        self.values_domain = values_domain
        self.capabilities = capabilities
        self.evaluation_expense = evaluation_expense
        self.version = version

    def is_completed(self, budget):
        return budget.consumption >= self.evaluation_expense

    @staticmethod
    def from_message(m):
        m = json.loads(m)
        assert m['type'] == 'PROBLEM_SPEC_CAST'

        capabilities = ProblemCapabilities.from_json(m['capabilities'])
        params_domain = [ParamDomain.from_json(p) for p in m['params-domain']]
        return ProblemSpec(
            name=m['name'],
            version=m['version'],
            params_domain=params_domain,
            values_domain=m['values-domain'],  # TODO
            capabilities=capabilities,
            evaluation_expense=m['evaluation-expense'])
