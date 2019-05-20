import json

from kurobako.budget import Budget


class SolverCapabilities(object):
    def __init__(self):
        self._capabilities = set()

    def categorical(self):
        self._capabilities.add('CATEGORICAL')
        return self

    def conditional(self):
        self._capabilities.add('CONDITIONAL')
        return self

    def discrete(self):
        self._capabilities.add('DISCRETE')
        return self

    def log_uniform(self):
        self._capabilities.add('LOG_UNIFORM')
        return self

    def to_list(self):
        return list(self._capabilities)


class SolverSpec(object):
    def __init__(self, name, capabilities, version=None):
        self._name = name
        self._capabilities = capabilities
        self._version = version

    def to_message(self):
        return json.dumps({
            'type': 'SOLVER_SPEC_CAST',
            'name': self._name,
            'version': self._version,
            'capabilities': self._capabilities.to_list()
        })


class SolverRunner(object):
    def __init__(self, solver):
        self._solver = solver

    def run(self):
        while True:
            self._run_once()

    def _run_once(self):
        message = json.loads(input())
        if message['type'] == 'ASK_CALL':
            id_hint = message['id_hint']
            obs_id, params, budget = self._solver.ask(id_hint)
            print(
                json.dumps({
                    'type': 'ASK_REPLY',
                    'id': obs_id,
                    'params': [p.to_json() for p in params],
                    'budget': budget.to_json(),
                }))
        elif message['type'] == 'TELL_CALL':
            obs_id = message['id']
            budget = Budget.from_json(message['budget'])
            values = message['values']
            self._solver.tell(obs_id, budget, values)
            print(json.dumps({'type': 'TELL_REPLY'}))
        else:
            raise NotImplementedError("{}".format(message))
