import enum


class Distribution(enum.Enum):
    UNIFORM = 0
    LOG_UNIFORM = 1

    @staticmethod
    def from_json(value):
        if value == 'uniform':
            return Distribution.UNIFORM
        elif value == 'log-uniform':
            return Distribution.LOG_UNIFORM
        else:
            raise ValueError('Unknown distribution: {}'.format(value))

    def to_json(self):
        if self == self.UNIFORM:
            return 'uniform'
        else:
            return 'log-uniform'


class ParamDomain(object):
    @staticmethod
    def ask_independent_values(params_domain, callback):
        values = []
        params = {}
        for p in params_domain:
            if isinstance(p, ConditionalParam):
                if not p.condition.is_satisfied(params):
                    values.append(ConditionalValue(None))
                    continue
                p = p.param

            v = callback(p)
            values.append(v)
            params[p.name] = (p, v)

        return values

    @staticmethod
    def from_json(p):
        kind, data = list(p.items())[0]
        if kind == 'continuous':
            return ContinuousParam(name=data['name'],
                                   low=data['range']['low'],
                                   high=data['range']['high'],
                                   distribution=Distribution.from_json(data['distribution']))
        elif kind == 'discrete':
            return DiscreteParam(name=data['name'],
                                 low=data['range']['low'],
                                 high=data['range']['high'])
        elif kind == 'categorical':
            return CategoricalParam(name=data['name'], choices=data['choices'])
        elif kind == 'conditional':
            condition = Condition.from_json(data['condition'])
            param = ParamDomain.from_json(data['param'])
            return ConditionalParam(condition, param)
        else:
            raise NotImplementedError('{}'.format(p))


class Condition(object):
    @staticmethod
    def from_json(o):
        kind, data = list(o.items())[0]
        if kind == 'member':
            return ConditionMember(data['name'], data['choices'])
        else:
            raise NotImplementedError('{}'.format(o))


class ConditionMember(object):
    def __init__(self, operand_name, choices):
        self.operand_name = operand_name
        self.choices = choices

    def to_json(self):
        return {'member': {'name': self.operand_name, 'choices': self.choices}}

    def is_satisfied(self, params):
        if self.operand_name not in params:
            return False

        domain, value = params[self.operand_name]
        category = domain.choices[value.index]
        return category in self.choices


class ConditionalParam(object):
    def __init__(self, condition, param):
        self.condition = condition
        self.param = param

    @property
    def name(self):
        return self.param.name

    def to_json(self):
        return {
            'conditional': {
                'condition': self.condition.to_json(),
                'param': self.param.to_json()
            }
        }


class ContinuousParam(object):
    def __init__(self, name, low, high, distribution=Distribution.UNIFORM):
        self.name = name
        self.low = low
        self.high = high
        self.distribution = distribution

    def to_json(self):
        return {
            'continuous': {
                'name': self.name,
                'distribution': self.distribution.to_json(),
                'range': {
                    'low': self.low,
                    'high': self.high
                }
            }
        }


class DiscreteParam(object):
    def __init__(self, name, low, high):
        self.name = name
        self.low = low
        self.high = high

    def to_json(self):
        return {'discrete': {'name': self.name, 'range': {'low': self.low, 'high': self.high}}}

    def make_value(self, value):
        return DiscreteValue(value)


class CategoricalParam(object):
    def __init__(self, name, choices):
        self.name = name
        self.choices = choices

    def to_json(self):
        return {'categorical': {'name': self.name, 'choices': self.choices}}

    def make_value(self, category):
        return CategoricalValue(self.choices.index(category))


class ContinuousValue(object):
    def __init__(self, value):
        self.value = value

    def to_json(self):
        return {'continuous': self.value}


class ConditionalValue(object):
    def __init__(self, value):
        self.value = value

    def to_json(self):
        return {'conditional': self.value}


class DiscreteValue(object):
    def __init__(self, value):
        self.value = value

    def to_json(self):
        return {'discrete': self.value}


class CategoricalValue(object):
    def __init__(self, index):
        self.index = index

    def to_json(self):
        return {'categorical': self.index}
