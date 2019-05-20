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
        else:
            raise NotImplementedError('{}'.format(p))


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
