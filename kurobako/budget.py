class Budget(object):
    def __init__(self, amount, consumption=0):
        self.consumption = consumption
        self.amount = amount

    @staticmethod
    def from_json(o):
        return Budget(amount=o['amount'], consumption=o['consumption'])

    def to_json(self):
        return {'consumption': self.consumption, 'amount': self.amount}
