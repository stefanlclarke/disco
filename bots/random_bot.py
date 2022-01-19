import numpy as np

class RandomAgent:
    def __init__(self, decision_rate=10):
        self.decision_rate = decision_rate

    def move(self, input):
        return np.random.choice(2, size=4)
