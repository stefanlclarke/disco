import numpy as np

class Team:
    def __init__(self, members, frame_rate=30, random_moves=False):
        self.members = members
        self.decision_rates = [member.decision_rate for member in members]
        self.decision_breaks = [frame_rate/dr for dr in self.decision_rates]
        self.frame_rate = frame_rate
        self.size = len(members)
        self.frame = 0
        self.current_moves = [None for _ in range(len(members))]
        self.team_size = len(members)
        self.random_moves = random_moves

    def move(self, input):
        for i in range(self.team_size):
            if self.frame % self.decision_breaks[i] == 0:
                if not self.random_moves:
                    self.current_moves[i] = self.members[i].move(input)
                else:
                    move = self.members[i].move(input).detach().numpy()
                    print(move)
                    oh_move = np.zeros(4)
                    for j in range(4):
                        oh_move[j] = np.random.choice(2, p=[1-move[j], move[j]])
                    self.current_moves[i] = oh_move

        self.frame += 1
        return np.concatenate(self.current_moves)
