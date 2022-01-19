import numpy as np
import torch
import torch.nn as nn

from parameters import Parameters
parameters = Parameters()

#parameters
actor_num_hidden = parameters.actor_num_hidden
actor_size_hidden = parameters.actor_size_hidden
critic_num_hidden = parameters.critic_num_hidden
critic_size_hidden = parameters.critic_size_hidden
exp_constant = parameters.exp_constant

class NN(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_hidden, out_dim):
        super(NN, self).__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.input_layer = nn.Linear(in_dim, hidden_dim)
        self.hidden_layers = [nn.Linear(hidden_dim, hidden_dim) for _ in range(num_hidden - 1)]
        for i in range(len(self.hidden_layers)):
            self.add_module("hidden_layer_"+str(i), self.hidden_layers[i])
        self.output_layer = nn.Linear(hidden_dim, out_dim)
        self.relu = nn.ReLU()

    def feed_forward(self, x):
        y = self.relu(self.input_layer(x))
        for layer in self.hidden_layers:
            y = self.relu(layer(y))
        output = self.output_layer(y)
        return output

class Actor(NN):
    def __init__(self, num_players):
        self.input_size = num_players * 8 + 3
        super().__init__(self.input_size, actor_size_hidden, actor_num_hidden, 4)

    def forward(self, x):
        output = self.feed_forward(x)

        if self.training:
            return 1/(1+torch.exp(exp_constant * output))
        else:
            return torch.round(1/(1+torch.exp(exp_constant * output)))

class Critic(NN):
    def __init__(self, num_players):
        self.input_size = num_players * 8 + 3
        super().__init__(self.input_size, critic_size_hidden, critic_num_hidden, 1)

    def forward(self, x):
        return self.feed_forward(x)

class ActorCritic(nn.Module):
    def __init__(self, num_players, decision_rate=5):
        super(ActorCritic, self).__init__()
        self.num_players = num_players
        self.actor = Actor(num_players)
        self.critic  = Critic(num_players)
        self.input_size = self.actor.input_size
        self.decision_rate = decision_rate

    def train_mode(self):
        self.actor.training = True
        self.critic.training = True
        self.training = True

    def test_mode(self):
        self.actor.training = False
        self.critic.training = False
        self.training = False

    def move(self, inp):
        torch_in = torch.nan_to_num(torch.from_numpy(inp.astype('float'))).float()
        if not self.training:
            return self.actor.forward(torch_in).detach().numpy()
        else:
            return self.actor.forward(torch_in)

    def value(self, x):
        return self.critic.forward(torch.nan_to_num(torch.from_numpy(x.astype('float'))).float())

class Memory:
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.critic_memory = []
        self.oh_actions = []

    def reset(self):
        self.__init__()

    def save(self, state, action, reward, critic_value, oh_action):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.critic_memory.append(critic_value)
        self.oh_actions.append(oh_action)
