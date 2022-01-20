import torch
import torch.nn as nn
import numpy as np
from game_environment import DiscoGame
from networks import ActorCritic, Memory
import torch.optim as optim
import copy
from torch.distributions import Bernoulli
from parameters import Parameters, load_parameters

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

parameters = Parameters()

#number of games to be played per episode:
number_of_rounds = parameters.number_of_rounds
number_of_players = parameters.number_of_players
epochs = parameters.epochs
save_freq = parameters.save_freq

#import and prepare bots here:
ac = ActorCritic(number_of_players).to(device)
#ac.load_state_dict(torch.load('./ac_models/10jan_6_9350'))
ac.train_mode()

#clips weights to reasonable values
class WeightClipper(object):
    def __init__(self, frequency=5):
        self.frequency = frequency

    def __call__(self, module):
        if hasattr(module, 'weight'):
            w = module.weight.data
            w = w.clamp(-1,1)

#class handling all of training
class SinglePlayerTrainer:
    def __init__(self, ac):
        self.ac = ac
        self.memory = Memory()
        self.game = DiscoGame(number_of_rounds, number_of_players=number_of_players, training=True, frame_rate=parameters.frame_rate)
        self.rounds_per_batch = parameters.rounds_per_batch
        self.gamma = parameters.gamma
        self.learning_rate = parameters.learning_rate
        self.optimizer = optim.Adam(self.ac.parameters(), lr=self.learning_rate)
        self.clipper = WeightClipper()
        self.max_grad_norm = parameters.max_grad_norm
        self.entropy_constant = parameters.entropy_constant
        self.actor_constant = parameters.actor_constant
        self.eligibility_trace = parameters.eligibility_trace
        self.lambda_actor = parameters.lambda_actor
        self.lambda_critic = parameters.lambda_critic

    def play_one_round(self):
        running = True
        self.game.reset()
        while running:
            state = self.game.return_state()
            command = self.ac.move(state)
            command_numpy = np.concatenate([command.cpu().detach().numpy()])
            command_oh = np.zeros(4)
            for i in range(4):
                try:
                    command_oh[i] = (np.random.choice(2, p=[1-command_numpy[i], command_numpy[i]]))
                except ValueError:
                    print('NaN Error')
                    print('command {}'.format(command_numpy))
                    print('model')
                    for name, param in self.ac.named_parameters():
                        if param.requires_grad:
                            print(name, param.data)
                            raise ValueError('NaN Error')

            critic_value = self.ac.value(state)
            new_state, reward = self.game.run_frame(command_oh)

            if new_state is None:
                running = False
                break

            self.memory.save(state, command, reward, critic_value, command_oh)

    def learn(self):

        Qvals = []
        Qval =  0
        for i in reversed(range(len(self.memory.rewards))):
            Qval_new = self.memory.rewards[i] + self.gamma * Qval
            Qvals.append(Qval_new)
            Qval = Qval_new
        Qvals = np.array(Qvals)

        #log_probs1 = torch.cat([(torch.log(self.memory.actions[i])*(torch.from_numpy(self.memory.oh_actions[i])).to(device).sum().unsqueeze(0)) for i in range(len(self.memory.actions))])
        #log_probs2 = torch.cat([(torch.log(1 - self.memory.actions[i])*(torch.from_numpy(1-self.memory.oh_actions[i])).to(device)).sum().unsqueeze(0) for i in range(len(self.memory.actions))])

        log_probs1 = torch.cat([((torch.log(self.memory.actions[i])*(torch.from_numpy(self.memory.oh_actions[i]).to(device))).sum().unsqueeze(0)) for i in range(len(self.memory.actions))])
        log_probs2 = torch.cat([((torch.log(1 - self.memory.actions[i])*(torch.from_numpy(1 - self.memory.oh_actions[i]).to(device))).sum().unsqueeze(0)) for i in range(len(self.memory.actions))])


        log_probs = log_probs1 + log_probs2
        advantage = torch.from_numpy(Qvals).to(device) - torch.cat(self.memory.critic_memory)
        #log_probs = torch.cat([torch.unsqueeze(log_prob, 0) for log_prob in log_probs])
        entropy = -torch.stack([torch.stack([torch.minimum(Bernoulli(action[i]).entropy(), torch.tensor(0.3)) for action in self.memory.actions]).sum() for i in range(4)]).sum()
        detached_advantage = advantage.detach()

        if self.eligibility_trace:
            elig_traces_actor = torch.zeros(len(log_probs)).to(device)
            elig_traces_critic = torch.zeros(len(log_probs)).to(device)
            elig_trace_actor = 0
            elig_trace_critic = 0

            for i in range(len(log_probs)):
                elig_trace_actor = self.gamma * self.lambda_actor * elig_trace_actor + log_probs[i]
                elig_traces_actor[i] = elig_trace_actor
                elig_trace_critic = self.gamma * self.lambda_critic * elig_trace_critic + self.memory.critic_memory[i]
                elig_traces_critic[i] = elig_trace_critic

            actor_loss = - (elig_traces_actor * (detached_advantage.to(device))).mean()
            critic_loss = (0.5 * advantage.pow(2)).mean() #* elig_traces_critic).mean()

        else:
            actor_loss = - (log_probs * detached_advantage.to(device)).mean()
            critic_loss = 0.5 * advantage.pow(2).mean()

        ac_loss = self.entropy_constant * entropy.to(device) + (self.actor_constant * actor_loss + critic_loss)
        ac_loss.backward(retain_graph=False)

        nn.utils.clip_grad_norm_(self.ac.parameters(), self.max_grad_norm)

        self.optimizer.step()

        for name, param in self.ac.named_parameters():
            if param.requires_grad:
                if torch.isnan(param).any():
                    print('rewards {}'.format(self.memory.rewards))
                    print('actions {}'.format(self.memory.actions))
                    print('choices {}'.format(self.memory.oh_actions))

                    print('Q Values {}'.format(Qvals))
                    print('log_probs', log_probs)

                    print('actor loss {}'.format(actor_loss))
                    print('critic_loss {}'.format(critic_loss))
                    print('entropy {}'.format(entropy))
                    print('advantage {}'.format(advantage))

                    print('after step')
                    for name, param in self.ac.named_parameters():
                        if param.requires_grad:
                            print(name, param.data)

                    raise ValueError

        self.memory.reset()

    def train(self, iterations, save_freq, name):
        scores = 0
        freq = 0
        for _ in range(iterations):
            self.play_one_round()
            scores += self.game.score
            self.learn()
            freq += 1
            if freq % save_freq == 0:
                self.save(str(freq), name)
                print('average score {}'.format(scores/save_freq))
                scores = 0

    def save(self, freq, name):
        torch.save(self.ac.state_dict(), './ac_models/{}'.format(name + freq))
        print('saved model {}'.format(name + freq))


trainer = SinglePlayerTrainer(ac)
trainer.train(epochs, save_freq, '12jan_ET_')
#print(trainer.memory.states)
