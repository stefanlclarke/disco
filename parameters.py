#all parameters for game and experiments
import pickle

#game parameters
player_size = 10
friction = 0.1
move_multiplier = 40
grid_size = 600
number_of_squares = 2
time_square_on = 6000
time_between_squares = 0
collision_constant = 10
wall_collision_constant = 0.5
number_of_players = 1

#neural network parameters
actor_num_hidden = 1
actor_size_hidden = 30
critic_num_hidden = 1
critic_size_hidden = 30
exp_constant = 0.05

#training parameters
number_of_rounds = 20
frame_rate = 3
rounds_per_batch = 1
gamma = 0.99
learning_rate = 0.0004
max_grad_norm = 1.
entropy_constant = 0.1
actor_constant = 1.
epochs = 100000
eligibility_trace = True
lambda_actor = 0.9
lambda_critic = 0.

#saving parameters
save_freq = 500

class Parameters:
    def __init__(self):
        self.player_size = player_size
        self.friction = friction
        self.move_multiplier = move_multiplier
        self.grid_size = grid_size
        self.number_of_squares = number_of_squares
        self.time_square_on = time_square_on
        self.time_between_squares = time_between_squares
        self.collision_constant = collision_constant
        self.wall_collision_constant = wall_collision_constant
        self.number_of_players = 1

        #neural network parameters
        self.actor_num_hidden = actor_num_hidden
        self.actor_size_hidden = actor_size_hidden
        self.critic_num_hidden = critic_num_hidden
        self.critic_size_hidden = critic_size_hidden
        self.exp_constant = exp_constant

        #training parameters
        self.number_of_rounds = number_of_rounds
        self.frame_rate = frame_rate
        self.rounds_per_batch = rounds_per_batch
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.max_grad_norm = max_grad_norm
        self.entropy_constant = entropy_constant
        self.actor_constant = actor_constant
        self.epochs = epochs
        self.eligibility_trace = eligibility_trace
        self.lambda_actor = lambda_actor
        self.lambda_critic = lambda_critic

        #saving parameters
        self.save_freq = save_freq

    def save(self, name):
        with open('./bots/{}.pkl'.format(name), 'rb') as outp:
            pickle.dump(self, outp, pickle.HIGHEST_PROTOCOL)

def load_parameters(name):
    with open('./bots/{}.pkl'.format(name), 'rb') as inp:
        parameters = pickle.load(inp)
        return parameters
