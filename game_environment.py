import numpy as np
from parameters import Parameters

#parameters
parameters = Parameters()

player_size = parameters.player_size
friction = parameters.friction
move_multiplier = parameters.move_multiplier
grid_size = parameters.grid_size
number_of_squares = parameters.number_of_squares
time_square_on = parameters.time_square_on
time_between_squares = parameters.time_between_squares
collision_constant = parameters.collision_constant
wall_collision_constant = parameters.wall_collision_constant

#calculations
square_size = grid_size/number_of_squares
window_width = grid_size + 500
window_height = grid_size

#game classes
class GameElement:
    def __init__(self, number_of_players, frame_rate=30):
        self.number_of_players = number_of_players
        self.player_size = player_size
        self.friction = friction
        self.move_multiplier = move_multiplier
        self.grid_size = grid_size
        self.frame_rate = frame_rate
        self.number_of_squares = number_of_squares
        self.time_square_on = time_square_on
        self.collision_constant = collision_constant
        self.wall_collision_constant = wall_collision_constant
        self.square_size = square_size
        self.window_width = window_width
        self.window_height = window_height
        self.time_between_squares = time_between_squares

class Player(GameElement):
    def __init__(self, start_pos, number_of_players=2, frame_rate=30):

        GameElement.__init__(self, number_of_players=number_of_players, frame_rate=frame_rate)

        self.pos = start_pos
        self.vel = np.array([0.,0.])
        self.acc = np.array([0.,0.])
        self.size = self.player_size

    def queue_move(self, move_dir):
        self.acc = np.array([0.,0.])
        self.acc += move_dir * self.move_multiplier

    def queue_force(self, force):
        self.acc += force

    def move(self):
        if np.linalg.norm(self.vel) != 0:
            self.vel += self.acc
            self.vel -= (self.vel) * self.friction
        else:
            self.vel += self.acc

        self.pos += self.vel/self.frame_rate


    def get_info(self):
        return np.concatenate([self.pos, self.vel, self.acc])

class Objectives(GameElement):
    def __init__(self, number_of_players=2, frame_rate=30):

        GameElement.__init__(self, number_of_players=number_of_players, frame_rate=frame_rate)

        self.number = self.number_of_players
        self.coordinates = None
        self.pixel_coords = None
        self.on_buttons = np.zeros(self.number)
        self.coordinates_visible = False
        self.get_coordinates()

    def get_coordinates(self):
        coord_1 = np.random.choice(self.number_of_squares, size=self.number, replace=False)
        coord_2 = np.random.choice(self.number_of_squares, size=self.number, replace=False)
        self.coordinates = [np.array([coord_1[i], coord_2[i]]) for i in range(self.number)]
        self.pixel_coords = [coord * self.grid_size / self.number_of_squares for coord in self.coordinates]

        self.coordinates_visible = False

    def check_on(self, player_positions):
        on = np.zeros(self.number)
        for i in range(self.number):
            for pos in player_positions:
                if pos[0] >= self.pixel_coords[i][0] and pos[0] <= self.pixel_coords[i][0] + square_size:
                    if pos[1] >= self.pixel_coords[i][1] and pos[1] <= self.pixel_coords[i][1] + square_size:
                        on[i] = 1
        self.on_buttons = on

    def make_squares_visible(self):
        self.coordinates_visible = True

    def get_info(self):
        if self.coordinates_visible:
            return np.concatenate(self.pixel_coords)
        else:
            return np.concatenate([np.array([None, None]) for _ in range(self.number_of_players)])

class DiscoGame(GameElement):
    def __init__(self, number_of_rounds, number_of_players=2, training=False, frame_rate=30):

        GameElement.__init__(self, number_of_players=number_of_players, frame_rate=frame_rate)

        self.players = [Player(np.random.uniform(self.grid_size, size=2), frame_rate=frame_rate) for _ in range(self.number_of_players)]
        self.objectives = Objectives(number_of_players, frame_rate=frame_rate)
        self.number_of_rounds = number_of_rounds
        self.score = 0
        self.round = 1
        self.time = 0
        self.recently_changed = True
        self.recently_visible = False
        self.time_to_go = (self.time_square_on - (self.time % self.time_square_on)) /1000
        self.playing = True
        self.training = training

    def check_player_collisions(self):
        colliding_players = []
        player_pairs = [[(a.pos,b.pos), (ida, idb + ida + 1)] for ida, a in enumerate(self.players) for idb, b in enumerate(self.players[ida + 1:])]
        for pair in player_pairs:
            if np.linalg.norm(pair[0][0] - pair[0][1]) < self.player_size*2:
                colliding_players.append((pair[1][0], pair[1][1]))
        return colliding_players

    def handle_collisions(self):
        collisions = self.check_player_collisions()
        for collision in collisions:
            player1 = self.players[collision[0]]
            player2 = self.players[collision[1]]
            pos1 = player1.pos
            pos2 = player2.pos
            normal_from_1 = (pos2 - pos1)/np.linalg.norm(pos2 - pos1)**2
            combined_velocity = (np.linalg.norm(player1.vel) + np.linalg.norm(player2.vel))
            player1.queue_force(-normal_from_1 * self.collision_constant * combined_velocity)
            player2.queue_force(normal_from_1 * self.collision_constant * combined_velocity)

    def check_wall_collisions(self):
        colliding_players = []
        for idx, player in enumerate(self.players):
            if player.pos[0] <= self.player_size:
                colliding_players.append([idx, np.array([1,0]), player.pos[0]])
            if player.pos[1] <= self.player_size:
                colliding_players.append([idx, np.array([0,1]), player.pos[1]])
            if player.pos[0] >= self.grid_size - self.player_size:
                colliding_players.append([idx, np.array([-1,0]), self.grid_size - player.pos[0]])
            if player.pos[1] >= self.grid_size - self.player_size:
                colliding_players.append([idx, np.array([0,-1]), self.grid_size - player.pos[1]])
        return colliding_players

    def handle_wall_collisions(self):
        colliding_players = self.check_wall_collisions()
        for collision in colliding_players:
            self.players[collision[0]].pos = np.maximum(np.minimum(self.grid_size - self.player_size, self.players[collision[0]].pos), self.player_size)

            self.players[collision[0]].queue_force(collision[1] * self.wall_collision_constant * np.linalg.norm(self.players[collision[0]].vel))

            if collision[1][0] == 0:
                self.players[collision[0]].vel[1] = 0
            else:
                self.players[collision[0]].vel[0] = 0

    def run_frame(self, commands):
        reward_this_iter = 0

        if not self.playing:
            return None, None

        self.time += 1000 * 1/self.frame_rate
        if self.time % (self.time_square_on + self.time_between_squares) >= 0 and self.time % (self.time_square_on + self.time_between_squares) <= 1000/self.frame_rate and not self.recently_changed:

            if sum(self.objectives.on_buttons) == self.number_of_players:
                self.score += 1
                reward_this_iter += 1
            else:
                reward_this_iter -= 0.1

            self.objectives.get_coordinates()
            self.recently_changed = True
            self.round += 1

            if self.round > self.number_of_rounds:
                self.playing = False

        if self.time % self.time_square_on > 100:
            self.recently_changed = False

        if self.time % (self.time_square_on + self.time_between_squares) >= self.time_between_squares and self.time % (self.time_square_on + self.time_between_squares) <= self.time_between_squares + 1000/self.frame_rate and not self.recently_visible:
            self.objectives.make_squares_visible()
            self.recently_visible = True

        if self.time % (self.time_square_on + self.time_between_squares) > self.time_between_squares + 100:
            self.recently_visible = False

        moves = [np.array([0., 0.]) for _ in range(self.number_of_players)]

        for i in range(self.number_of_players):
            if commands[4*i + 1]==1:
                moves[i] += np.array([-1.,0.])
            if commands[4*i + 3]==1:
                moves[i] += np.array([1.,0.])
            if commands[4*i + 0]==1:
                moves[i] += np.array([0.,-1.])
            if commands[4*i + 2]==1:
                moves[i] += np.array([0.,1.])

            self.players[i].queue_move(moves[i])

        self.handle_collisions()
        self.handle_wall_collisions()

        for player in self.players:
            player.move()

        self.objectives.check_on([self.players[i].pos for i in range(self.number_of_players)])
        self.time_to_go = (self.time_square_on + self.time_between_squares - (self.time % (self.time_square_on + self.time_between_squares))) /1000

        if self.training:
            return self.return_state(), reward_this_iter

    def run_multiple_frames(self, commands, number_of_frames):
        for _ in range(number_of_frames):
            self.run_frame(commands)

    def return_state(self):
        player_info =  np.concatenate([player.get_info() for player in self.players])
        goal_info = self.objectives.get_info()
        additional_info = np.array([self.score, self.time_to_go, self.number_of_rounds - self.round])
        return np.concatenate([player_info, goal_info, additional_info])

    def reset(self):
        self.__init__(self.number_of_rounds, self.number_of_players, self.training, self.frame_rate)
