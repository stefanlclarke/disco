import pygame
import pygame.gfxdraw
import numpy as np
import math

player_size = 10
friction = 0.1
move_multiplier = 40
grid_size = 600
frame_rate = 30
number_of_squares = 10
time_square_on = 4000
time_between_squares = 1000
collision_constant = 10
wall_collision_constant = 0.5

square_size = grid_size/number_of_squares

window_width = grid_size + 500
window_height = grid_size

pygame.init()
win = pygame.display.set_mode((window_width, window_height))
pygame.display.set_caption("Game")
font = pygame.font.SysFont('freesansbold.ttf', 80)
clock = pygame.time.Clock()

class Player(object):
    def __init__(self, start_pos):
        self.pos = start_pos
        self.vel = np.array([0.,0.])
        self.acc = np.array([0.,0.])
        self.size = player_size

    def queue_move(self, move_dir):
        self.acc += move_dir * move_multiplier

    def queue_force(self, force):
        self.acc += force

    def move(self):
        if np.linalg.norm(self.vel) != 0:
            self.vel += self.acc
            self.vel -= (self.vel) * friction
        else:
            self.vel += self.acc

        self.pos += self.vel/frame_rate
        self.acc = 0

class Objectives(object):
    def __init__(self, number_of_players):
        self.number = number_of_players
        self.coordinates = None
        self.pixcel_coords = None
        self.on_buttons = np.zeros(self.number)
        self.off_colours = [(0,102,51),(0,225,0)]
        self.on_colours = [(204,102,0), (255,153,51)]

        self.get_coordinates()

    def get_coordinates(self):
        coord_1 = np.random.choice(number_of_squares, size=self.number, replace=False)
        coord_2 = np.random.choice(number_of_squares, size=self.number, replace=False)
        self.coordinates = [np.array([coord_1[i], coord_2[i]]) for i in range(self.number)]
        self.pixel_coords = [coord * grid_size / number_of_squares for coord in self.coordinates]

    def check_on(self, player_positions):
        on = np.zeros(self.number)
        for i in range(self.number):
            for pos in player_positions:
                if pos[0] >= self.pixel_coords[i][0] and pos[0] <= self.pixel_coords[i][0] + square_size:
                    if pos[1] >= self.pixel_coords[i][1] and pos[1] <= self.pixel_coords[i][1] + square_size:
                        on[i] = 1
        self.on_buttons = on

def draw_arrow(screen, colour, start, end):
    pygame.draw.line(screen,colour,start,end,2)
    rotation = math.degrees(math.atan2(start[1]-end[1], end[0]-start[0]))+90
    pygame.draw.polygon(screen, colour, ((end[0]+20*math.sin(math.radians(rotation)), end[1]+20*math.cos(math.radians(rotation))), (end[0]+20*math.sin(math.radians(rotation-120)), end[1]+20*math.cos(math.radians(rotation-120))), (end[0]+20*math.sin(math.radians(rotation+120)), end[1]+20*math.cos(math.radians(rotation+120)))))

def draw_keypad(win, centre, buttons_on, on_col, off_col):
    col = []
    for i in range(len(buttons_on)):
        if buttons_on[i] == 0:
            col.append(off_col)
        else:
            col.append(on_col)
    draw_arrow(win, col[0], centre - np.array([0,10]), centre - np.array([0,30]))
    draw_arrow(win, col[1], centre + np.array([0,10]), centre + np.array([0,30]))
    draw_arrow(win, col[2], centre - np.array([10,0]), centre - np.array([30,0]))
    draw_arrow(win, col[3], centre + np.array([10,0]), centre + np.array([30,0]))

def draw_board(player_positions, player_colours, objectives, buttons_pressed=np.zeros(8), score=0, time=0):
    obj_positions = objectives.pixel_coords
    win.fill((0,0,0))
    pygame.draw.rect(win, (25,51,0), pygame.Rect((0,0),(grid_size, grid_size)))

    for i in range(len(obj_positions)):
        obj = obj_positions[i]
        if objectives.on_buttons[i] == 0:
            pygame.draw.rect(win, objectives.off_colours[0], pygame.Rect((obj[0], obj[1]), (grid_size/number_of_squares, grid_size/number_of_squares)))
            pygame.draw.rect(win, objectives.off_colours[1] , pygame.Rect((obj[0]+2, obj[1]+2), (grid_size/number_of_squares-4, grid_size/number_of_squares-4)))
        else:
            pygame.draw.rect(win, objectives.on_colours[0], pygame.Rect((obj[0], obj[1]), (grid_size/number_of_squares, grid_size/number_of_squares)))
            pygame.draw.rect(win, objectives.on_colours[1], pygame.Rect((obj[0]+2, obj[1]+2), (grid_size/number_of_squares-4, grid_size/number_of_squares-4)))


    for i in range(len(player_positions)):
        pos = player_positions[i]
        colour = player_colours[i]
        pygame.gfxdraw.filled_circle(win, int(pos[0]), int(pos[1]), player_size, (255,255,255))
        pygame.gfxdraw.filled_circle(win, int(pos[0]), int(pos[1]), player_size-2, colour)
        pygame.gfxdraw.aacircle(win, int(pos[0]), int(pos[1]), player_size, (255,255,255))

    draw_keypad(win, np.array([grid_size + 100,70]), buttons_pressed[:4], (255,255,255), (255,0,0))
    draw_keypad(win, np.array([grid_size + 400,70]), buttons_pressed[4:], (255,255,255), (0,0,255))

    pygame.draw.rect(win, (178,102,255), pygame.Rect((grid_size+218,38), (62,62)))
    pygame.draw.rect(win, (255,153,255), pygame.Rect((grid_size+220,40), (60,60)))

    pygame.draw.rect(win, (178,102,255), pygame.Rect((grid_size+218,498), (62,62)))
    pygame.draw.rect(win, (204,0,102), pygame.Rect((grid_size+220,500), (60,60)))

    score_text = font.render('{}'.format(score), True, (255,255,255))
    win.blit(score_text, np.array([grid_size + 233,45]))

    time_text = font.render('{}'.format(time), True, (255,255,255))
    win.blit(time_text, np.array([grid_size + 233,505]))

def check_player_collisions(players):
    colliding_players = []
    player_pairs = [[(a.pos,b.pos), (ida, idb + ida + 1)] for ida, a in enumerate(players) for idb, b in enumerate(players[ida + 1:])]
    for pair in player_pairs:
        if np.linalg.norm(pair[0][0] - pair[0][1]) < player_size*2:
            colliding_players.append((pair[1][0], pair[1][1]))
    return colliding_players

def handle_collisions(players):
    collisions = check_player_collisions(players)
    for collision in collisions:
        player1 = players[collision[0]]
        player2 = players[collision[1]]
        pos1 = player1.pos
        pos2 = player2.pos
        print(pos1)
        print(pos2)
        normal_from_1 = (pos2 - pos1)/np.linalg.norm(pos2 - pos1)**2
        combined_velocity = (np.linalg.norm(player1.vel) + np.linalg.norm(player2.vel))
        player1.queue_force(-normal_from_1 * collision_constant * combined_velocity)
        player2.queue_force(normal_from_1 * collision_constant * combined_velocity)

def check_wall_collisions(players):
    colliding_players = []
    for idx, player in enumerate(players):
        if player.pos[0] <= player_size:
            colliding_players.append([idx, np.array([1,0]), player.pos[0]])
        if player.pos[1] <= player_size:
            colliding_players.append([idx, np.array([0,1]), player.pos[1]])
        if player.pos[0] >= grid_size - player_size:
            colliding_players.append([idx, np.array([-1,0]), grid_size - player.pos[0]])
        if player.pos[1] >= grid_size - player_size:
            colliding_players.append([idx, np.array([0,-1]), grid_size - player.pos[1]])
    return colliding_players

def handle_wall_collisions(players):
    colliding_players = check_wall_collisions(players)
    for collision in colliding_players:
        players[collision[0]].pos = np.maximum(np.minimum(grid_size - player_size, players[collision[0]].pos), player_size)

        players[collision[0]].queue_force(collision[1] * wall_collision_constant * np.linalg.norm(players[collision[0]].vel))

        if collision[1][0] == 0:
            players[collision[0]].vel[1] = 0
        else:
            players[collision[0]].vel[0] = 0
        print(collision)


player1 = Player(np.array([grid_size/2, grid_size/3]))
player2 = Player(np.array([grid_size/3, grid_size/2]))

squares_on = False
recently_changed = False

objectives = Objectives(2)

players = [player1, player2]

buttons_pressed = np.zeros(8)
score = 0


run = True
while run:
    clock.tick(frame_rate)

    time = pygame.time.get_ticks()
    if time % time_square_on >= 0 and time % time_square_on <= 100 and not recently_changed:

        if sum(objectives.on_buttons) == 2:
            score += 1

        coord_1 = np.random.choice(number_of_squares, size=2, replace=False)
        coord_2 = np.random.choice(number_of_squares, size=2, replace=False)

        objectives.get_coordinates()

        recently_changed = True

    if time % time_square_on > 100:
        recently_changed = False

    pygame.event.pump()
    keys = pygame.key.get_pressed()

    move1 = np.array([0., 0.])
    move2 = np.array([0., 0.])

    if keys[pygame.K_a]:
        move1 += np.array([-1.,0.])
        buttons_pressed[2] = 1
    if keys[pygame.K_d]:
        move1 += np.array([1.,0.])
        buttons_pressed[3] = 1
    if keys[pygame.K_w]:
        move1 += np.array([0.,-1.])
        buttons_pressed[0] = 1
    if keys[pygame.K_s]:
        move1 += np.array([0.,1.])
        buttons_pressed[1] = 1

    if keys[pygame.K_h]:
        move2 += np.array([-1.,0.])
        buttons_pressed[6] = 1
    if keys[pygame.K_k]:
        move2 += np.array([1.,0.])
        buttons_pressed[7] = 1
    if keys[pygame.K_u]:
        move2 += np.array([0.,-1.])
        buttons_pressed[4] = 1
    if keys[pygame.K_j]:
        move2 += np.array([0.,1.])
        buttons_pressed[5] = 1

    player1.queue_move(move1)
    player2.queue_move(move2)

    handle_collisions(players)
    handle_wall_collisions(players)
    player1.move()
    player2.move()

    objectives.check_on([player1.pos, player2.pos])

    time_to_go = int((time_square_on - (time % time_square_on)) /1000) + 1
    draw_board([player1.pos, player2.pos], [(255, 0, 0), (0, 0, 255)], objectives, buttons_pressed, score, time_to_go)
    buttons_pressed = np.zeros(8)

    pygame.display.update()

    for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
