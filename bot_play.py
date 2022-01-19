import numpy as np
import pygame
from game_environment import DiscoGame
import pygame.gfxdraw
import math
from draw_functions import draw_board
from bots.team import Team
import time
from networks import ActorCritic
import torch
from parameters import Parameters

parameters = Parameters()

#number of games to be played:
number_of_rounds = 100

#import and prepare bots here:
from bots.random_bot import RandomAgent
bot1 = RandomAgent(decision_rate=parameters.frame_rate)
bot2 = RandomAgent(decision_rate=parameters.frame_rate)
ac = ActorCritic(1)
ac.load_state_dict(torch.load('./ac_models/11jan_100000'))

ac.train_mode()


bots = [ac]
#preparing team and game:
team = Team(bots, random_moves = True)
game = DiscoGame(number_of_rounds, number_of_players=len(bots))

#pygame code:
pygame.init()
win = pygame.display.set_mode((game.window_width, game.window_height))
pygame.display.set_caption("Game")
font = pygame.font.SysFont('freesansbold.ttf', 80)
small_font = pygame.font.SysFont('freesansbold.ttf', 20)
clock = pygame.time.Clock()

run = True
last_wait = 0
while run:
    clock.tick(game.frame_rate)

    pygame.event.pump()
    keys = pygame.key.get_pressed()

    input = game.return_state()
    commands = team.move(input)

    game.run_frame(commands)

    draw_board(win, game, commands, font, small_font, other_info=commands)
    pygame.display.update()

    if keys[pygame.K_RETURN] and not game.playing:
        game.reset()

    if keys[pygame.K_SPACE] and time.time() - last_wait > 0.1:
        waiting = True
        wait_start = time.time()
        while waiting:
            for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        run = False

            if (time.time() - wait_start) > 0.1:

                pygame.event.pump()
                keys = pygame.key.get_pressed()
                if keys[pygame.K_SPACE]:
                    waiting = False
                    last_wait = time.time()

    for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
