import numpy as np
import pygame
from game_environment import DiscoGame
import pygame.gfxdraw
import math
from draw_functions import draw_board

number_of_rounds = 10
frame_rate = 30

game = DiscoGame(number_of_rounds, frame_rate=frame_rate)

pygame.init()
win = pygame.display.set_mode((game.window_width, game.window_height))
pygame.display.set_caption("Game")
font = pygame.font.SysFont('freesansbold.ttf', 80)
small_font = pygame.font.SysFont('freesansbold.ttf', 20)
clock = pygame.time.Clock()

run = True
while run:
    clock.tick(game.frame_rate)

    pygame.event.pump()
    keys = pygame.key.get_pressed()

    commands = np.zeros(8)

    if keys[pygame.K_a]:
        commands[1] = 1
    if keys[pygame.K_d]:
        commands[3] = 1
    if keys[pygame.K_w]:
        commands[0] = 1
    if keys[pygame.K_s]:
        commands[2] = 1

    if keys[pygame.K_h]:
        commands[5] = 1
    if keys[pygame.K_k]:
        commands[7] = 1
    if keys[pygame.K_u]:
        commands[4] = 1
    if keys[pygame.K_j]:
        commands[6] = 1

    game.run_frame(commands)

    draw_board(win, game, commands, font, small_font)
    pygame.display.update()

    if keys[pygame.K_RETURN] and not game.playing:
        game.reset()

    for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
