import numpy as np
import pygame
import pygame.gfxdraw
import math

player_colours = [(255,0,0), (0,0,255)]
off_colours = [(0,102,51),(0,225,0)]
on_colours = [(204,102,0), (255,153,51)]

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
    draw_arrow(win, col[2], centre + np.array([0,10]), centre + np.array([0,30]))
    draw_arrow(win, col[1], centre - np.array([10,0]), centre - np.array([30,0]))
    draw_arrow(win, col[3], centre + np.array([10,0]), centre + np.array([30,0]))


def draw_board(win, game, commands, font, small_font, other_info=None):
    obj_positions = game.objectives.pixel_coords
    player_positions = [player.pos for player in game.players]

    win.fill((0,0,0))
    pygame.draw.rect(win, (25,51,0), pygame.Rect((0,0),(game.grid_size, game.grid_size)))

    if game.objectives.coordinates_visible:
        for i in range(len(obj_positions)):
            obj = obj_positions[i]
            if game.objectives.on_buttons[i] == 0:
                pygame.draw.rect(win, off_colours[0], pygame.Rect((obj[0], obj[1]), (game.square_size, game.square_size)))
                pygame.draw.rect(win, off_colours[1] , pygame.Rect((obj[0]+2, obj[1]+2), (game.square_size-4, game.square_size-4)))
            else:
                pygame.draw.rect(win, on_colours[0], pygame.Rect((obj[0], obj[1]), (game.square_size, game.square_size)))
                pygame.draw.rect(win, on_colours[1], pygame.Rect((obj[0]+2, obj[1]+2), (game.square_size-4, game.square_size-4)))


    for i in range(len(player_positions)):
        pos = player_positions[i]
        colour = player_colours[i]
        pygame.gfxdraw.filled_circle(win, int(pos[0]), int(pos[1]), game.player_size, (255,255,255))
        pygame.gfxdraw.filled_circle(win, int(pos[0]), int(pos[1]), game.player_size-2, colour)
        pygame.gfxdraw.aacircle(win, int(pos[0]), int(pos[1]), game.player_size, (255,255,255))

    draw_keypad(win, np.array([game.grid_size + 100,70]), commands[:4], (255,255,255), (255,0,0))

    if game.number_of_players > 1:
        draw_keypad(win, np.array([game.grid_size + 400,70]), commands[4:], (255,255,255), (0,0,255))

    pygame.draw.rect(win, (178,102,255), pygame.Rect((game.grid_size+218,38), (62,62)))
    pygame.draw.rect(win, (255,153,255), pygame.Rect((game.grid_size+220,40), (60,60)))

    pygame.draw.rect(win, (178,102,255), pygame.Rect((game.grid_size+218,498), (62,62)))
    pygame.draw.rect(win, (204,0,102), pygame.Rect((game.grid_size+220,500), (60,60)))

    pygame.draw.rect(win, (0,102,0), pygame.Rect((game.grid_size+218+167,498), (62,62)))
    pygame.draw.rect(win, (0,255,100), pygame.Rect((game.grid_size+220+167,500), (60,60)))

    score_text = font.render('{}'.format(game.score), True, (255,255,255))
    win.blit(score_text, np.array([game.grid_size + 233,45]))

    time_text = font.render('{}'.format(int(game.time_to_go)+1), True, (255,255,255))
    win.blit(time_text, np.array([game.grid_size + 233,505]))

    round_text = font.render('{}'.format(game.number_of_rounds - int(game.round)+1), True, (255,255,255))
    win.blit(round_text, np.array([game.grid_size + 400,505]))

    state = game.return_state()
    for i in range(len(state)):
        try:
            state_text = small_font.render(str(np.around(state[i], decimals=4)), True, (255,255,255))
        except TypeError:
            state_text = small_font.render(str(state[i]), True, (255,255,255))

        win.blit(state_text, np.array([game.grid_size + 30,200 + 20*i]))

    if other_info is not None:
        other_text = small_font.render('{}'.format(other_info), True, (255,255,255))
        win.blit(other_text, np.array([game.grid_size + 200, 200]))

    if not game.playing:
        pygame.draw.rect(win, (0,0,100), pygame.Rect((game.grid_size//3,game.grid_size//3), (380,90)))
        pygame.draw.rect(win, (0,0,200), pygame.Rect((game.grid_size//3+2,game.grid_size//3+2), (380-2,90-2)))
        over_text = font.render('GAME OVER', True, (255,255,255))
        win.blit(over_text, np.array([game.grid_size//3 + 20,game.grid_size//3 + 20]))
