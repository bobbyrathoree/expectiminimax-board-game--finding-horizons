#!/usr/local/bin/python3

"""
This is where you should write your AI code!

Authors: Kelly Wheeler (kellwhee), Neha Supe (nehasupe), Bobby Rathore (brathore)

Based on skeleton code by Abhilash Kuhikar, October 2019
"""

from logic_IJK import Game_IJK
import random
import math

# Suggests next move to be played by the current player given the current game
#
# inputs:
#     game : Current state of the game
#
# This function should analyze the current state of the game and determine the
# best move for the current player. It should then call "yield" on that move.

def isGameOver(board):
#None game is not over
#-1: game is a draw
#0: player + wins
#1: player - wins
    moreMoves = -1
    for i in range(len(board)):
        for j in range(len(board[0])):
            if board[i][j] == 'K':
                return 0
            if board[i][j] == 'k':
                return 1
            if board[i][j] == ' ':
                moreMoves = None
    return moreMoves

#+:1
#-:-1
#draw:0
def minimax(game, board, depth, player):
    
#is the game over, return draw or winner
    terminate = isGameOver(board)
    if terminate == 0:
        #player + wins
        return None, 1
    if terminate == 1:
        #player - wins
        return None, -1
    if terminate == -1:
        #draw
        return None, 0

    if depth == 0:
        #hit depth game over
        return None, 0

    values = []
    moves = []
    for move in ["U", "D", "L", "R", "S"]:
        child = game.makeMove(move)
        value = minimax(game, child, depth - 1, player)[1]
        values.append(value)
        moves.append(move)

    if player == '+':
        bestVal = max(values)
    else: #player is -
        bestVal = min(values)
    bestMove = moves[values.index(bestVal)]
    return bestMove, bestVal

def alphaBetaMinimax(game, board, depth, player, alpha, beta):
#is the game over, return draw or winner
    terminate = isGameOver(board)
    if terminate == 0:
        #player + wins
        return None, 1
    if terminate == 1:
        #player - wins
        return None, -1
    if terminate == -1:
        #draw
        return None, 0

    if depth == 0:
        #hit depth game over
        return None, 0

    values = []
    moves = []
    for move in ["U", "D", "L", "R", "S"]:
        game.makeMove(move)
        child = game.getGame()
        value = alphaBetaMinimax(game, child, depth - 1, player, alpha, beta)[1]
        values.append(value)
        moves.append(move)
        if player == '+':
            if value > alpha:
                alpha = value
        elif player == '-':
            if value < beta:
                beta = value

        if alpha >= beta:
            break

    if player == '+':
        bestVal = max(values)
    else: #player is -
        bestVal = min(values)
    bestMove = moves[values.index(bestVal)]
    return bestMove, bestVal


def next_move(game: Game_IJK) -> None:

    """board: list of list of strings -> current state of the game
       current_player: int -> player who will make the next move either ('+') or -'-')
       deterministic: bool -> either True or False, indicating whether the game is deterministic or not
    """

    board = game.getGame()
    player = game.getCurrentPlayer()
    deterministic = game.getDeterministic()

    move, score = alphaBetaMinimax(game, board, 1000, player, -math.inf, math.inf)
    yield move

    #yield random.choice(["U", "D", "L", "R", "S"])
