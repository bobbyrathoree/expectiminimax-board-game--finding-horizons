#!/usr/local/bin/python3

"""
This is where you should write your AI code!

Authors: Kelly Wheeler (kellwhee), Neha Supe (nehasupe), Bobby Rathore (brathore)

Based on skeleton code by Abhilash Kuhikar, October 2019
"""

from logic_IJK import Game_IJK
import random
import math
import copy

# Suggests next move to be played by the current player given the current game
#
# inputs:
#     game : Current state of the game
#
# This function should analyze the current state of the game and determine the
# best move for the current player. It should then call "yield" on that move.

def isGameOver(board):
#None game is not over
#O: game is a draw
#1: player + wins
#-1: player - wins
    moreMoves = 0
    for i in range(len(board)):
        for j in range(len(board[0])):
            if board[i][j] == 'K':
                return 1
            if board[i][j] == 'k':
                return -1
            if board[i][j] == ' ':
                moreMoves = None
    return moreMoves

#+:1
#-:-1
#draw:0

def freeTile(board):
    moreMoves = 0
    for i in range(len(board)):
        for j in range(len(board[0])):
            if board[i][j] == ' ':
                moreMoves+= 1
    if moreMoves > 0.6 * len(board) * len(board[0]):
        return moreMoves* 50 #what value to give dont know but when the board is more than half filled preferrence should be given to merges
    else:
        return 0

def smoothness(board):
    penalty = 0
    #letterValue = {'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 5, 'f': 6, 'g': 7, 'h': 8, 'i': 9, 'j': 10, 'k': 11}
    letterValue = {'a': pow(2, 1), 'b': pow(2, 2), 'c': pow(2, 3), 'd': pow(2,4), 'e': pow(2, 5), 'f': pow(2, 6), 'g': pow(2, 7), 'h':pow(2, 8), 'i': pow(2, 9), 'j': pow(2, 10), 'k': pow(2, 11)}
    for i in range(1, len(board)):
        for j in range(1, len(board[0])):
            if board[i][j] != ' ':
                if board[i-1][j].lower() == board[i][j].lower():
                    penalty = 10 * letterValue[board[i][j].lower()]
                if board[i][j-1].lower()== board[i][j].lower():
                    penalty = 10 * letterValue[board[i][j].lower()]
    return penalty


def numBoard(board):
    #letterValue = {'a': pow(2, 1), 'b': pow(2, 2), 'c': pow(2, 3), 'd': pow(2,4), 'e': pow(2, 5), 'f': pow(2, 6), 'g': pow(2, 7), 'h':pow(2, 8), 'i': pow(2, 9), 'j': pow(2, 10), 'k': pow(2, 11)}
    letterValue = {'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 5, 'f': 6, 'g': 7, 'h': 8, 'i': 9, 'j': 10, 'k': 11}
    newBoard = board
    for i in range(len(board)):
        for j in range(len(board[0])):
            index = board[i][j].lower()
            if index!= ' ':
                newBoard[i][j] = letterValue[index]
            else:
                newBoard[i][j] = 0# or 1
    return newBoard

def monotonocity(board):
    board = numBoard(board)
    gradient = [[[0, -1, -2, -3, -4, -5],[1, 0, -1, -2, -3, -4],[2, 1, 0, -1, -2, -3],[3, 2, 1, 0, -1, -2],[4, 3, 2, 1, 0, -1], [5, 4, 3, 2, 1, 0]],
                [[5, 4, 3, 2, 1, 0], [4, 3, 2, 1, 0, -1], [3, 2, 1, 0, -1, -2], [2, 1, 0, -1, -2, -3], [1, 0, -1, -2, -3, -4], [0, -1, -2, -3, -4, -5]],
                [[0, 1, 2, 3, 4, 5], [-1, 0, 1, 2, 3,4],[-2, -1, 0, 1, 2, 3],[-3, -2, -1, 0, 1, 2], [-4, -3, -2,-1, 0, 1], [-5, -4, -3, -2, -1, 0]],
                [[-5, -4, -3, -2, -1, 0], [-4, -3, -2,-1, 0, 1], [-3, -2, -1, 0, 1, 2], [-2, -1, 0, 1, 2, 3], [-1, 0, 1, 2, 3,4], [0, 1, 2, 3, 4, 5]]]
    value = [0]*len(gradient)
    for i in range(len(gradient)):
        for j in range(len(board)):
            for k in range(len(board[0])):
                value[i] = value[i]+ gradient[i][j][k]*board[j][k]
    return max(value)

def evalHeuristic(board):
    heuristic = freeTile(board) + smoothness(board) + monotonocity(board)
    return heuristic

def miniMax(childBoard, depth, alpha, beta, player):
    if depth == 0:# isgameover should check if there are any moves- any possible merges or blank isGameOver(board) or 
        return evalHeuristic(childBoard.getGame())
    if player:
        maxUtility = -math.inf
        for move in ["L", "R", "U", "D"]:
            copyboard = copy.deepcopy(childBoard)
            newChild = copyboard.makeMove(move)
            newboard = newChild.getGame()
            maxUtility = max(maxUtility, miniMax(newChild, depth-1, alpha, beta, False))
            alpha = max(alpha, maxUtility)
            if beta <= alpha:
                break
        return maxUtility
    else:
        minUtility = math.inf
        for move in ["L", "R", "U", "D"]:
            copyboard = copy.deepcopy(childBoard)
            newChild = copyboard.makeMove(move)
            newboard = newChild.getGame()
            minUtility = min(minUtility, miniMax(newChild, depth-1, alpha, beta, True))
            beta = min(beta, minUtility)
            if beta <= alpha:
                break
        return minUtility

def findOptimalMoveDet(newGame, board, depth, player):
    moves = ["L", "R", "U", "D"]
    maxUtility = []
    nextMove = -1
    realnewGame = copy.deepcopy(newGame)
    for move in moves:
        reallynewGame = copy.deepcopy(newGame)
        child = reallynewGame.makeMove(move)
        childBoard = child.getGame()
        utility = miniMax(child, depth, -math.inf, math.inf, False)
        maxUtility.append(utility)
        # not sure if this part works
        if utility >= max(maxUtility):
            nextMove = move
    return nextMove

# 
def expectedBoards(board, player):
    if player =='+':
        letter = 'A'
    elif player == '-':
        letter = 'a'
    list_boards = []
    for i in range(len(board)):
        for j in range(len(board[0])):
            if board[i][j]==' ':
                newBoard = board
                newBoard[i][j] = letter
                list_boards.append(newBoard)
    return list_boards
'''
def expctiMiniMax(child, depth, player):
    if player == '+':
        otherPlayer = '-'
    elif player == '-':
        otherPlayer = '+'
    max_list_boards = expectedBoards(child.getGame(), player)
    # find children for another layer- list_boards
    min_list_boards = []
    min_layer_expect = {}
    for board in max_list_boards:
        totalHeuristic = 0
        min_list_boards = expectedBoards(board, otherPlayer)
        for minBoard in min_list_boards:
            totalHeuristic = totalHeuristic + evalHeuristic(minboard)
        totalHeuristic = totalHeuristic/len(min_list_boards)
        min_layer_expect[board] = totalHeuristic


    # maximize the new

    return 
'''
def findOptimalMoveNonDet(newGame, board, depth, player):
    moves = ["L", "R", "U", "D"]
    maxUtility = []
    nextMove = -1
    realnewGame = copy.deepcopy(newGame)
    for move in moves:
        reallynewGame = copy.deepcopy(newGame)
        child = reallynewGame.makeMove(move)
        childBoard = child.getGame()
        utility = expctiMiniMax(child, depth, player)
        maxUtility.append(utility)
        # not sure if this part works
        if utility >= max(maxUtility):
            nextMove = move
    return nextMove

def next_move(game: Game_IJK) -> None:

    """board: list of list of strings -> current state of the game
       current_player: int -> player who will make the next move either ('+') or -'-')
       deterministic: bool -> either True or False, indicating whether the game is deterministic or not
    """

    board = game.getGame()
    player = game.getCurrentPlayer()
    deterministic = game.getDeterministic()
    newGame = copy.deepcopy(game)
    print(player)

    #move, score = minimax(game, board, 1000, player)
    if deterministic == True:
        move = findOptimalMoveDet(newGame, board, 4, player)
    #elif deterministic == False:
    yield move

    #yield random.choice(["U", "D", "L", "R", "S"])
