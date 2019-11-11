#!/usr/local/bin/python3

"""
This is where you should write your AI code!

Authors: Kelly Wheeler (kellwhee), Neha Supe (nehasupe), Bobby Rathore (brathore)

Based on skeleton code by Abhilash Kuhikar, October 2019
"""

from logic_IJK import Game_IJK
import random
import math
import statistics
import copy

# Suggests next move to be played by the current player given the current game
#
# inputs:
#     game : Current state of the game
#
# This function should analyze the current state of the game and determine the
# best move for the current player. It should then call "yield" on that move.

# Finds empty tiles on the current board configuration and returns a list of their co-ordinates
def freespaces(board):
    child = [(i, j) for j in range(6) for i in range(6) if board[i][j] == " "]
    return child


# Our heuristic function consists of 3 functions- freeTile, merges, highestvaltile and gradient
# Referred discussion on stackoverflow:
# https://stackoverflow.com/questions/22342854/what-is-the-optimal-algorithm-for-the-game-2048/22498940#22498940

# freeTile counts the number of empty tiles on the board. When the board is getting filled up, configurations with more empty tiles and merges will have preference over other configurations
def freeTile(board):
    moreMoves = 0
    for i in range(len(board)):
        for j in range(len(board[0])):
            if board[i][j] == " ":
                moreMoves += 1
    if moreMoves < 0.6 * len(board) * len(board[0]):
        return moreMoves * 50
    else:
        return 0


# counts the number of possible merges on the board, and assigns a bonus value, higher character merges are assigned higher values
def merges(board):
    bonus = 0
    # letterValue = {'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 5, 'f': 6, 'g': 7, 'h': 8, 'i': 9, 'j': 10, 'k': 11}
    letterValue = {
        "a": pow(2, 1),
        "b": pow(2, 2),
        "c": pow(2, 3),
        "d": pow(2, 4),
        "e": pow(2, 5),
        "f": pow(2, 6),
        "g": pow(2, 7),
        "h": pow(2, 8),
        "i": pow(2, 9),
        "j": pow(2, 10),
        "k": pow(2, 11),
    }
    for i in range(1, len(board)):
        for j in range(1, len(board[0])):
            if board[i][j] != " ":
                if board[i - 1][j].lower() == board[i][j].lower():
                    bonus = 10 * letterValue[board[i][j].lower()]
                if board[i][j - 1].lower() == board[i][j].lower():
                    bonus = 10 * letterValue[board[i][j].lower()]
    return bonus


# This function assigns a penalty value if the opponent has the highest value tile on the board
def highestvaltile(board, currentPlayer, otherPlayer):
    letterValue = {
        "a": 1,
        "b": 2,
        "c": 3,
        "d": 4,
        "e": 5,
        "f": 6,
        "g": 7,
        "h": 8,
        "i": 9,
        "j": 10,
        "k": 11,
    }
    upper = []
    lower = []
    penalty = 0
    for i in range(6):
        for j in range(6):
            if board[i][j] != " ":
                if board[i][j].isupper():
                    upper.append(board[i][j])
                if board[i][j].islower():
                    lower.append(board[i][j])
    if currentPlayer == "+" and len(lower) > 0:
        if max(upper) < max(lower):
            penalty = 10 * letterValue[max(lower)]
    elif currentPlayer == "-" and len(upper) > 0:
        if max(lower) < max(upper):
            penalty = 10 * letterValue[(max(upper)).lower()]
    return penalty


# converts the passed board into numerical board for testing the gradient property
def numBoard(board):
    letterValue = {
        "a": 1,
        "b": 2,
        "c": 3,
        "d": 4,
        "e": 5,
        "f": 6,
        "g": 7,
        "h": 8,
        "i": 9,
        "j": 10,
        "k": 11,
    }
    newBoard = board
    for i in range(len(board)):
        for j in range(len(board[0])):
            index = board[i][j].lower()
            if index != " ":
                newBoard[i][j] = letterValue[index]
            else:
                newBoard[i][j] = 0
    return newBoard


# Checks for the value of gradiance in the passed board
def gradient(board):
    board = numBoard(board)
    gradient = [
        [
            [0, -1, -2, -3, -4, -5],
            [1, 0, -1, -2, -3, -4],
            [2, 1, 0, -1, -2, -3],
            [3, 2, 1, 0, -1, -2],
            [4, 3, 2, 1, 0, -1],
            [5, 4, 3, 2, 1, 0],
        ],
        [
            [5, 4, 3, 2, 1, 0],
            [4, 3, 2, 1, 0, -1],
            [3, 2, 1, 0, -1, -2],
            [2, 1, 0, -1, -2, -3],
            [1, 0, -1, -2, -3, -4],
            [0, -1, -2, -3, -4, -5],
        ],
        [
            [0, 1, 2, 3, 4, 5],
            [-1, 0, 1, 2, 3, 4],
            [-2, -1, 0, 1, 2, 3],
            [-3, -2, -1, 0, 1, 2],
            [-4, -3, -2, -1, 0, 1],
            [-5, -4, -3, -2, -1, 0],
        ],
        [
            [-5, -4, -3, -2, -1, 0],
            [-4, -3, -2, -1, 0, 1],
            [-3, -2, -1, 0, 1, 2],
            [-2, -1, 0, 1, 2, 3],
            [-1, 0, 1, 2, 3, 4],
            [0, 1, 2, 3, 4, 5],
        ],
    ]
    value = [0] * len(gradient)
    for i in range(len(gradient)):
        for j in range(len(board)):
            for k in range(len(board[0])):
                value[i] = value[i] + gradient[i][j][k] * board[j][k]
    return max(value)


# Heuristic function for Deterministic IJK
def evalHeuristic(board):
    heuristic = (
        freeTile(board) + merges(board) + gradient(board)
    )  # - highestvaltile(board, currentPlayer)
    return heuristic


# Heuristic Evaluation function for Non Deterministic IJK
def evalHeuristicNonDet(board, currentPlayer, otherPlayer):
    heuristic = (
        freeTile(board) + merges(board) + gradient(board)
    )  # - highestvaltile(board, currentPlayer, otherPlayer)
    return heuristic


# Implementation of Minimax algorithm with alpha-beta pruning
def miniMax(childBoard, depth, alpha, beta, player):
    if depth == 0:
        return evalHeuristic(childBoard.getGame())

    if player:
        maxUtility = -math.inf
        for move in ["L", "R", "U", "D"]:
            copyboard = copy.deepcopy(childBoard)
            newChild = copyboard.makeMove(move)
            newboard = newChild.getGame()
            maxUtility = max(
                maxUtility, miniMax(newChild, depth - 1, alpha, beta, False)
            )
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
            minUtility = min(
                minUtility, miniMax(newChild, depth - 1, alpha, beta, True)
            )
            beta = min(beta, minUtility)
            if beta <= alpha:
                break
        return minUtility


# Funtion returns the best move for Deterministic IJK
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


# Performs the passed move on the deepcopy of the passed board
def makeMyMove(boardObject, move):
    if move == "L":
        boardObject._Game_IJK__left(boardObject.getGame())
    if move == "R":
        boardObject._Game_IJK__right(boardObject.getGame())
    if move == "U":
        boardObject._Game_IJK__up(boardObject.getGame())
    if move == "D":
        boardObject._Game_IJK__down(boardObject.getGame())


# Implementation of expectiminimax algorithm
def expectiminimimax(
    game, depth, node, currentPlayer, letter, otherPlayer, otherLetter
):
    if depth == 0:

        return evalHeuristicNonDet(game.getGame(), currentPlayer, otherPlayer)

    elif node == "player":
        minUtility = []
        copy_gameL = copy.deepcopy(game)
        makeMyMove(copy_gameL, "L")
        minUtility.append(
            expectiminimimax(
                copy_gameL,
                depth - 1,
                "chance",
                currentPlayer,
                letter,
                otherPlayer,
                otherLetter,
            )
        )
        copy_gameR = copy.deepcopy(game)
        makeMyMove(copy_gameR, "R")
        minUtility.append(
            expectiminimimax(
                copy_gameR,
                depth - 1,
                "chance",
                currentPlayer,
                letter,
                otherPlayer,
                otherLetter,
            )
        )
        copy_gameU = copy.deepcopy(game)
        makeMyMove(copy_gameL, "U")
        minUtility.append(
            expectiminimimax(
                copy_gameU,
                depth - 1,
                "chance",
                currentPlayer,
                letter,
                otherPlayer,
                otherLetter,
            )
        )
        copy_gameD = copy.deepcopy(game)
        makeMyMove(copy_gameL, "D")
        minUtility.append(
            expectiminimimax(
                copy_gameD,
                depth - 1,
                "chance",
                currentPlayer,
                letter,
                otherPlayer,
                otherLetter,
            )
        )
        return min(minUtility)

    elif node == "chance":
        expectVal = 0
        freespace = freespaces(game.getGame())
        for (i, j) in freespace:
            successor = copy.deepcopy(game)
            successor._Game_IJK__game[i][j] = otherLetter
            expectVal = expectVal + expectiminimimax(
                successor,
                depth - 1,
                "player",
                otherPlayer,
                otherLetter,
                currentPlayer,
                letter,
            )
        return expectVal / len(freespace)


# Returns the best move for Non-Deterministic IJK
def findOptimalMoveNonDet(game, board, currentPlayer):
    if currentPlayer == "+":
        letter = "A"
        otherPlayer = "-"
        otherLetter = "a"
    elif currentPlayer == "-":
        letter = "a"
        otherPlayer = "+"
        otherLetter = "A"
    Moves = ["L", "R", "U", "D"]
    expectedUtility = []
    nextMove = -1

    copy_gameL = copy.deepcopy(game)
    makeMyMove(copy_gameL, "L")
    expectedUtility.append(
        expectiminimimax(
            copy_gameL, 3, "chance", currentPlayer, letter, otherPlayer, otherLetter
        )
    )

    copy_gameR = copy.deepcopy(game)
    makeMyMove(copy_gameR, "R")
    expectedUtility.append(
        expectiminimimax(
            copy_gameR, 3, "chance", currentPlayer, letter, otherPlayer, otherLetter
        )
    )

    copy_gameU = copy.deepcopy(game)
    makeMyMove(copy_gameL, "U")
    expectedUtility.append(
        expectiminimimax(
            copy_gameU, 3, "chance", currentPlayer, letter, otherPlayer, otherLetter
        )
    )

    copy_gameD = copy.deepcopy(game)
    makeMyMove(copy_gameL, "D")
    expectedUtility.append(
        expectiminimimax(
            copy_gameD, 3, "chance", currentPlayer, letter, otherPlayer, otherLetter
        )
    )
    return Moves[expectedUtility.index(max(expectedUtility))]


def next_move(game: Game_IJK) -> None:

    """board: list of list of strings -> current state of the game
       current_player: int -> player who will make the next move either ('+') or -'-')
       deterministic: bool -> either True or False, indicating whether the game is deterministic or not
    """

    board = game.getGame()
    player = game.getCurrentPlayer()
    deterministic = game.getDeterministic()

    if deterministic == True:
        move = findOptimalMoveDet(game, board, 4, player)
    elif deterministic == False:
        move = findOptimalMoveNonDet(game, board, player)
    yield move
