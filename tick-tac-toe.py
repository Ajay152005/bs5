import random

def print_board(board):
    for row in board:
        print(" | ".join(row))
        print("-" * 5)

def check_winner(board, player):
    #check rows, columns, and diagonals for a win
    for i in range(3):
        if all(cell == player for cell in board[i]):
            return True
        if all(board[j][i] == player for j in range(3)):
            return True
    if all(board[i][i] == player for i in range(3) or all(board[i][2-i] == player for i in range(3))):
        return True
    return False

def available_moves(board):
    return [(i, j) for i in range(3) for j in range(3) if board[i][j]== " "]

def minimax(board, depth, maximizing_player):
    if check_winner(board, "X"):
        return -10 + depth, None
    elif check_winner(board, "O"):
        return 10 -depth, None
    elif len(available_moves(board)) == 0:
        return 0, None

    if maximizing_player:
        max_eval = float("-inf")
        best_move = None
        for move in available_moves(board):
            board[move[0]][move[1]] = "O"
            eval, _ = minimax(board, depth + 1, False)
            board[move[0]][move[1]] = " "
            if eval > max_eval:
                max_eval = eval
                best_move = move
        return max_eval, best_move
    else:
        min_eval = float("inf")
        best_move = None
        for move in available_moves(board):
            board[move[0]][move[1]] = "X"
            eval, _ = minimax(board, depth+1, True)
            board[move[0]][move[1]] = " "
            if eval < min_eval:
                min_eval = eval
                best_move = move
        return min_eval, best_move