from defines import *
import time
import numpy as np
# import pandas as pd
# import random
import itertools as it




def isValidPos(x,y):
    """
    Check if the given position is a valid position on the game board.

    Args:
    x (int): The x-coordinate of the position to check.
    y (int): The y-coordinate of the position to check.

    Returns:
    bool: True if the position is valid, False otherwise.
    """
    return x>0 and x<Defines.GRID_NUM-1 and y>0 and y<Defines.GRID_NUM-1


def init_board(board):    
    """
    Initializes the game board with NOSTONE values, and sets the border values to BORDER.

    Args:
    board (numpy.ndarray): The game board to be initialized.

    Returns:
    numpy.ndarray: The initialized game board.
    """
    board[:] = Defines.NOSTONE
    board[0, :] = board[-1, :] = board[:, 0] = board[:, -1] = Defines.BORDER
    return board


def create_move(positions):
    """
    Creates a move from the given positions.

    Args:
    positions (list): A list of positions ((x1, y1), (x2, y2)).

    """
    move = StoneMove()
    move.positions[0].x = positions[0][0]
    move.positions[0].y = positions[0][1]
    move.positions[1].x = positions[1][0]
    move.positions[1].y = positions[1][1]

    
    return move


def make_move(board, move, color):
    """
    Hacer un movimiento sobre board
    """
    board[move.positions[0].x, move.positions[0].y] = color
    board[move.positions[1].x, move.positions[1].y] = color

    return board


def unmake_move(board, move):
    """
    Deshacer un movimiento sobre board
    """
    board[move.positions[0].x, move.positions[0].y] = Defines.NOSTONE
    board[move.positions[1].x, move.positions[1].y] = Defines.NOSTONE


def is_win_by_premove(board, preMove):
    """
    Determines if a player has won the game by making a pre-move.

    Args:
    - board (numpy.ndarray): The game board
    - preMove (Move): pre-move made by the player

    Returns:
    - True if the pre-move results in a win for the player, False otherwise
    """
    directions = [(1, 0), (0, 1), (1, 1), (1, -1)]

    for direction in directions:
        for position in preMove.positions:
            count = 0
            n = x = position.x
            m = y = position.y
            movStone = board[n, m]

            if (movStone == Defines.BORDER or movStone == Defines.NOSTONE):
                continue

            while board[x, y] == movStone and isValidPos(x + direction[0], y + direction[1]):
                x += direction[0]
                y += direction[1]
                count += 1

            x = n - direction[0]
            y = m - direction[1]

            while board[x, y] == movStone and isValidPos(x - direction[0], y - direction[1]):
                x -= direction[0]
                y -= direction[1]
                count += 1

            if count >= 6:
                return True

    return False


def check_winner(board):
    """
    Check if there is a winner on the Connect6 board.

    Args:
      board (numpy.ndarray): The Connect6 board.

    Returns:
      int: The winner's value (1 or 2) if there is a winner, otherwise 0.
    """
    board = board[1:-1, 1:-1]
    rows, cols = board.shape
    directions = [(1, 0), (0, 1), (1, 1), (1, -1)]
    
    for x in range(rows):
        for y in range(cols):
            if board[x, y] != 0:
                for dx, dy in directions:
                    if 0 <= x + 5*dx < rows and 0 <= y + 5*dy < cols:
                        if all(board[x + i*dx, y + i*dy] == board[x, y] for i in range(6)):
                            return board[x, y]
    return 0



def get_msg(max_len):
    """
    Input a message from the console.

    Args:
        max_len (int): The maximum length of the message to be input.

    Returns:
        str: The input message, truncated to the specified maximum length.
    """
    buf = input().strip()
    return buf[:max_len]


def log_to_file(msg):
    """
    Appends a message to a log file.

    Args:
        msg (str): The message to be logged.

    Returns:
        int: 0 if the message was successfully logged, -1 otherwise.
    """

    g_log_file_name = Defines.LOG_FILE
    try:
        with open(g_log_file_name, "a") as file:
            tm = time.time()
            ptr = time.ctime(tm)
            ptr = ptr[:-1]
            file.write(f"[{ptr}] - {msg}\n")
        return 0
    except Exception as e:
        print(f"Error: Can't open log file - {g_log_file_name}")
        return -1


def move2msg(move):
    if move.positions[0].x == move.positions[1].x and move.positions[0].y == move.positions[1].y:
        msg = f"{chr(ord('S') - move.positions[0].x + 1)}{chr(move.positions[0].y + ord('A') - 1)}"
        return msg
    else:
        msg = f"{chr(move.positions[0].y + ord('A') - 1)}{chr(ord('S') - move.positions[0].x + 1)}" \
              f"{chr(move.positions[1].y + ord('A') - 1)}{chr(ord('S') - move.positions[1].x + 1)}"
        return msg

def msg2move(msg):
    move = StoneMove()
    if len(msg) == 2:
        move.positions[0].x = move.positions[1].x = ord('S') - ord(msg[1]) + 1
        move.positions[0].y = move.positions[1].y = ord(msg[0]) - ord('A') + 1
        move.score = 0
        return move
    else:
        move.positions[0].x = ord('S') - ord(msg[1]) + 1
        move.positions[0].y = ord(msg[0]) - ord('A') + 1
        move.positions[1].x = ord('S') - ord(msg[3]) + 1
        move.positions[1].y = ord(msg[2]) - ord('A') + 1
        move.score = 0
        return move

def print_board(board):
    """
    Prints the Connect6 board with the current state of the game.

    Args:
    - board (numpy.ndarray): The Connect6 board
    
    Returns:
    - None
    """
    header = "   " + "".join([chr(i + ord('A') - 1) + " " for i in range(1, Defines.GRID_NUM - 1)])
    print(header)
    
    for i in range(1, Defines.GRID_NUM - 1):
        row_char = chr(ord('A') - 1 + i)
        print(row_char, end=" ")
        
        # Using numpy array slicing for efficient access
        row_data = board[Defines.GRID_NUM - 1 - i, 1:Defines.GRID_NUM - 1]
        row_symbols = np.where(row_data == Defines.NOSTONE, "-", np.where(row_data == 1, "O", "*"))
        print(" ".join(row_symbols), end=" ")
        
        print(row_char)
    print(header)

def display(board):
    """
    Prints the Connect6 board in a more visual format.

    Args:
        board (list): The Connect6 board represented as a list of lists.

    Returns:
        None
    """

    symbols = {
        0: '.',
        1: '○',
        2: '●',
        3: 'X'
    }

    for row in board:
        print(' '.join(symbols[cell] for cell in row))
    print("\n")




def positions_within_distance(x, y, distance=5):
    """
    Function that returns the positions within a radius of distance from the position (x,y)

    Args:
        x (int): The x-coordinate of the center position.
        y (int): The y-coordinate of the center position.
        distance (int, optional): The radius within which to find positions. Defaults to 5.

    Returns:
        list: A list of positions within the specified distance from the center position.
    """
    positions = set()

    for dx in range(-distance, distance + 1):
        for dy in range(-distance, distance + 1):
            nx, ny = x + dx, y + dy
            if 0 < nx < 20 and 0 < ny < 20:
                positions.add((nx, ny))

    positions.discard((x, y))  # Optionally remove the center point if not needed
    return list(positions)


def get_available_moves_with_score(board, color, valores):
    """
    Get a list of available moves with their corresponding scores.

    Args:
        board (numpy.ndarray): The game board.
        color (int): The color of the player making the moves.
        valores (dict): A dictionary containing the values for different game elements.

    Returns:
        list: A list of moves with their scores.
    """
    board_1 = np.copy(board)
    piece_coords = np.argwhere((board == 1) | (board == 2))# Obtiene las coordenadas de las piezas sobre el tablero
    
    # Filtra las coordenadas de las piezas para obtener solo las que estan a una distancia de 2 de una pieza
    piece_coords = [coord for coord in piece_coords if 
                    board_1[coord[0] - 1, coord[1]] == 0 or 
                    board_1[coord[0] + 1, coord[1]] == 0 or 
                    board_1[coord[0], coord[1] - 1] == 0 or 
                    board_1[coord[0], coord[1] + 1] == 0 or 
                    board_1[coord[0] - 1, coord[1] - 1] == 0 or
                    board_1[coord[0] - 1, coord[1] + 1] == 0 or
                    board_1[coord[0] + 1, coord[1] - 1] == 0 or
                    board_1[coord[0] + 1, coord[1] + 1] == 0]
    
    # Obtiene las coordenadas vacias a una distancia de 2 de una pieza
    empty_coords = list(set(pos for x, y in piece_coords for pos in positions_within_distance(x, y, 2) if board[pos[0], pos[1]] == 0))
    moves = []
    
    # Crea las combinaciones de movimientos posibles
    comb_moves = it.combinations(empty_coords, 2)
    for move_coords in comb_moves:
        move = create_move(move_coords)
        board_1 = board.copy()
        make_move(board_1, move, color)
        # Evaluacion del estado del tablero
        move.score = defensive_evaluate_state(board_1, color, valores)
        
        moves.append(move)
    return moves




def get_center_multiplier(x, y, mult, board_size=19):
    """
    Calculates the center multiplier for a given position on the board (Manhattan).

    Args:
        x (int): The x-coordinate of the position.
        y (int): The y-coordinate of the position.
        mult (float): The initial multiplier.
        board_size (int, optional): The size of the board. Defaults to 19.

    Returns:
        float: The center multiplier.
    """

    center = board_size // 2
    distance_to_center = abs(x - center) + abs(y - center)
    max_distance = 2 * center
    return mult + (max_distance - distance_to_center) / max_distance


def defensive_evaluate_state(board, color):
    """
    Heuristica: Evaluate the state of the board from a defensive perspective.

    Args:
        board (numpy.ndarray): The game board represented as a 2D numpy array.
        color (int): The color of the player.
        valores (dict): A dictionary containing the values for different sequence lengths.

    Returns:
        int: The score difference between the defensive player and the offensive player.
    """

    board = board[1:-1, 1:-1]

    directions = [(1, 0), (0, 1), (1, 1), (1, -1)]
    offensive_values = {1: 1, 2: 5, 3: 25, 4: 100, 5: 500, 6: 10000000}
    defensive_values = {1: 1, 2: 10, 3: 60, 4: 600, 5: 2500, 6: 10000000}

    scores = {1: 0, 2: 0}

    for x in range(19):
        for y in range(19):
            current_cell = board[x, y]
            if current_cell == 0:  # Skip empty cells
                continue

            for dx, dy in directions:
                # Check if the sequence in this direction has already been counted
                if 0 <= x - dx < 19 and 0 <= y - dy < 19 and board[x - dx, y - dy] == current_cell:
                    continue
                
                sequence_length = 1
                blocked_left = False
                blocked_right = False
                if np.count_nonzero(board != 0) < 5:
                    multiplier_center = get_center_multiplier(x, y, 2)
                else: 
                    multiplier_center = 1

                # Check left
                if 0 <= x - dx < 19 and 0 <= y - dy < 19 and board[x - dx, y - dy] != 0:
                    blocked_left = True

                # Count sequence length
                nx, ny = x + dx, y + dy
                while 0 <= nx < 19 and 0 <= ny < 19 and board[nx, ny] == current_cell:
                    sequence_length += 1
                    nx += dx
                    ny += dy

                # Check right
                if 0 <= nx < 19 and 0 <= ny < 19 and board[nx, ny] != 0:
                    blocked_right = True

                # Update score based on sequence length
                if current_cell != color:
                    values = defensive_values  
                    if sequence_length in values:
                        score_increment = values[sequence_length]
                        if (not blocked_left and not blocked_right):
                            score_increment *= 1.5
                        elif (blocked_left and not blocked_right) or (not blocked_left and blocked_right):
                            score_increment *= 1.25    
                        elif blocked_left and blocked_right:
                            score_increment *= 0.5
                        scores[current_cell] += score_increment * multiplier_center
                if current_cell == color:
                    values = offensive_values  
                    if sequence_length in values:
                        score_increment = values[sequence_length]
                        scores[current_cell] += score_increment * multiplier_center

    return scores[1] - scores[2]






def check_first_move(board):
    """
    Check if it is the first move on the board.

    Args:
        board (list): The game board.

    Returns:
        bool: True if it is the first move, False otherwise.
    """
    for i in range(1,len(board)-1):
        for j in range(1, len(board[i])-1):
            if(board[i][j] != Defines.NOSTONE):
                return False
    return True





def mide_tiempo(funcion):
    def funcion_medida(*args, **kwargs):
        inicio = time.time()
        c = funcion(*args, **kwargs)
        print(time.time() - inicio)
        return c
    return funcion_medida



