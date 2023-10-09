from defines import *
import time
import numpy as np
import random
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

def print_board(board, preMove=None):
    """
    Prints the Connect6 board with the current state of the game.

    Args:
    - board (numpy.ndarray): The Connect6 board
    - preMove (Move): Previous move made on the board (optional)

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


def print_score(move_list, n):
    board_scores = np.zeros((Defines.GRID_NUM, Defines.GRID_NUM), dtype=int)
    
    for move in move_list:
        board_scores[move.x, move.y] = move.score

    header = "  " + "".join([f"{i:4}" for i in range(1, Defines.GRID_NUM - 1)])
    print(header)
    
    for i in range(1, Defines.GRID_NUM - 1):
        print(f"{i:2}", end="")
        
        # Using numpy array slicing for efficient access
        row_data = board_scores[i, 1:Defines.GRID_NUM - 1]
        row_strings = np.where(row_data == 0, "   -", row_data.astype(str))
        print("".join(row_strings))


def get_available_moves(board):
    """
    Returns a list of available moves on the board.

    Parameters:
    board (numpy.ndarray): A numpy array representing the Connect6 board.

    Returns:
    list: A list of tuples representing the available moves on the board.
    """
    
    # Encuentra las coordenadas de las casillas vacías
    empty_coords = np.argwhere(board == Defines.NOSTONE)

    # Devuelve todas las combinaciones de 2 movimientos posibles
    coords = list(it.combinations(map(tuple, empty_coords), 2))
    
    return [create_move(coord) for coord in coords]



def get_random_move(board):
    """
    Devuelve un movimiento aleatorio de la lista de movimientos posibles
    """
    moves = get_available_moves(board)
    return moves[random.randint(0, len(moves) - 1)]


def get_window_scoring(window, player, k=0):
    
    """
    Calcula el score de una ventana de 6 fichas, para un jugador dado. 
    """
    player_count = np.count_nonzero(window == player)
    empty_count = np.count_nonzero(window == 0)
    
    if player_count + empty_count == 6:
        return player_count ** (player_count + k)
    return 0
    

def get_score(m_board):
    """
    Función que evalúa el estado de una partida basándose en el número de amenazas de cada
    jugador y la diferencia entre ellas para determinar quién tiene ventaja en la partida.

    negro  ->  + inf (maximiza)
    blanco -> - inf (minimiza)
    """
    board = m_board[1:-1, 1:-1]
    
    score = 0
    
    # Iterar sobre todas las posibles ventanas de 6 fichas en el tablero
    for i in range(board.shape[0] - 5):
        for j in range(board.shape[1] - 5):
            
            # Amenazas horizontales
            horizontal_window = board[i,j:j+6]
            score += get_window_scoring(horizontal_window, Defines.BLACK)
            score -= get_window_scoring(horizontal_window, Defines.WHITE)
            
            # Amenazas verticales
            vertical_window = board[j:j+6,i]
            score += get_window_scoring(vertical_window, Defines.BLACK)
            score -= get_window_scoring(vertical_window, Defines.WHITE)   
            
            # Amenazas diagonales
            diagonal_window = [board[i+k,j+k] for k in range(6)]
            score += get_window_scoring(diagonal_window, Defines.BLACK)
            score -= get_window_scoring(diagonal_window, Defines.WHITE)
            
            anti_diagonal_window = [board[i+k,j+5-k] for k in range(6)]
            score += get_window_scoring(anti_diagonal_window, Defines.BLACK)
            score -= get_window_scoring(anti_diagonal_window, Defines.WHITE)

    return score






def mide_tiempo(funcion):
    def funcion_medida(*args, **kwargs):
        inicio = time.time()
        c = funcion(*args, **kwargs)
        print(time.time() - inicio)
        return c
    return funcion_medida



test = False

if test:
    test_board = init_board(np.zeros((Defines.GRID_NUM, Defines.GRID_NUM), dtype=int))
    moves = get_available_moves(test_board)
    

    move = create_move(moves[1])
    comando = move2msg(move)
    print(comando)
    make_move(test_board, move, Defines.BLACK)
    print_board(test_board)