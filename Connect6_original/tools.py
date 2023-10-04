from defines import *
import time
import numpy as np


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
    None
    """
    board[:] = Defines.NOSTONE
    board[0, :] = board[-1, :] = board[:, 0] = board[:, -1] = Defines.BORDER


def make_move(board, move, color):
    """
    Updates the board with the given move and color.

    Args:
        board (numpy.ndarray): The Connect6 board.
        move (Move): The move to make.
        color (int): The color of the player making the move.

    Returns:
        None
    """
    board[move.positions[0].x, move.positions[0].y] = color
    board[move.positions[1].x, move.positions[1].y] = color

def unmake_move(board, move):
    """
    Undo a move on the board by setting the positions of the move to Defines.NOSTONE.

    Args:
    - board (numpy.ndarray): The Connect6 board
    - move (Move): a Move object representing the move to be undone

    Returns:
    - None
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

def get_msg(max_len):
    """
    Reads a string from standard input and returns a substring of it with a maximum length of max_len.

    Args:
        max_len (int): The maximum length of the substring to return.

    Returns:
        str: A substring of the input string with a maximum length of max_len.
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
    """
    Converts a move object to a message string.

    Args:
        move (Move): A move object containing two Position objects.

    Returns:
        str: A message string representing the move.
    """
    if move.positions[0].x == move.positions[1].x and move.positions[0].y == move.positions[1].y:
        msg = f"{chr(ord('S') - move.positions[0].x + 1)}{chr(move.positions[0].y + ord('A') - 1)}"
        return msg
    else:
        msg = f"{chr(move.positions[0].y + ord('A') - 1)}{chr(ord('S') - move.positions[0].x + 1)}" \
              f"{chr(move.positions[1].y + ord('A') - 1)}{chr(ord('S') - move.positions[1].x + 1)}"
        return msg

def msg2move(msg):
    """
    Converts a message string to a StoneMove object.

    Args:
        msg (str): A string representing the move.

    Returns:
        StoneMove: A StoneMove object representing the move.
    """
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


# ------------------------------------------------------------

@mide_tiempo
def get_window_scoring(window, player):
    k=0
    """
    Calcula el score de una ventana de 6 fichas, para un jugador dado. 
    """
    player_count = np.count_nonzero(window == player)
    empty_count = np.count_nonzero(window == 0)
    
    if player_count + empty_count == 6:
        return player_count ** (player_count + k)
    return 0
    
@mide_tiempo
def get_score(m_board, player):
    """
    Función que evalúa el estado de una partida basándose en el número de amenazas de cada
    jugador y la diferencia entre ellas para determinar quién tiene ventaja en la partida.
    """
    board = m_board[1:-1, 1:-1]
    
    JUGADOR = player
    OPONENTE = 1 if JUGADOR == 2 else 2
    
    score = 0
    
    # Iterar sobre todas las posibles ventanas de 6 fichas en el tablero
    for i in range(board.shape[0] - 5):
        for j in range(board.shape[1] - 5):
            
            # Amenazas horizontales
            horizontal_window = board[i,j:j+6]
            score += get_window_scoring(horizontal_window, JUGADOR)
            score -= get_window_scoring(horizontal_window, OPONENTE)
            
            # Amenazas verticales
            vertical_window = board[j:j+6,i]
            score += get_window_scoring(vertical_window, JUGADOR)
            score -= get_window_scoring(vertical_window, OPONENTE)   
            
            # Amenazas diagonales
            diagonal_window = [board[i+k,j+k] for k in range(6)]
            score += get_window_scoring(diagonal_window, JUGADOR)
            score -= get_window_scoring(diagonal_window, OPONENTE)
            
            anti_diagonal_window = [board[i+k,j+5-k] for k in range(6)]
            score += get_window_scoring(anti_diagonal_window, JUGADOR)
            score -= get_window_scoring(anti_diagonal_window, OPONENTE)

    return score


        
def mide_tiempo(funcion):
    def funcion_medida(*args, **kwargs):
        inicio = time.time()
        c = funcion(*args, **kwargs)
        print(time.time() - inicio)
        return c
    return funcion_medida

