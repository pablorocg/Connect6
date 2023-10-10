from defines import *
import time
import numpy as np
import random
import itertools as it

# Visualizacion del tablero
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


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



def get_available_moves_with_score(board, color):
    """
    Returns a list of available moves on the board.

    """
    emtpy_coords = np.argwhere(board == Defines.NOSTONE)

    coords = list(it.combinations(map(tuple, emtpy_coords), 2))

    # Para cada movimiento, calcula el score de ese movimiento y crear un objeto StoneMove con el score y el movimiento
    moves = [create_move(coord) for coord in coords]

    for move in moves:
        #Cpia el tablero
        board_copy = np.array(board)
        make_move(board_copy, move, color)
        # Asigna el score al movimiento
        move.score = offensive_evaluate_state(board_copy)

    return moves
        

def get_window_scoring(window, player, k=1):
    
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

def offensive_evaluate_state(board):
    """
    Optimized evaluation of the board state for both players.

    :param board: 2D numpy array representing the game board. 0 for empty, 1 for player 1, and 2 for player 2.
    :return: Heuristic value of the board state for player 1 and player 2.
    """
    board = board[1:-1, 1:-1]
    directions = [(1, 0), (0, 1), (1, 1), (1, -1)]
    values = {1: 1, 2: 10, 3: 50, 4: 200, 5: 1000}

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
                if sequence_length in values:
                    score_increment = values[sequence_length]
                    if not blocked_left and not blocked_right:
                        score_increment *= 1.5
                    elif blocked_left and blocked_right:
                        score_increment *= 0.5
                    scores[current_cell] += score_increment

    return scores[1] - scores[2]

def defensive_evaluate_state(board):
    """
    A more defensive heuristic for evaluating the board state.
    Positive values favor player 1, negative values favor player 2.

    :param board: 2D numpy array representing the game board. 0 for empty, 1 for player 1, and 2 for player 2.
    :return: Heuristic value of the board state.
    """
    board = board[1:-1, 1:-1]

    directions = [(1, 0), (0, 1), (1, 1), (1, -1)]
    offensive_values = {1: 1, 2: 5, 3: 25, 4: 100, 5: 500}
    defensive_values = {1: 1, 2: 10, 3: 60, 4: 300, 5: 1500}

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
                values = defensive_values if current_cell == 2 else offensive_values
                if sequence_length in values:
                    score_increment = values[sequence_length]
                    if not blocked_left and not blocked_right:
                        score_increment *= 1.5
                    elif blocked_left and blocked_right:
                        score_increment *= 0.5
                    scores[current_cell] += score_increment

    return scores[1] - scores[2]





def mide_tiempo(funcion):
    def funcion_medida(*args, **kwargs):
        inicio = time.time()
        c = funcion(*args, **kwargs)
        print(time.time() - inicio)
        return c
    return funcion_medida



def show_m_board(m_board):
    """
    Visualizacion del estado del juego
    """

    col = 'ABCDEFGHIJKLMNOPQRS'
    fil = 'SRQPONMLKJIHGFEDCBA'

    # Array de anotaciones de 19x19
    annot = []
    for l1 in fil:
        fila = []
        for l2 in col:
            fila.append(l2 + l1)
        annot.append(fila)

    #Obtener el tablero sin los bordes
    m_board_rep = np.array(m_board)
    m_board_rep = m_board_rep[1:20,1:20]
    
    # Reemplazar los 2 por -1
    m_board_rep = np.where(m_board_rep == 2, -1, m_board_rep)
    
    fontdict = {'fontsize': 10,
                'fontweight' : 50}
    # Crear un colormap personalizado en el que el negro = 1, blanco = -1 y naranja = 0.
    # Define los colores
    colors = [(1, 1, 1), (1, 0.8, 0.4), (0, 0, 0)]

    # Crea el colormap
    n_bins = 3  
    cmap_name = "custom_colormap"
    cm = mcolors.LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bins)

    # Normaliza los valores para que estén en el rango [0, 1]
    norm = mcolors.Normalize(vmin=-1, vmax=1)

    plt.figure(figsize=(6,6))
    # Mostrar movimientos sobre el tablero
    plt.imshow(m_board_rep, cmap=cm, norm=norm) 
    
    # Mostrar anotaciones sobre la imagen
    for i in range(19):
        for j in range(19):
            plt.text(j, i, annot[i][j], ha="center", va="center", color="green", fontdict=fontdict)

    plt.axis('off')
    plt.tight_layout()
    plt.show()