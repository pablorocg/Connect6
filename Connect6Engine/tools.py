from defines import *
import time

# Point (x, y) if in the valid position of the board.
def isValidPos(x,y):
    """Funcion que verifica si una posicion es valida en el tablero"""
    return x>0 and x<Defines.GRID_NUM-1 and y>0 and y<Defines.GRID_NUM-1
    
def init_board(board):
    """Funcion que inicializa el tablero"""
    for i in range(21):
        board[i][0] = board[0][i] = board[i][Defines.GRID_NUM - 1] = board[Defines.GRID_NUM - 1][i] = Defines.BORDER
    for i in range(1, Defines.GRID_NUM - 1):
        for j in range(1, Defines.GRID_NUM - 1):
            board[i][j] = Defines.NOSTONE
            
def make_move(board, move, color):
    """Funcion que realiza un movimiento en el tablero"""
    board[move.positions[0].x][move.positions[0].y] = color
    board[move.positions[1].x][move.positions[1].y] = color

def unmake_move(board, move):
    """Funcion que deshace un movimiento en el tablero"""
    board[move.positions[0].x][move.positions[0].y] = Defines.NOSTONE
    board[move.positions[1].x][move.positions[1].y] = Defines.NOSTONE

def is_win_by_premove(board, preMove):
    """Funcion que verifica si un movimiento es ganador"""
    directions = [(1, 0), (0, 1), (1, 1), (1, -1)]

    for direction in directions:
        for i in range(len(preMove.positions)):
            count = 0
            position = preMove.positions[i]
            n = x = position.x
            m = y = position.y
            movStone = board[n][m]
            
            if (movStone == Defines.BORDER or movStone == Defines.NOSTONE):
                return False;
                
            while board[x][y] == movStone:
                x += direction[0]
                y += direction[1]
                count += 1
            x = n - direction[0]
            y = m - direction[1]
            while board[x][y] == movStone:
                x -= direction[0]
                y -= direction[1]
                count += 1
            if count >= 6:
                return True
    return False

def get_msg(max_len):
    """Funcion que obtiene un mensaje de entrada"""
    buf = input().strip()
    return buf[:max_len]

def log_to_file(msg):
    """Funcion que escribe un mensaje en el archivo de log"""
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
    """Funcion que convierte un movimiento a un mensaje"""
    if move.positions[0].x == move.positions[1].x and move.positions[0].y == move.positions[1].y:
        msg = f"{chr(ord('S') - move.positions[0].x + 1)}{chr(move.positions[0].y + ord('A') - 1)}"
        return msg
    else:
        msg = f"{chr(move.positions[0].y + ord('A') - 1)}{chr(ord('S') - move.positions[0].x + 1)}" \
              f"{chr(move.positions[1].y + ord('A') - 1)}{chr(ord('S') - move.positions[1].x + 1)}"
        return msg

def msg2move(msg):
    """Funcion que convierte un mensaje a un movimiento"""
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
    """Funcion que imprime el tablero"""
    print("  " + "".join([chr(i + ord('A') - 1) for i in range(1, Defines.GRID_NUM - 1)]))
    for i in range(1, Defines.GRID_NUM - 1):
        print(f"{chr(ord('T') - i)}", end="")
        for j in range(1, Defines.GRID_NUM - 1):
            x = i
            y = j
            if preMove and ((x == preMove.positions[0].x and y == preMove.positions[0].y) or
                            (x == preMove.positions[1].x and y == preMove.positions[1].y)):
                print(" X", end="")
                continue
            stone = board[x][y]
            if stone == Defines.NOSTONE:
                print(" -", end="")
            elif stone == 'O':
                print(" O", end="")
            elif stone == '*':
                print(" *", end="")
        print(f"{chr(ord('T') - i)}")
    print("  " + "".join([chr(i + ord('A') - 1) for i in range(1, Defines.GRID_NUM - 1)]))

def print_score(move_list, n):
    """Funcion que imprime la lista de movimientos"""
    board = [[0] * Defines.GRID_NUM for _ in range(Defines.GRID_NUM)]
    for move in move_list:
        board[move.x][move.y] = move.score

    print("  " + "".join([f"{i:4}" for i in range(1, Defines.GRID_NUM - 1)]))
    for i in range(1, Defines.GRID_NUM - 1):
        print(f"{i:2}", end="")
        for j in range(1, Defines.GRID_NUM - 1):
            score = board[i][j]
            if score == 0:
                print("   -", end="")
            else:
                print(f"{score:4}", end="")
        print()
#------------------------------------------------------------------

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np


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





