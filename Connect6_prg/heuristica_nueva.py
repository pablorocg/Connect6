import numpy as np
import random

def positions_within_distance(x, y, distance=5):
    """Funcion que devuelve las posiciones dentro de un radio de distancia de la posicion (x,y)"""
    positions = []

    # Vertical positions above and below
    for d in range(1, distance+1):
        positions.append((x, y-d))  # Above
        positions.append((x, y+d))  # Below

    # Horizontal positions left and right
    for d in range(1, distance+1):
        positions.append((x-d, y))  # Left
        positions.append((x+d, y))  # Right

    # Diagonal positions
    for d in range(1, distance+1):
        positions.append((x-d, y-d))  # Top-left
        positions.append((x+d, y+d))  # Bottom-right
        positions.append((x-d, y+d))  # Bottom-left
        positions.append((x+d, y-d))  # Top-right

    # Remove positions that are out of bounds
    positions = [(x, y) for x, y in positions if 0 <= x < 19 and 0 <= y < 19]
    return positions



def initial_move_score_matrix():
    matrix = np.zeros((19, 19), dtype=int)
    center = 19 // 2

    # Solo calcula la esquina superior izquierda de la matriz
    for i in range(center + 1):
        for j in range(center + 1):
            matrix[i, j] = len(positions_within_distance(i, j, 5))

    # Refleja los valores en la esquina superior derecha
    matrix[:, center+1:] = matrix[:, center-1::-1]

    # Refleja los valores en la parte inferior del tablero
    matrix[center+1:, :] = matrix[center-1::-1, :]

    # Normaliza la matriz
    matrix = matrix*10 / np.max(matrix)
    return matrix


def display(board):

    """Prints the Connect6 board in a visual format."""
    
    symbols = {
        0: '.',
        1: '○',
        2: '●',
        3: 'X'
    }
    
    for row in board:
        print(' '.join(symbols[cell] for cell in row))
    print("\n")


def posiciones_a_evaluar(board, player):
    """
    Funcion que devuelve las posiciones a evaluar para atacar y defender para 2 turnos seguidos
    devuelve una lista con forma [(m1,m2), (m1,m2), ...] donde m1 y m2 son tuplas con las posiciones x e y
    """
    
    oc_pos = np.argwhere(board!=0) # Posiciones ocupadas
    pi_pos = []

    # Generar una lista con todas las posiciones de interes
    for pos in oc_pos:
        px, py = pos
        pi_pos.extend([x for x in positions_within_distance(px, py, 5) if x not in oc_pos])

    # pi_pos = [x for x in pi_pos if x not in oc_pos]
    
    # # Eliminar duplicados
    # pi_pos = list(set(pi_pos))

    # Asignar un 3 a las posiciones de interes y display
    for pos in pi_pos:
        px, py = pos
        board[px, py] = 3

    display(board)
    
    




m = np.zeros((19, 19), dtype=int)

player = 1
# Generar una partida aleatoria
for i in range(3):
    if i==0:
        m[10,10] = player
        player = 3 - player
        continue

    else:
        for j in range(2):
            valid_pos = np.argwhere(m == 0)
            x, y = random.choice(valid_pos)
            m[x,y] = player

        player = 3 - player


posiciones_a_evaluar(m, 1)

display(m)


