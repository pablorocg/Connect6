import numpy as np

def get_window_scoring(combination, player):
    """
    Calcula el score de una ventana de 6 fichas, para un jugador dado. 
    La puntuacion es 2^(n + k) donde n es el numero de fichas del jugador en la ventana y k es el coeficiente de la exponencial.
    """
    k = 1
    combination = np.array(combination)
    
    score = 0

    for i in range(1, 7):
        if np.count_nonzero(combination == player) == i and np.count_nonzero(combination == 0) == 6-i:
            return 2**(i + k)
    
    return score



def get_m_board_score(m_board, player):
    m_board = np.array(m_board)
    m_board = m_board[1:-1, 1:-1]

    if player == 2:
        opponent = 1
    else:
        opponent = 2
    
    score_jug, score_op = 0, 0
    
    
    # Amenazas horizontales
    for i in range(m_board.shape[0]):
        for j in range(m_board.shape[1] - 6 + 1):
            window = m_board[i][j:j+6]
            score_jug += get_window_scoring(window, player)
            score_op += get_window_scoring(window, opponent)
                
                
    
    # Amenazas verticales
    for col in range(m_board.shape[1]):
        for row in range(m_board.shape[0] - 6 + 1):
            window = m_board[row:row+6, col]  
            score_jug += get_window_scoring(window, player)
            score_op += get_window_scoring(window, opponent)


    # Amenazas diagonales
    for k in range(-13, 14):
  
        for i in range(len(np.diag(m_board, k=k)) - 6 + 1):
            v1 = np.diag(m_board, k=k)[i:i+6]
            v2 = np.diag(np.flip(m_board, axis=0), k=k)[i:i+6]

            score_jug += get_window_scoring(v1, player)
            score_op += get_window_scoring(v1, opponent)
            score_jug += get_window_scoring(v2, player)
            score_op += get_window_scoring(v2, opponent)


    if score_jug == score_op:
        return 0
    
    elif score_jug > score_op:
        diferencia = score_jug - score_op
        return 1 - 1/(1 + diferencia)
    
    else:
        diferencia = score_op - score_jug
        return -1 + 1/(1 + diferencia)       