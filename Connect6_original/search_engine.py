from tools import *

from tools import *

class SearchEngine():
    def __init__(self):
        self.m_board = None
        self.m_chess_type = None
        self.m_alphabeta_depth = None
        self.m_total_nodes = 0

    def before_search(self, board, color, alphabeta_depth):
        self.m_board = board.copy()
        self.m_chess_type = color
        self.m_alphabeta_depth = alphabeta_depth
        self.m_total_nodes = 0

    
           

        
        
    def check_first_move(self):
        for i in range(1,len(self.m_board)-1):
            for j in range(1, len(self.m_board[i])-1):
                if(self.m_board[i][j] != Defines.NOSTONE):
                    return False
        return True
        
    def find_possible_move(self):
        for i in range(1,len(self.m_board)-1):
            for j in range(1, len(self.m_board[i])-1):
                if(self.m_board[i][j] == Defines.NOSTONE):
                    return (i,j)
        return (-1,-1)



def get_window_scoring(window, player):
    """
    Calcula el score de una ventana de 6 fichas, para un jugador dado. 
    La puntuacion es 2^(n + k) donde n es el numero de fichas del jugador en la ventana y k es el coeficiente de la exponencial.
    """
    k = 0# Constante de la exponencial (Cuanto mayor sea, mas se premia el numero de fichas del jugador)
    
    
    for i in range(1, 7):
        if np.count_nonzero(window == player) == i and np.count_nonzero(window == 0) == 6-i:
            return i**(i + k)
    return 0

def get_score(m_board, player):
    """
    Funcion que evalua el estado de una partida basandose en el numero de amenazas de cada
    jugador y la diferencia entre ellas para determinar quien tiene ventaja en la partida.
    :param m_board: Tablero de juego

    """

    
    board = m_board[1:-1, 1:-1]

    JUGADOR = player
    OPONENTE = 1 if JUGADOR == 2 else 2
    
    score = 0
    
    # Amenazas diagonales
    for k in range(-13, 14):

        for i in range(len(np.diag(board, k=k)) - 5):
            v1 = np.diag(board, k=k)[i:i+6]
            v2 = np.diag(np.flip(board, axis=0), k=k)[i:i+6]

            score += get_window_scoring(v1, JUGADOR)
            score -= get_window_scoring(v1, OPONENTE)

            score += get_window_scoring(v2, JUGADOR)
            score -= get_window_scoring(v2, OPONENTE)

    for i in range(19):
        for j in range(16):
            
            # Amenazas horizontales
            horizontal_window = board[i,j:j+6]
            score += get_window_scoring(horizontal_window, JUGADOR)
            score -= get_window_scoring(horizontal_window, OPONENTE)
            
            #Amenazas verticales
            vertical_window = board[j:j+6,i]
            score += get_window_scoring(vertical_window, JUGADOR)
            score -= get_window_scoring(vertical_window, OPONENTE)   
    
    return score

    

    




def flush_output():
    import sys
    sys.stdout.flush()