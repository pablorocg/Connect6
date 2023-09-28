from tools import *
from heuristicas import *
import random

class SearchEngine():
    def __init__(self):
        self.m_board = None
        self.m_chess_type = None
        self.m_alphabeta_depth = None
        self.m_total_nodes = 0

    def before_search(self, board, color, alphabeta_depth):
        """
        Funcion que se ejecuta antes de realizar una busqueda y 
        que inicializa los valores de la clase
        
        """
        self.m_board = [row[:] for row in board]
        self.m_chess_type = color
        self.m_alphabeta_depth = alphabeta_depth
        self.m_total_nodes = 0


        
    def check_first_move(self):
        """
        Funcion que verifica si es el primer movimiento
        """
        for i in range(1,len(self.m_board)-1):
            for j in range(1, len(self.m_board[i])-1):
                if(self.m_board[i][j] != Defines.NOSTONE):
                    return False
        return True
    
    
    
    def get_window_scoring(window, player):
        """
        Calcula el score de una ventana de 6 fichas, para un jugador dado. 
        La puntuacion es 2^(n + k) donde n es el numero de fichas del jugador en la ventana y k es el coeficiente de la exponencial.
        """
        k = 1# Constante de la exponencial (Cuanto mayor sea, mas se premia el numero de fichas del jugador)
        window = np.array(window)
        
        for i in range(1, 7):
            if np.count_nonzero(window == player) == i and np.count_nonzero(window == 0) == 6-i:
                return 2**(i + k)
        
    def get_score(self, m_board, player):
        """
        Funcion que evalua el estado de una partida basandose en el numero de amenazas de cada
        jugador y la diferencia entre ellas para determinar quien tiene ventaja en la partida.
        :param m_board: Tablero de juego

        """

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
    

    


    
    def is_terminal_node(self, m_board, preMove):
            """
            Funcion que verifica si un nodo es terminal
            :param m_board: Tablero de juego
            """
            return self.check_win(m_board)  or len(self.get_valid_locations(m_board)) == 0








    
    def get_best_move(self, m_board, ourColor, depth=3):
       
        posibles_movimientos = self.get_valid_locations(m_board)
        mejor_movimiento = None
        mejor_evaluacion = -np.inf

        for movimiento in posibles_movimientos:
            nuevo_estado = m_board.copy()
            nuevo_estado[movimiento[0]][movimiento[1]] = ourColor
            evaluacion = self.minimax_alphabeta(nuevo_estado, depth-1, False, -np.inf, np.inf, ourColor)

            if evaluacion > mejor_evaluacion:
                mejor_evaluacion = evaluacion
                mejor_movimiento = movimiento

        return mejor_movimiento


    def minimax_alphabeta(self, nodo, depth, es_max, alpha, beta, ourColor, preMove=None):
        if depth == 0 or self.is_terminal_node(nodo, preMove):
            return self.get_score(nodo, ourColor)

        if es_max:
            max_eval = -np.inf
            for hijo in self.generar_hijos(nodo):
                eval = self.minimax_alphabeta(hijo, depth-1, False, alpha, beta)
                max_eval = max(max_eval, eval)
                alpha = max(alpha, eval)
                if beta <= alpha:
                    break
            return max_eval
        else:
            min_eval = np.inf
            for hijo in self.generar_hijos(nodo):
                eval = self.minimax_alphabeta(hijo, depth-1, True, alpha, beta)
                min_eval = min(min_eval, eval)
                beta = min(beta, eval)
                if beta <= alpha:
                    break
            return min_eval
        
    def generar_hijos(self, nodo, ourColor):
        hijos = []
        for move in self.get_valid_locations(nodo):
            hijo = [row[:] for row in nodo]
            make_move(hijo, move, ourColor)
            hijos.append(hijo)
        return hijos
    
    
    
    
    def get_valid_locations(self, m_board):
        """
        Funcion que obtiene las posiciones no ocupadas en el tablero (posibles movimientos). Devuelve una lista de tuplas (x,y)
        :param m_board: Tablero de juego
        """
        # Lista de posiciones no ocupadas
        valid_locations = []
        for idx, fila in enumerate(m_board):
            for idy, elemento in enumerate(fila):
                if elemento == 0:
                    valid_locations.append((idx, idy))
        return valid_locations
        
    def is_terminal_node(self, m_board):
	    return is_win_by_premove(m_board, 1) or is_win_by_premove(m_board, 2) or len(self.get_valid_locations(m_board)) == 0
          
def flush_output():
    import sys
    sys.stdout.flush()



    # def alpha_beta_search(self, depth, alpha, beta, ourColor, bestMove, preMove):
    #     """
        
    #     """
    #     print('OurColor: ', ourColor)
    #     #Check game result
    #     if (is_win_by_premove(self.m_board, preMove)):
    #         if (ourColor == self.m_chess_type):
    #             #Opponent wins.
    #             return 0;
    #         else:
    #             #Self wins.
    #             return Defines.MININT + 1;
        
    #     alpha = 0
    #     if(self.check_first_move()):
    #         bestMove.positions[0].x = 10
    #         bestMove.positions[0].y = 10
    #         bestMove.positions[1].x = 10
    #         bestMove.positions[1].y = 10
            
    #     else:
            
    #         # move1 = self.find_possible_move()
    #         # print('Move1: ', move1)
    #         move1 = self.get_best_move(self.m_board, ourColor, 3)#Obtiene el mejor movimiento posible (fila, columna)
    #         # get_best_move(self, m_board, ourColor, depth=3)
    #         bestMove.positions[0].x = move1[0]
    #         bestMove.positions[0].y = move1[1]
            
    #         make_move(self.m_board,bestMove,ourColor)
            
    #         '''#Check game result
    #         if (is_win_by_premove(self.m_board, bestMove)):
    #             #Self wins.
    #             return Defines.MININT + 1;'''
            
    #         # available_moves = self.get_valid_locations(self.m_board)

    #         move2 = self.get_best_move(self.m_board, ourColor, 3)#Obtiene el mejor movimiento posible (fila, columna)
    #         bestMove.positions[1].x = move2[0]
    #         bestMove.positions[1].y = move2[1]
    #         make_move(self.m_board,bestMove,ourColor)
        
        
    #     return alpha

# Funcion que pone piedras por orden del tablero (modificar)    
    # def find_possible_move(self):
    #     """
    #     Devuelve posicion del tablero (fila, columna) donde poner la piedra
    #     """
    #     for i in range(1,len(self.m_board)-1):
    #         for j in range(1, len(self.m_board[i])-1):
    #             if(self.m_board[i][j] == Defines.NOSTONE):
    #                 return (i,j)
    #     return (-1,-1)

# def get_score(self, m_board, player):
    #     """
    #     Funcion que evalua el estado de una partida basandose en el numero de amenazas de cada 
    #     jugador y la diferencia entre ellas para determinar quien tiene ventaja en la partida.
    #     :param m_board: Tablero de juego
    #     :param player: Jugador actual (blanco: 2 o negro: 1)
         
    #     """
    #     m_board = np.array(m_board)
    #     m_board = m_board[1:-1, 1:-1]

    #     if player == 2:
    #         opponent = 1
    #     else:
    #         opponent = 2
        
    #     amenazas_jug, amenazas_op = 0, 0
        
        
    #     # Amenazas horizontales
    #     for i in range(m_board.shape[0]):
    #         for j in range(m_board.shape[1] - 6 + 1):

    #             if np.count_nonzero(m_board[i][j:j+6] == player) >= 4 and np.count_nonzero(m_board[i][j:j+6] == 0) > 1:
    #                 amenazas_jug += 1
                    
    #             if np.count_nonzero(m_board[i][j:j+6] == opponent) >= 4 and np.count_nonzero(m_board[i][j:j+6] == 0) > 1:
    #                 amenazas_op += 1
                    
                    
        
    #     # Amenazas verticales
    #     for col in range(m_board.shape[1]):
    #         for row in range(m_board.shape[0] - 6 + 1):
                
    #             if np.count_nonzero(m_board[row:row+6, col] == player) >= 4 and np.count_nonzero(m_board[row:row+6, col] == 0) > 1:
    #                 amenazas_jug += 1
                    
    #             if np.count_nonzero(m_board[row:row+6, col] == 2) >= 4 and np.count_nonzero(m_board[row:row+6, col] == 0) > 1:
    #                 amenazas_op += 1

    #     # Amenazas diagonales
    #     for k in range(-13, 14):
    
    #         for i in range(len(np.diag(m_board, k=k)) - 6 + 1):
    #             v1 = np.diag(m_board, k=k)[i:i+6]
    #             v2 = np.diag(np.flip(m_board, axis=0), k=k)[i:i+6]

    #             if np.count_nonzero(v1 == player) >= 4 and np.count_nonzero(v1 == 0) > 1:
    #                 amenazas_jug += 1

    #             if np.count_nonzero(v1 == opponent) >= 4 and np.count_nonzero(v1 == 0) > 1:
    #                 amenazas_op += 1

    #             if np.count_nonzero(v2 == player) >= 4 and np.count_nonzero(v2 == 0) > 1:
    #                 amenazas_jug += 1

    #             if np.count_nonzero(v2 == opponent) >= 4 and np.count_nonzero(v2 == 0) > 1:
    #                 amenazas_op += 1

    #     if amenazas_jug == amenazas_op:
    #         return 0
        
    #     elif amenazas_jug > amenazas_op:
    #         diferencia = amenazas_jug - amenazas_op
    #         return 1 - 1/(1 + diferencia)
        
    #     else:
    #         diferencia = amenazas_op - amenazas_jug
    #         return -1 + 1/(1 + diferencia)
