from tools import *

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

    def alpha_beta_search(self, depth, alpha, beta, ourColor, bestMove, preMove):
        """
        Funcion que realiza la busqueda del mejor movimiento
        """
        print('OurColor: ', ourColor)
        #Check game result
        if (is_win_by_premove(self.m_board, preMove)):
            if (ourColor == self.m_chess_type):
                #Opponent wins.
                return 0;
            else:
                #Self wins.
                return Defines.MININT + 1;
        
        alpha = 0
        if(self.check_first_move()):
            bestMove.positions[0].x = 10
            bestMove.positions[0].y = 10
            bestMove.positions[1].x = 10
            bestMove.positions[1].y = 10
        else:   
            move1 = self.find_possible_move()
            bestMove.positions[0].x = move1[0]
            bestMove.positions[0].y = move1[1]
            bestMove.positions[1].x = move1[0]
            bestMove.positions[1].y = move1[1]
            make_move(self.m_board,bestMove,ourColor)
            
            '''#Check game result
            if (is_win_by_premove(self.m_board, bestMove)):
                #Self wins.
                return Defines.MININT + 1;'''
            
            move2 = self.find_possible_move()
            bestMove.positions[1].x = move2[0]
            bestMove.positions[1].y = move2[1]
            make_move(self.m_board,bestMove,ourColor)

        return alpha
        
    def check_first_move(self):
        """
        Funcion que verifica si es el primer movimiento
        """
        for i in range(1,len(self.m_board)-1):
            for j in range(1, len(self.m_board[i])-1):
                if(self.m_board[i][j] != Defines.NOSTONE):
                    return False
        return True
        
    def find_possible_move(self):
        """
        Funcion que encuentra un movimiento posible
        """
        for i in range(1,len(self.m_board)-1):
            for j in range(1, len(self.m_board[i])-1):
                if(self.m_board[i][j] == Defines.NOSTONE):
                    return (i,j)
        return (-1,-1)

    def get_threats(self, m_board, player):
        m_board = np.array(m_board)
        m_board = m_board[1:-1, 1:-1]

        m_board = np.where(m_board == 2, -1, m_board) * player
        amenazas_jug, amenazas_op = 0, 0
        
        
        # Amenazas horizontales
        for i in range(m_board.shape[0]):
            for j in range(m_board.shape[1] - 6 + 1):

                if np.count_nonzero(m_board[i][j:j+6] == 1) >= 4 and np.count_nonzero(m_board[i][j:j+6] == 0) > 1:
                    amenazas_jug += 1
                    
                if np.count_nonzero(m_board[i][j:j+6] == -1) >= 4 and np.count_nonzero(m_board[i][j:j+6] == 0) > 1:
                    amenazas_op += 1
                    
                    
        
        # Amenazas verticales
        for col in range(m_board.shape[1]):
            for row in range(m_board.shape[0] - 6 + 1):
                
                if np.count_nonzero(m_board[row:row+6, col] == 1) >= 4 and np.count_nonzero(m_board[row:row+6, col] == 0) > 1:
                    amenazas_jug += 1
                    
                if np.count_nonzero(m_board[row:row+6, col] == -1) >= 4 and np.count_nonzero(m_board[row:row+6, col] == 0) > 1:
                    amenazas_op += 1

        # Amenazas diagonales
        for k in range(-13, 14):
    
            for i in range(len(np.diag(m_board, k=k)) - 6 + 1):
                v1 = np.diag(m_board, k=k)[i:i+6]
                v2 = np.diag(np.flip(m_board, axis=0), k=k)[i:i+6]

                if np.count_nonzero(v1 == 1) >= 4 and np.count_nonzero(v1 == 0) > 1:
                    amenazas_jug += 1

                if np.count_nonzero(v1 == -1) >= 4 and np.count_nonzero(v1 == 0) > 1:
                    amenazas_op += 1

                if np.count_nonzero(v2 == 1) >= 4 and np.count_nonzero(v2 == 0) > 1:
                    amenazas_jug += 1

                if np.count_nonzero(v2 == -1) >= 4 and np.count_nonzero(v2 == 0) > 1:
                    amenazas_op += 1
        
        
        if amenazas_jug == amenazas_op:
            return 0
        
        elif amenazas_jug > amenazas_op:
            diferencia = amenazas_jug - amenazas_op
            return 1 - 1/(1 + diferencia)
        
        else:
            diferencia = amenazas_op - amenazas_jug
            return -1 + 1/(1 + diferencia)

        # return amenazas_jug, amenazas_op



def flush_output():
    import sys
    sys.stdout.flush()



