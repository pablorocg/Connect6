from tools import *
import itertools as it
import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed

class SearchEngine():
    """
    Implement a search engine for Connect6 using minimax with alpha-beta pruning.
    Using numpy array for the board representation and operations.
    """
    def __init__(self):
        self.state = None
        self.player = None
        self.depth = None
        self.m_total_nodes = 0

    def before_search(self, state, player, depth):
        self.state = state
        self.player = player
        self.depth = depth
    
    
    @mide_tiempo
    def evaluate_board(self, player):
        """
        Función que evalúa el estado de una partida basándose en el número de amenazas de cada
        jugador y la diferencia entre ellas para determinar quién tiene ventaja en la partida.
        """
        board = self.state[1:-1, 1:-1]
        
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


    @mide_tiempo
    def alpha_beta_search(self, depth, alpha, beta, ourColor, bestMove, preMove):
   
        

        if self.check_first_move():
            move = create_move((10,10),(10,10))
            make_move(self.state, move, ourColor)
            return self.evaluate_board(self.player)
              
        if self.is_terminal(self.state, preMove):
            return self.evaluate_board(self.player)
        
        moves = self.get_available_moves(self.state)
        # moves = self.order_moves_heuristic(moves)
        
        is_maximizing = ourColor == self.player
        best_score = -np.inf if is_maximizing else np.inf

        
        for move in moves:
            move = create_move(move)
            # Realiza el movimiento en 'board' y busca la evaluación.
            make_move(self.state, move, ourColor)
            eval_score = self.alpha_beta_search(depth-1, alpha, beta, 3 - ourColor, bestMove, move)
            unmake_move(self.state, move)

            if is_maximizing:
                if eval_score > best_score:
                    best_score = eval_score
                    best_move = move
                alpha = max(alpha, eval_score)
            else:
                if eval_score < best_score:
                    best_score = eval_score
                    best_move = move
                beta = min(beta, eval_score)
            if beta <= alpha:
                break
        return best_score
            
    # Codigo intentando paralelizar ------------------------------------------------------------------------------
    

    def evaluate_move_parallel(self, move, depth, alpha, beta, ourColor, bestMove):
        move = create_move(move)
        make_move(self.state, move, ourColor)
        eval_score = self.alpha_beta_search(depth-1, alpha, beta, 3 - ourColor, bestMove, move)
        unmake_move(self.state, move)
        return eval_score, move

    def alpha_beta_search_paralel(self, depth, alpha, beta, ourColor, bestMove, preMove):
        if self.check_first_move():
            move = create_move((10,10),(10,10))
            make_move(self.state, move, ourColor)
            return self.evaluate_board(self.player)
                
        if self.is_terminal(self.state, preMove):
            return self.evaluate_board(self.player)
            
        moves = self.get_available_moves(self.state)
        is_maximizing = ourColor == self.player

        # Parallel evaluation of moves
        results = Parallel(n_jobs=-1)(delayed(self.evaluate_move_parallel)(move, depth, alpha, beta, ourColor, bestMove) for move in moves)

        best_score = -np.inf if is_maximizing else np.inf
        best_move = None
        for eval_score, move in results:
            if is_maximizing:
                if eval_score > best_score:
                    best_score = eval_score
                    best_move = move
                alpha = max(alpha, eval_score)
            else:
                if eval_score < best_score:
                    best_score = eval_score
                    best_move = move
                beta = min(beta, eval_score)
            if beta <= alpha:
                break

        return best_score


    
    #-------------------------------------------------------------------------------------------------------------
    def is_terminal(self, board, preMove):
        return is_win_by_premove(board, preMove) or len(self.get_available_moves(board)) == 0
    
    def check_first_move(self):
        return all(self.state[i][j] == Defines.NOSTONE for i in range(1, len(self.state)-1) for j in range(1, len(self.state[i])-1))
    
    @mide_tiempo
    def get_available_moves(self, board):
        """
        Returns a list of available moves on the board.

        Parameters:
        board (numpy.ndarray): A numpy array representing the Connect6 board.

        Returns:
        list: A list of tuples representing the available moves on the board.
        """
        NOSTONE = Defines.NOSTONE
        
        # Encuentra las coordenadas de las casillas vacías
        empty_coords = np.argwhere(board == NOSTONE)
        
        # Devuelve todas las combinaciones de 2 movimientos posibles
        return list(it.combinations(map(tuple, empty_coords), 2))
    
    @mide_tiempo
    def order_moves_heuristic(self, moves):
        """
        Ordena los movimientos de mayor a menor prioridad, según la heurística de ordenación.
        """
        print('Ordenando movimientos...')
        return sorted(moves, key=lambda move: self.evaluate_move(move), reverse=True)
    
    @mide_tiempo
    def evaluate_move(self, move):
        """
        Evalúa un movimiento según la heurística de ordenación.
        """
        
        move = create_move(move)
        # Realiza el movimiento en el tablero copiado
        make_move(self.state, move, self.player)
        # Evalúa el tablero resultante
        score = self.evaluate_board(self.player)
        unmake_move(self.state, move)
        return score
    

def flush_output():
    import sys
    sys.stdout.flush()

def mide_tiempo(funcion):
    def funcion_medida(*args, **kwargs):
        inicio = time.time()
        c = funcion(*args, **kwargs)
        print(f"Time taken to execute {funcion.__name__}: {time.time() - inicio}")
        return c
    return funcion_medida