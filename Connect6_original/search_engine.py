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
    
    
    
    def evaluate_board(self, board,  player):
        """
        Función que evalúa el estado de una partida basándose en el número de amenazas de cada
        jugador y la diferencia entre ellas para determinar quién tiene ventaja en la partida.
        """
        board = board[1:-1, 1:-1]
        
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
        """
        
        """
        

        if self.check_first_move():
            move = create_move((10,10),(10,10))
            make_move(self.state, move, ourColor)
            return self.evaluate_board(self.player)
              
        if self.is_terminal(self.state, preMove) or depth == 0:
            return self.evaluate_board(self.player)
        
        moves = self.get_available_moves(self.state)
        print(moves)
        # moves = self.order_moves_heuristic(moves)
        
        is_maximizing = ourColor != self.player
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
        # Apply the best move to the board
        if best_move:
            # Crear el movimiento
            best_move = create_move(best_move)
            make_move(self.state, best_move, ourColor)

        return best_score


    # Implementacion minimax con poda alfa-beta---------------------------------------------------------------------
#     función alfa-beta(nodo //en nuestro caso el tablero, profundidad, α, β, jugador)
    @mide_tiempo
    def alfa_beta(self, nodo, depth, alfa, beta, jugador):
        """
        Alpha-beta search algorithm.
        
        Parameters:
        ----------
        nodo (int): Nodo a explorar.
        depth (int): Depth of the search tree.
        alpha (int): Alpha value.
        beta (int): Beta value.
        jugador (int): Color of the player.

        Returns:
        -------
        int: The best score found.
        """

        best_move = None

        if self.check_first_move():
            best_move = create_move((10,10),(10,10))
            make_move(self.state, best_move, self.player)
            return self.evaluate_board(nodo, jugador)
        
        # Si nodo es un nodo terminal o profundidad = 0 --> devolver el valor heurístico del nodo
        if self.is_terminal(nodo) or depth == 0:

            return self.evaluate_board(nodo, jugador)
        
        # Si jugador == max
        if jugador == self.player:
            
            for move, child in self.get_moves_and_children(nodo, jugador):
                
                score = self.alfa_beta(child, depth-1, alfa, beta, 3 - jugador)
                
                if score > alfa:
                    alfa = score
                    best_move = move

                if beta <= alfa:    
                    break
            if best_move:
                best_move = create_move(best_move)
                make_move(self.state, best_move, self.player)
            return alfa
        # Si no
        else:
            
            for move, child in self.get_moves_and_children(nodo, 3 - jugador):
                
                score = self.alfa_beta(child, depth-1, alfa, beta, 3 - jugador)
                
                if score < beta:
                    beta = score
                    best_move = move

                if beta <= alfa:
                    break

            if best_move:
                best_move = create_move(best_move)
                make_move(self.state, best_move, self.player)
            
            return beta




    # Codigo intentando paralelizar ------------------------------------------------------------------------------
    

    def evaluate_move_parallel(self, move, depth, alpha, beta, ourColor, bestMove):
        move = create_move(move)
        make_move(self.state, move, ourColor)
        eval_score = self.alpha_beta_search(depth-1, alpha, beta, 3 - ourColor, bestMove, move)
        unmake_move(self.state, move)
        return eval_score, move

    def alpha_beta_search_paralel(self, depth, alpha, beta, ourColor, bestMove, preMove):
        """
        Alpha-beta search algorithm with parallel evaluation of moves.
        
        Parameters:
        ----------
        depth (int): Depth of the search tree.
        alpha (int): Alpha value.
        beta (int): Beta value.
        ourColor (int): Color of the player.
        bestMove (Move): Best move found so far.
        preMove (Move): Previous move made on the board.

        Returns:
        -------
        int: The best score found.
        """

        if self.check_first_move():# Comprobar si es el primer movimiento
            move = create_move((10,10),(10,10)) # Crear el movimiento
            make_move(self.state, move, ourColor) # Realizar el movimiento en el tablero
            return self.evaluate_board(self.player) # Evaluar el tablero resultante


        if self.is_terminal(self.state, preMove) or depth == 0: # Comprobar si es un nodo terminal

            return self.evaluate_board(self.player) # Evaluar el tablero resultante
            
        moves = self.get_available_moves(self.state)
        is_maximizing = ourColor != self.player

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
    def is_terminal(self, board):
        return is_win(board) or len(self.get_available_moves(board)) == 0
    
    def check_first_move(self):
        return all(self.state[i][j] == Defines.NOSTONE for i in range(1, len(self.state)-1) for j in range(1, len(self.state[i])-1))
    
    
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
    
    def get_moves_and_children(self, node, player):
        #Obtener los movimientos disponibles
        moves = self.get_available_moves(node)

        # Crear los hijos
        children = []
        for move1, move2 in moves:
            child = node.copy()
            child[move1[0], move1[1]] = player
            child[move2[0], move2[1]] = player

            children.append([(move1, move2), child])
        return children




    @mide_tiempo
    def order_moves_heuristic(self, moves):
        """
        Ordena los movimientos de mayor a menor prioridad, según la heurística de ordenación.
        """
        print('Ordenando movimientos...')
        return sorted(moves, key=lambda move: self.evaluate_move(move), reverse=True)
    
    
    
    

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