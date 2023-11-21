import numpy as np
import random
import tools as tl
from joblib import Parallel, delayed

# MINIMAX IMPLEMENTATION-------------------------------------------------------------------------
class MiniMax:
    """
    Implementacion del algoritmo de busqueda MiniMax sin optimizaciones. No poner a mas de 1 de profundidad, ya que es muy lento. 

    Atributos:
    m_board (np.ndarray): El estado del juego.
    m_chess_type (int): El color del jugador (1 o 2).
    m_depth (int): La profundidad de busqueda.

    """
    
    def __init__(self):
        self.m_board = None
        self.m_chess_type = None
        self.m_depth = None
        self.m_is_maximizing = None


    def before_search(self, board, color, depth):
        self.m_board = board.copy() # Actualiza el tablero
        self.m_chess_type = color # Actualiza el color del jugador (1 o 2)
        self.m_depth = depth # Actualiza la profundidad de búsqueda
        self.m_is_maximizing = True if self.m_chess_type == 1 else False

    def evaluate_board(self, board):
        return tl.evaluate_board_new(board)

    def search(self, state, depth, is_maximizing):
        
        if depth == 0 or tl.check_winner(state) != 0 or len(tl.get_available_moves(state)) == 0:
            return self.evaluate_board(state)

        if is_maximizing:
            max_value = -np.inf
            for child in self.get_children(state):
                value = self.search(child, depth-1, False)
                max_value = max(max_value, value)
            return max_value
        else:
            min_value = np.inf
            for child in self.get_children(state):
                value = self.search(child, depth-1, True)
                min_value = min(min_value, value)
            return min_value

    def get_best_move(self):
        best_score = -np.inf if self.m_is_maximizing else np.inf
        best_move = None

        for move in tl.get_available_moves(self.m_board):
            child_state = self.m_board.copy()
            tl.make_move(child_state, move, self.m_chess_type)
            
            score = self.search(child_state, self.m_depth-1, not self.m_is_maximizing)
            if (self.m_is_maximizing and score > best_score) or (not self.m_is_maximizing and score < best_score):
                best_score = score
                best_move = move

        return best_score, best_move


    def get_children(self, state):
        children = []
        for move in tl.get_available_moves(state):
            child_state = state.copy()
            tl.make_move(child_state, move, self.m_chess_type)
            children.append(child_state)
        return children
    




# ALPHA-BETA PRUNING IMPLEMENTATION--------------------------------------------------------------
# class MiniMaxAlphaBeta:
#     def __init__(self):
#         self.m_board = None
#         self.m_chess_type = None
#         self.m_depth = None
#         self.m_is_maximizing = None
#         self.node_count = 0


#     def before_search(self, board, color, depth):
#         self.m_board = board.copy()
#         self.m_chess_type = color
#         self.m_depth = depth
#         self.m_is_maximizing = True if self.m_chess_type == 1 else False
#         self.node_count = 0


#     def evaluate_board(self, board):
#         return tl.defensive_evaluate_state(board)#tl.get_score(board)

#     def search(self, state, depth, is_maximizing, alpha=-np.inf, beta=np.inf):
#         self.node_count += 1

#         if depth == 0 or tl.check_winner(state) != 0 or len(tl.get_available_moves(state)) == 0:
#             return self.evaluate_board(state)

#         if is_maximizing:
#             max_value = -np.inf
#             for child in self.get_children(state):
#                 value = self.search(child, depth-1, False, alpha, beta)
#                 max_value = max(max_value, value)
#                 alpha = max(alpha, value)
#                 if beta <= alpha:
#                     print(f"Poda en profundidad {depth} con alpha={alpha} y beta={beta}")
#                     break
#             return max_value
#         else:
#             min_value = np.inf
#             for child in self.get_children(state):
#                 value = self.search(child, depth-1, True, alpha, beta)
#                 min_value = min(min_value, value)
#                 beta = min(beta, value)
#                 if beta <= alpha:
#                     print(f"Poda en profundidad {depth} con alpha={alpha} y beta={beta}")
#                     break
#             return min_value

#     def get_best_move(self):
#         best_score = -np.inf if self.m_is_maximizing else np.inf
#         best_move = None

#         moves = []
#         for move in tl.get_available_moves(self.m_board):
#             child_state = self.m_board.copy()
#             tl.make_move(child_state, move, self.m_chess_type)
            
#             move.score = self.search(child_state, self.m_depth-1, not self.m_is_maximizing)
#             moves.append(move)

#         # Ordenamos los movimientos basados en los scores
#         moves.sort(key=lambda x: x.score, reverse=self.m_is_maximizing)

#         for move in moves:
#             if (self.m_is_maximizing and move.score > best_score) or (not self.m_is_maximizing and move.score < best_score):
#                 best_score = move.score
#                 best_move = move

#         return best_score, best_move

#     def get_children(self, state):
#         children = []
#         for move in tl.get_available_moves(state):
#             child_state = state.copy()
#             tl.make_move(child_state, move, self.m_chess_type)
#             children.append(child_state)
#         return children
class MiniMaxAlphaBeta:
    def __init__(self):
        self.m_board = None
        self.m_chess_type = None
        self.m_depth = None
        self.m_is_maximizing = None
        self.node_count = 0

    def before_search(self, board, color, depth):
        self.m_board = board.copy()
        self.m_chess_type = color
        self.m_depth = depth
        self.m_is_maximizing = True if self.m_chess_type == 1 else False
        self.node_count = 0

    def evaluate_board(self, board):
        return tl.defensive_evaluate_state_v2(board)

    def search(self, state, depth, is_maximizing, alpha=-np.inf, beta=np.inf):
        self.node_count += 1

        if depth == 0 or tl.check_winner(state) != 0 or len(tl.get_available_moves(state)) == 0:
            return self.evaluate_board(state)

        children = self.get_ordered_children(state, is_maximizing)

        if is_maximizing:
            max_value = -np.inf
            for child in children:
                value = self.search(child, depth-1, False, alpha, beta)
                max_value = max(max_value, value)
                alpha = max(alpha, value)
                if beta <= alpha:
                    print(f"Poda en profundidad {depth} con alpha={alpha} y beta={beta}")
                    break
            return max_value
        else:
            min_value = np.inf
            for child in children:
                value = self.search(child, depth-1, True, alpha, beta)
                min_value = min(min_value, value)
                beta = min(beta, value)
                if beta <= alpha:
                    print(f"Poda en profundidad {depth} con alpha={alpha} y beta={beta}")
                    break
            return min_value

    def get_best_move(self):
        best_score = -np.inf if self.m_is_maximizing else np.inf
        best_move = None

        if tl.check_first_move(self.m_board):
            best_move = tl.create_move(((10, 10), (10, 10)))
            return 0, best_move
            

        moves = tl.get_available_moves_with_score(self.m_board, self.m_chess_type)
        # Ordenamos los movimientos basados en los scores
        moves.sort(key=lambda x: x.score, reverse=self.m_is_maximizing)

        for move in moves:
            if (self.m_is_maximizing and move.score > best_score) or (not self.m_is_maximizing and move.score < best_score):
                best_score = move.score
                best_move = move

        return best_score, best_move

    def get_ordered_children(self, state, is_maximizing):
        children = []
        moves = tl.get_available_moves(state)
        for move in moves:
            child_state = state.copy()
            tl.make_move(child_state, move, self.m_chess_type)
            move.score = self.evaluate_board(child_state)  # Set heuristic score for sorting
            children.append(move)
        
        # Order children based on heuristic scores
        children.sort(key=lambda x: x.score, reverse=is_maximizing)
        return children


# MONTE CARLO TREE SEARCH IMPLEMENTATION--------------------------------------------------------------



def flush_output():
    import sys
    sys.stdout.flush()