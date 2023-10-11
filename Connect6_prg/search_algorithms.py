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
        #tl.get_score(board)
        return tl.defensive_evaluate_state(board)

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
    
# MINIMAX PARALELIZADO IMPLEMENTATION-------------------------------------------------------------------------
class MiniMaxParalelizado:
    """
    Implementacion del algoritmo de busqueda MiniMax. 

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
        return tl.defensive_evaluate_state(board)

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

        # Paralelizamos la búsqueda usando joblib
        moves = tl.get_available_moves(self.m_board)
        scores = Parallel(n_jobs=-1)(delayed(self.search_parallel)(move) for move in moves)

        # Buscamos el mejor movimiento basado en los resultados
        for score, move in zip(scores, moves):
            if (self.m_is_maximizing and score > best_score) or (not self.m_is_maximizing and score < best_score):
                best_score = score
                best_move = move

        return best_score, best_move

    def search_parallel(self, move):
        board_copy = self.m_board.copy()
        tl.make_move(board_copy, move, self.m_chess_type)
        return self.search(board_copy, self.m_depth-1, not self.m_is_maximizing)

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
        return tl.defensive_evaluate_state(board)

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

        # moves = []
        # for move in tl.get_available_moves(self.m_board):
        #     child_state = self.m_board.copy()
        #     tl.make_move(child_state, move, self.m_chess_type)
            
        #     move.score = self.search(child_state, self.m_depth-1, not self.m_is_maximizing)
        #     moves.append(move)
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



# NEGAMAX IMPLEMENTATION--------------------------------------------------------------        
class NegaMaxAlphaBeta:
    
    def __init__(self):
        self.m_board = None
        self.m_chess_type = None
        self.m_depth = None

    def before_search(self, board, color, depth):
        self.m_board = board.copy()
        self.m_chess_type = 1 if color == 1 else -1  # Convertir a 1 o -1
        self.m_depth = depth

    def evaluate_board(self, board):
        return self.m_chess_type * tl.get_score(board)  # Considerar el color

    def negamax(self, state, depth, color, alpha, beta):
        
        if depth == 0 or tl.check_winner(state) != 0 or len(tl.get_available_moves(state)) == 0:
            return self.evaluate_board(state)
        
        max_value = -np.inf
        for child in self.get_children(state):
            value = -self.negamax(child, depth-1, -color, -beta, -alpha)
            max_value = max(max_value, value)
            alpha = max(alpha, value)
            if alpha >= beta:
                break
        return max_value

    def get_best_move(self):
        best_score = -np.inf
        best_move = None
        alpha = -np.inf
        beta = np.inf

        for move in tl.get_available_moves(self.m_board):
            child_state = self.m_board.copy()
            tl.make_move(child_state, move, 1 if self.m_chess_type == 1 else 2)  # Convertir de nuevo a 1 o 2 para el movimiento
            
            score = -self.negamax(child_state, self.m_depth-1, -self.m_chess_type, -beta, -alpha)
            if score > best_score:
                best_score = score
                best_move = move
            alpha = max(alpha, score)

        return best_score, best_move

    def get_children(self, state):
        children = []
        for move in tl.get_available_moves(state):
            child_state = state.copy()
            tl.make_move(child_state, move, 1 if self.m_chess_type == 1 else 2)  # Convertir de nuevo a 1 o 2 para el movimiento
            children.append(child_state)
        return children


    
# NEGASCOUT IMPLEMENTATION--------------------------------------------------------------
class NegaScout:
    def __init__(self, game_state, depth=3):
        self.game_state = game_state
        self.depth = depth

    def evaluate_board(self, state):
        # This is a simple evaluation function that returns a random score.
        # In a real-world scenario, this function should analyze the board 
        # and return a score representing the advantage of one player over the other.
        return random.random()

    def search(self, state, depth, alpha, beta):
        if depth == 0 or state.check_winner() != 0:
            return self.evaluate_board(state)

        children = self.get_children(state)
        if len(children) == 0:
            return self.evaluate_board(state)

        score = -float("inf")
        for child in children:
            if child == children[0]:
                score = -self.search(child, depth-1, -beta, -alpha)
            else:
                score = -self.search(child, depth-1, -alpha-1, -alpha)
                if alpha < score < beta:
                    score = -self.search(child, depth-1, -beta, -score)
            alpha = max(alpha, score)
            if alpha >= beta:
                break

        return score

    def get_best_move(self):
        best_score = -float("inf")
        best_move = None
        alpha = -float("inf")
        beta = float("inf")

        for child in self.get_children(self.game_state):
            score = -self.search(child, self.depth-1, -beta, -alpha)
            if score > best_score:
                best_score = score
                best_move = child
            alpha = max(alpha, score)
            if alpha >= beta:
                break

        return best_move

    def get_children(self, state):
        children = []
        for move in state.get_available_moves():
            child_state = state.copy()
            child_state.make_move(*move)
            children.append(child_state)
        return children



# MONTE CARLO TREE SEARCH IMPLEMENTATION--------------------------------------------------------------
class MCTSNode:
    """
    
    """
    def __init__(self, game_state, parent=None):
        self.game_state = game_state
        self.parent = parent
        self.children = []
        self.visits = 0
        self.wins = 0
        self.unexplored_moves = game_state.get_available_moves()

    def uct_value(self):
        if self.visits == 0:
            return float("inf")  # Always explore unvisited nodes
        exploitation = self.wins / self.visits
        exploration = np.sqrt(2 * np.log(self.parent.visits) / self.visits)
        return exploitation + exploration

    def best_child(self):
        return max(self.children, key=lambda child: child.uct_value())

    def expand(self):
        move = self.unexplored_moves.pop()
        new_game_state = self.game_state.copy()
        new_game_state.make_move(*move)
        child_node = MCTSNode(new_game_state, parent=self)
        self.children.append(child_node)
        return child_node

    def is_fully_expanded(self):
        return len(self.unexplored_moves) == 0

    def rollout(self):
        current_rollout_state = self.game_state.copy()
        while current_rollout_state.check_winner() == 0:
            possible_moves = current_rollout_state.get_available_moves()
            move = random.choice(possible_moves)
            current_rollout_state.make_move(*move)
        return current_rollout_state.check_winner()

    def backpropagate(self, result):
        self.visits += 1
        if self.game_state.current_player == result:
            self.wins += 1
        if self.parent:
            self.parent.backpropagate(result)

class MCTS:
    def __init__(self, game_state, iterations=1000):
        self.root = MCTSNode(game_state)
        self.iterations = iterations

    def search(self):
        for _ in range(self.iterations):
            node = self.select_node(self.root)
            result = node.rollout()
            node.backpropagate(result)
        return self.root.best_child().game_state

    def select_node(self, node):
        while node.game_state.check_winner() == 0:
            if not node.is_fully_expanded():
                return node.expand()
            node = node.best_child()
        return node
    





def flush_output():
    import sys
    sys.stdout.flush()