import numpy as np
import tools as tl
from defines import *
import time

class HeuristicSearch:
    """
    Represents a heuristic search algorithm for finding the best move in a game.

    Attributes:
        m_board (numpy.ndarray): The current game board.
        m_chess_type (int): The type of chess piece for the AI player.
        m_depth (int): The maximum depth to search in the game tree.
        m_is_maximizing (bool): Indicates whether the AI player is maximizing or minimizing.
        node_count (int): The number of nodes visited during the search.
    """

    def __init__(self):
        self.m_board = None
        self.m_chess_type = None
        self.m_depth = None
        self.m_is_maximizing = None
        

    def before_search(self, board, color, depth):
        """
        Initializes the search parameters before starting the search.

        Args:
            board (numpy.ndarray): The current game board.
            color (int): The type of chess piece for the AI player.
            depth (int): The maximum depth to search in the game tree.
        """
        self.m_board = np.copy(board)
        self.m_chess_type = color
        self.m_depth = depth
        self.m_is_maximizing = True if self.m_chess_type == 1 else False
        

    def get_best_move(self):
        """
        Returns the best move found by the search algorithm.

        Returns:
            tuple: A tuple containing the evaluation score and the best move.
        """
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


class MiniMaxAlphaBeta:
    """
    Implementation of the MiniMax algorithm with Alpha-Beta pruning for game search.

    Attributes:
        m_board (Board): The current game board.
        m_chess_type (int): The type of chess piece for the AI player.
        m_depth (int): The maximum depth to search in the game tree.
        m_is_maximizing (bool): Flag indicating whether the AI player is maximizing or minimizing.
        node_count (int): The number of nodes visited during the search.

    Methods:
        before_search(board, color, depth): Initializes the search parameters before starting the search.
        evaluate_board(board): Evaluates the given game board state.
        search(state, depth, is_maximizing, alpha, beta): Performs the MiniMax search with Alpha-Beta pruning.
        get_best_move(): Returns the best move found by the search algorithm.
        get_ordered_children(state, is_maximizing): Returns the ordered list of child states based on heuristic scores.
    """

    def __init__(self):
        self.m_board = None
        self.m_chess_type = None
        self.m_depth = None
        self.m_is_maximizing = None
        self.node_count = 0

    def before_search(self, board, color, depth):
        """
        Initializes the search parameters before starting the search.

        Args:
            board (Board): The current game board.
            color (int): The type of chess piece for the AI player.
            depth (int): The maximum depth to search in the game tree.
        """
        self.m_board = np.copy(board)
        self.m_chess_type = color
        self.m_depth = depth
        self.m_is_maximizing = True if self.m_chess_type == 1 else False
        self.node_count = 0

    def evaluate_board(self, board):
        """
        Evaluates the given game board state.

        Args:
            board (Board): The game board state to evaluate.

        Returns:
            float: The evaluation score for the given board state.
        """
        return tl.defensive_evaluate_state(board, self.m_chess_type)

    def search(self, state, depth, is_maximizing, alpha=-np.inf, beta=np.inf):
        
        winner = tl.check_winner(state)
        available_moves = np.any(state == 0)  # tl.get_available_moves(state)

        #Stop searching if the game is over or the maximum depth is reached (Nodo terminal)
        if depth == 0 or winner != 0 or not available_moves:
            return self.evaluate_board(state)

        children = self.get_ordered_children(state, is_maximizing)# Obtener los hijos ordenados por score (heuristica)

        if is_maximizing:
            max_value = -np.inf
            for child in children:
                value = self.search(child, depth - 1, False, alpha, beta)
                max_value = max(max_value, value)
                alpha = max(alpha, value)
                if beta <= alpha:
                    break
            return max_value
        else:
            min_value = np.inf
            for child in children:
                value = self.search(child, depth - 1, True, alpha, beta)  # Note: changed to False
                min_value = min(min_value, value)
                beta = min(beta, value)
                if beta <= alpha:
                    break
            return min_value


    def get_best_move(self):
        
        best_score = -np.inf if self.m_is_maximizing else np.inf
        best_move = None

        for move in tl.get_available_moves_optimizada(self.m_board):
            child_state = self.m_board.copy()
            tl.make_move(child_state, move, self.m_chess_type)

            score = self.search(child_state, self.m_depth - 1, not self.m_is_maximizing)
            if (self.m_is_maximizing and score > best_score) or (not self.m_is_maximizing and score < best_score):
                best_score = score
                best_move = move
        
        return best_score, best_move

    def get_ordered_children(self, state, is_maximizing) -> list[StoneMove]:
            """
            Returns the ordered list of child states based on heuristic scores.

            Args:
                state (Board): The current game board state.
                is_maximizing (bool): Flag indicating whether the current player is maximizing or minimizing.

            Returns:
                list: The ordered list of child states.
            """
            children = []
            moves = tl.get_available_moves_optimizada(state)
            moves.sort(key=lambda x: x.score, reverse=is_maximizing)
            for move in moves[:4]:
                child_state = state.copy()
                tl.make_move(child_state, move, self.m_chess_type)
                children.append(child_state)
            return children

def flush_output():
    import sys
    sys.stdout.flush()