import numpy as np
import unittest
from search_engine import SearchEngine

class TestSearchEngine(unittest.TestCase):
    
    def setUp(self):
        self.engine = SearchEngine()
        
    def test_evaluate_board(self):
        # Test empty board
        board = np.zeros((8,8))
        score = self.engine.evaluate_board(1)
        self.assertEqual(score, 0)
        
        # Test board with one player's pieces
        board = np.zeros((8,8))
        board[0,0] = 1
        board[1,1] = 1
        board[2,2] = 1
        score = self.engine.evaluate_board(1)
        self.assertEqual(score, 3)
        
        # Test board with both players' pieces
        board = np.zeros((8,8))
        board[0,0] = 1
        board[1,1] = 1
        board[2,2] = 1
        board[3,3] = 2
        board[4,4] = 2
        score = self.engine.evaluate_board(1)
        self.assertEqual(score, 0)
        
    def test_get_available_moves(self):
        # Test empty board
        board = np.zeros((8,8))
        moves = self.engine.get_available_moves(board)
        self.assertEqual(len(moves), 28)
        
        # Test board with one player's pieces
        board = np.zeros((8,8))
        board[0,0] = 1
        board[1,1] = 1
        board[2,2] = 1
        moves = self.engine.get_available_moves(board)
        self.assertEqual(len(moves), 25)
        
        # Test board with both players' pieces
        board = np.zeros((8,8))
        board[0,0] = 1
        board[1,1] = 1
        board[2,2] = 1
        board[3,3] = 2
        board[4,4] = 2
        moves = self.engine.get_available_moves(board)
        self.assertEqual(len(moves), 23)
        
    def test_order_moves_heuristic(self):
        # Test empty board
        board = np.zeros((8,8))
        moves = self.engine.get_available_moves(board)
        ordered_moves = self.engine.order_moves_heuristic(moves)
        self.assertEqual(moves, ordered_moves)
        
        # Test board with one player's pieces
        board = np.zeros((8,8))
        board[0,0] = 1
        board[1,1] = 1
        board[2,2] = 1
        moves = self.engine.get_available_moves(board)
        ordered_moves = self.engine.order_moves_heuristic(moves)
        self.assertEqual(ordered_moves[0], ((3,3),(4,4)))
        
        # Test board with both players' pieces
        board = np.zeros((8,8))
        board[0,0] = 1
        board[1,1] = 1
        board[2,2] = 1
        board[3,3] = 2
        board[4,4] = 2
        moves = self.engine.get_available_moves(board)
        ordered_moves = self.engine.order_moves_heuristic(moves)
        self.assertEqual(ordered_moves[0], ((3,2),(4,1)))
        
if __name__ == '__main__':
    unittest.main()





# def test_get_available_moves():
#     se = SearchEngine()
#     board = np.zeros((Defines.GRID_NUM, Defines.GRID_NUM), dtype=int)
#     init_board(board)
#     se.before_search(state = board, player = 1, depth = 0)
#     ti = time.time()
#     moves = se.get_available_moves(se.state)
#     tf = time.time()
#     print("time: {}".format(tf-ti))
#     assert len(moves) == 64980
#     print("test_get_available_moves passed")



# test_get_available_moves()