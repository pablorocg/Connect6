import numpy as np
from defines import *
from tools import *

def test_get_window_scoring():
    window1 = np.array([1, 0, 0, 0, 0, 0])
    window2 = np.array([1, 1, 0, 0, 0, 0])
    window3 = np.array([1, 1, 0, 1, 0, 0])
    window4 = np.array([1, 1, 0, 0, 1, 1])
    window6 = np.array([1, 1, 1, 1, 1, 1])
    window5 = np.array([1, 1, 1, 1, 1, 0])
    assert get_window_scoring(window1, 1) == 1
    assert get_window_scoring(window1, 2) == 0
    assert get_window_scoring(window2, 1) == 4
    assert get_window_scoring(window2, 2) == 0
    assert get_window_scoring(window3, 1) == 64
    assert get_window_scoring(window3, 2) == 0
    assert get_window_scoring(window4, 1) == 0
    assert get_window_scoring(window4, 2) == 0
    assert get_window_scoring(window5, 1) == 0
    assert get_window_scoring(window5, 2) == 0
    assert get_window_scoring(window6, 1) == 0
    assert get_window_scoring(window6, 2) == 0

    
def test_get_score():
    # Test 1: Empty board
    board = np.zeros((Defines.GRID_NUM, Defines.GRID_NUM), dtype=int)
    init_board(board)
    assert get_score(board, 1) == 0
    assert get_score(board, 2) == 0
    
    # # Test 2: Board with one stone
    # board = np.zeros((8, 8), dtype=int)
    # board[3, 3] = 1
    # assert get_score(board, 1) == 0
    # assert get_score(board, 2) == 0
    
    # # Test 3: Board with two stones
    # board = np.zeros((8, 8), dtype=int)
    # board[3, 3] = 1
    # board[4, 4] = 2
    # assert get_score(board, 1) == 0
    # assert get_score(board, 2) == 0
    
    # # Test 4: Board with three stones
    # board = np.zeros((8, 8), dtype=int)
    # board[3, 3] = 1
    # board[4, 4] = 2
    # board[5, 5] = 1
    # assert get_score(board, 1) == 1
    # assert get_score(board, 2) == -1
    
    # # Test 5: Board with four stones
    # board = np.zeros((8, 8), dtype=int)
    # board[3, 3] = 1
    # board[4, 4] = 2
    # board[5, 5] = 1
    # board[6, 6] = 2
    # assert get_score(board, 1) == 0
    # assert get_score(board, 2) == 0
    
    # # Test 6: Board with five stones
    # board = np.zeros((8, 8), dtype=int)
    # board[3, 3] = 1
    # board[4, 4] = 2
    # board[5, 5] = 1
    # board[6, 6] = 2
    # board[7, 7] = 1
    # assert get_score(board, 1) == 32
    # assert get_score(board, 2) == -32
    
    # # Test 7: Board with six stones
    # board = np.zeros((8, 8), dtype=int)
    # board[3, 3] = 1
    # board[4, 4] = 2
    # board[5, 5] = 1
    # board[6, 6] = 2
    # board[7, 7] = 1
    # board[2, 2] = 2
    # assert get_score(board, 1) == 0
    # assert get_score(board, 2) == 0
    


def test_isValidPos():
    assert isValidPos(0, 0) == False
    assert isValidPos(1, 1) == True
    assert isValidPos(7, 7) == True
    assert isValidPos(8, 8) == True

def test_make_move():
    board = np.zeros((Defines.GRID_NUM, Defines.GRID_NUM), dtype=int)
    move = StoneMove([StonePosition(1, 1), StonePosition(1, 2)])
    make_move(board, move, Defines.BLACK)
    assert board[1, 1] == Defines.BLACK
    assert board[1, 2] == Defines.BLACK

def test_unmake_move():
    board = np.zeros((Defines.GRID_NUM, Defines.GRID_NUM), dtype=int)
    move = StoneMove()
    move[0], move[1] = StonePosition(1, 1), StonePosition(1, 2)
    make_move(board, move, Defines.BLACK)
    unmake_move(board, move)
    assert board[1, 1] == Defines.NOSTONE
    assert board[1, 2] == Defines.NOSTONE

def test_is_win_by_premove():
    board = np.zeros((Defines.GRID_NUM, Defines.GRID_NUM), dtype=int)
    preMove = StoneMove()
    preMove[0], preMove[1] = StonePosition(1, 1), StonePosition(1, 2)
    
    make_move(board, preMove, Defines.BLACK)
    assert is_win_by_premove(board, preMove) == False

def test_init_board():
    board = np.zeros((Defines.GRID_NUM, Defines.GRID_NUM), dtype=int)
    init_board(board)
    print(board)
    assert board[0, 0] == Defines.BORDER
    assert board[0, 1] == Defines.BORDER
    assert board[1, 0] == Defines.BORDER
    assert board[1, 1] == Defines.NOSTONE


test_init_board()
test_isValidPos()
test_make_move()
test_unmake_move()
test_is_win_by_premove()
test_get_score()
test_ge