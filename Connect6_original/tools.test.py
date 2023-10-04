import numpy as np
from defines import *
from tools import *

def test_get_window_scoring():
    window1 = np.array([1, 0, 0, 0, 0, 0])
    window2 = np.array([1, 1, 0, 0, 0, 0])
    window3 = np.array([1, 1, 0, 1, 0, 0])
    window4 = np.array([1, 1, 0, 0, 1, 1])
    window5 = np.array([1, 1, 1, 1, 1, 0])
    window6 = np.array([1, 1, 1, 1, 1, 1])
    
    assert get_window_scoring(window1, 1) == 1
    assert get_window_scoring(window1, 2) == 0

    assert get_window_scoring(window2, 1) == 4
    assert get_window_scoring(window2, 2) == 0

    assert get_window_scoring(window3, 1) == 27
    assert get_window_scoring(window3, 2) == 0

    assert get_window_scoring(window4, 1) == 256
    assert get_window_scoring(window4, 2) == 0

    assert get_window_scoring(window5, 1) == 3125
    assert get_window_scoring(window5, 2) == 0

    assert get_window_scoring(window6, 1) == 46656
    assert get_window_scoring(window6, 2) == 0

    print("Test get_window_scoring passed")


def test_isValidPos():
    assert isValidPos(0, 0) == False
    assert isValidPos(1, 1) == True
    assert isValidPos(7, 7) == True
    assert isValidPos(8, 8) == True
    
    print("Test isValidPos passed")

def test_make_move():
    board = np.zeros((Defines.GRID_NUM, Defines.GRID_NUM), dtype=int)
    move = StoneMove()
    move.positions[0], move.positions[1] = StonePosition(1, 1), StonePosition(1, 2)
    make_move(board, move, Defines.BLACK)
    assert board[1, 1] == Defines.BLACK
    assert board[1, 2] == Defines.BLACK
    
    print("Test make_move passed")

def test_unmake_move():
    board = np.zeros((Defines.GRID_NUM, Defines.GRID_NUM), dtype=int)
    move = StoneMove()
    move.positions[0], move.positions[1] = StonePosition(1, 1), StonePosition(1, 2)
    make_move(board, move, Defines.BLACK)
    unmake_move(board, move)
    
    assert board[1, 1] == Defines.NOSTONE
    assert board[1, 2] == Defines.NOSTONE
    
    print("Test unmake_move passed")

def test_is_win_by_premove():
    board = np.zeros((Defines.GRID_NUM, Defines.GRID_NUM), dtype=int)
    preMove = StoneMove()
    preMove.positions[0], preMove.positions[1] = StonePosition(1, 1), StonePosition(1, 2)
    
    make_move(board, preMove, Defines.BLACK)
    assert is_win_by_premove(board, preMove) == False
    print("Test is_win_by_premove passed")

def test_init_board():
    board = np.zeros((Defines.GRID_NUM, Defines.GRID_NUM), dtype=int)
    init_board(board)
    print(board)
    assert board[0, 0] == Defines.BORDER
    assert board[0, 1] == Defines.BORDER
    assert board[1, 0] == Defines.BORDER
    assert board[1, 1] == Defines.NOSTONE
    print("Test init_board passed")


def mide_tiempo(funcion):
    def funcion_medida(*args, **kwargs):
        inicio = time.time()
        c = funcion(*args, **kwargs)
        print(time.time() - inicio)
        return c
    print("Test tools.py")
    return funcion_medida
    

test_init_board()
test_isValidPos()
test_make_move()
test_unmake_move()
test_is_win_by_premove()
test_get_window_scoring()

