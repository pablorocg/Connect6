import numpy as np
from tools import *
from defines import *


def test_init_board():
    
    board = init_board(np.zeros((Defines.GRID_NUM, Defines.GRID_NUM))) 
    
    assert board[0, 0] == Defines.BORDER
    assert board[0, -1] == Defines.BORDER
    assert board[-1, 0] == Defines.BORDER
    assert board[-1, -1] == Defines.BORDER
    assert np.all(board[1:-1, 1:-1] == Defines.NOSTONE)
    
    print("test_init_board passed")

def test_create_move():
    
    move = create_move(((1,2),(3,4)))
    assert move.positions[0].x == 1
    assert move.positions[0].y == 2
    assert move.positions[1].x == 3
    assert move.positions[1].y == 4
    print("test_create_move passed")

def test_make_move():
    board = init_board(np.zeros((Defines.GRID_NUM, Defines.GRID_NUM)))
    
    move = create_move(((1,2),(3,4)))
    make_move(board, move, Defines.BLACK)
    
    move = create_move(((1,3),(2,4)))
    make_move(board, move, Defines.WHITE)


    assert board[1,3] == Defines.WHITE
    assert board[2,4] == Defines.WHITE
    assert board[1,2] == Defines.BLACK
    assert board[3,4] == Defines.BLACK
    print("test_make_move passed")

def test_check_winner():
    board = init_board(np.zeros((Defines.GRID_NUM, Defines.GRID_NUM)))
    
    for i in range(1,6):  
        move = create_move(((i, i),(i+1, i+1)))
        make_move(board, move, Defines.BLACK)
    
    assert check_winner(board) == Defines.BLACK
    print("test_check_winner passed")

def test_move2msg():
    move = create_move(((1,2),(3,4)))
    msg = move2msg(move)
    assert msg == "BSDQ"
    print("test_move2msg passed")

def test_msg2move():
    move = msg2move("AABB")
    assert move.positions[0].x == 19
    assert move.positions[0].y == 1
    assert move.positions[1].x == 18
    assert move.positions[1].y == 2
    print("test_msg2move passed")

def test_get_available_moves():
    """ 
    n!/(n-r)!r! donde r = 2 y n = n huecos libres al cuadrado
    """

    board = init_board(np.zeros((Defines.GRID_NUM, Defines.GRID_NUM)))
    
    move_list = get_available_moves(board)
    # [print(move.positions[0].x, move.positions[0].y, move.positions[1].x, move.positions[1].y) for move in move_list]
    assert len(move_list) == 64980
    
    make_move(board, move_list[0], Defines.BLACK)
    move_list = get_available_moves(board)
    assert len(move_list) == 64261
    
    print("test_get_available_moves passed")





test_init_board()
test_create_move()
test_make_move()
test_check_winner()
test_move2msg()
test_msg2move()
test_get_available_moves()