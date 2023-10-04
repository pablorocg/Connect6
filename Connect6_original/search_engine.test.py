import numpy as np
from search_engine import *

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
    # Crea un array de 21x21 con 3s 
    board = np.full((21, 21), 3)
    board[1:20, 1:20] = 0
    assert get_score(board, 1) == 0
    
test_get_score()