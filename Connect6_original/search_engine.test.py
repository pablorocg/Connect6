import numpy as np
from search_engine import *



def test_get_score():
    # Crea un array de 21x21 con 3s 
    board = np.full((21, 21), 3)
    board[1:20, 1:20] = 0
    assert get_score(board, 1) == 0
    
test_get_score()