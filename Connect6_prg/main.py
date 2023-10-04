from tictactoe_game import TicTacToe
from connect6_game import Connect6
from mcts_algorithm import MCTS, AlphaMCTS
import numpy as np


tictactoe = Connect6()#TicTacToe()
player = 1

args = {
    'C': 1.41,
    'num_searches': 300,
}

mcts = AlphaMCTS(tictactoe, args)



state = tictactoe.get_initial_state()
while True:
    print(state)
    tictactoe.show_state(state)
    
    if player == 1:
        
            valid_moves = tictactoe.get_valid_moves(state)
            print('valid moves: ', [i for i in range(tictactoe.action_size) if valid_moves[i] == 1])
            action = int(input(f"player {player}'s turn: "))
            
            if valid_moves[action] == 0:
                print('invalid move')
                continue
    else:

        neutral_state = tictactoe.change_perspective(state, player)
        mcts_probs = mcts.search(neutral_state)
        action = np.argmax(mcts_probs)
    # update state
    state = tictactoe.get_next_state(state, action, player)
    
    # check win
    value, is_terminal = tictactoe.get_value_and_terminated(state, action)
    
    #
    if is_terminal:
        print(state)
        if value == 1:
            print(f'player {player} win')
        else:
            print('draw')
        break
        
    # switch player
    player = tictactoe.get_opponent(player)