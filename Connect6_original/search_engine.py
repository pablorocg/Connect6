from tools import *

class SearchEngine():
    """
    Implement a search engine for Connect6 using minimax with alpha-beta pruning.
    Using numpy array for the board representation and operations.
    """
    def __init__(self):
        self.state = None
        self.player = None
        self.depth = None
        self.m_total_nodes = 0

    def before_search(self, state, player, depth):
        self.state = state
        self.player = player
        self.depth = depth
    
    def find_possible_move(self):
        for i in range(1,len(self.state)-1):
            for j in range(1, len(self.state[i])-1):
                if(self.state[i][j] == Defines.NOSTONE):
                    return (i,j)
        return (-1,-1)
    
    def alpha_beta_search(self, depth, alpha, beta, ourColor, bestMove, preMove):
            print('score_pablo: {}'.format(get_score(self.state, self.player)))
            #Check game result
            if (is_win_by_premove(self.state, preMove)):
                if (ourColor == self.m_chess_type):
                    #Opponent wins.
                    return 0;
                else:
                    #Self wins.
                    return Defines.MININT + 1;
            
            alpha = 0
            if(self.check_first_move()):
                bestMove.positions[0].x = 10
                bestMove.positions[0].y = 10
                bestMove.positions[1].x = 10
                bestMove.positions[1].y = 10
            else:   
                move1 = self.find_possible_move()
                bestMove.positions[0].x = move1[0]
                bestMove.positions[0].y = move1[1]
                bestMove.positions[1].x = move1[0]
                bestMove.positions[1].y = move1[1]
                make_move(self.state,bestMove,ourColor)
                
                '''#Check game result
                if (is_win_by_premove(self.m_board, bestMove)):
                    #Self wins.
                    return Defines.MININT + 1;'''
                
                move2 = self.find_possible_move()
                bestMove.positions[1].x = move2[0]
                bestMove.positions[1].y = move2[1]
                make_move(self.state,bestMove,ourColor)

            return alpha
    
    def check_first_move(self):
        for i in range(1,len(self.state)-1):
            for j in range(1, len(self.state[i])-1):
                if(self.state[i][j] != Defines.NOSTONE):
                    return False
        return True


def flush_output():
    import sys
    sys.stdout.flush()