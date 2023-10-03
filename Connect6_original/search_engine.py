from tools import *

class SearchEngine():
    def __init__(self):
        self.m_board = None
        self.m_chess_type = None
        self.m_alphabeta_depth = None
        self.m_total_nodes = 0

    def before_search(self, board, color, alphabeta_depth):
        self.m_board = [row[:] for row in board]
        self.m_chess_type = color
        self.m_alphabeta_depth = alphabeta_depth
        self.m_total_nodes = 0

    # def alpha_beta_search(self, depth, alpha, beta, ourColor, bestMove, preMove):
    
    #     #Check game result
    #     if (is_win_by_premove(self.m_board, preMove)):
    #         if (ourColor == self.m_chess_type):
    #             #Opponent wins.
    #             return 0;
    #         else:
    #             #Self wins.
    #             return Defines.MININT + 1;
        
    #     alpha = 0
    #     if self.check_first_move():
    #         bestMove.positions[0].x = 10
    #         bestMove.positions[0].y = 10
    #         bestMove.positions[1].x = 10
    #         bestMove.positions[1].y = 10
    #     else:   
    #         move1 = self.find_possible_move()
    #         bestMove.positions[0].x = move1[0]
    #         bestMove.positions[0].y = move1[1]
    #         bestMove.positions[1].x = move1[0]
    #         bestMove.positions[1].y = move1[1]
    #         make_move(self.m_board,bestMove,ourColor)
            
    #         '''#Check game result
    #         if (is_win_by_premove(self.m_board, bestMove)):
    #             #Self wins.
    #             return Defines.MININT + 1;'''
            
    #         move2 = self.find_possible_move()
    #         bestMove.positions[1].x = move2[0]
    #         bestMove.positions[1].y = move2[1]
    #         make_move(self.m_board,bestMove,ourColor)

    #     return alpha
    def evaluate_and_sort_moves(self, moves, ourColor):
        """
        Evaluate and sort the moves based on their quality.
        """
        # Evaluate each move
        move_scores = []
        for move in moves:
            hypothetical_move = StoneMove()
            hypothetical_move.positions[0].x = move.x
            hypothetical_move.positions[0].y = move.y
            hypothetical_move.positions[1].x = move.x
            hypothetical_move.positions[1].y = move.y
            make_move(self.m_board, hypothetical_move, ourColor)
            
            score = get_score(self.m_board, ourColor)
            move_scores.append((move, score))
            
            unmake_move(self.m_board, hypothetical_move)

        # Sort moves based on their scores
        move_scores.sort(key=lambda x: x[1], reverse=(ourColor == self.m_chess_type))

        # Return sorted moves
        return [move[0] for move in move_scores]


    def alpha_beta_search(self, depth, alpha, beta, ourColor, bestMove, preMove):
    
        # Check if the game has been won with the previous move
        if is_win_by_premove(self.m_board, preMove):
            if ourColor == self.m_chess_type:
                # Opponent wins
                return 0
            else:
                # Self wins
                return Defines.MININT + 1
                
        # Base case: if depth is 0, evaluate the board
        if depth == 0:
            return get_score(self.m_board, ourColor)

        # Get all possible moves and sort them based on their quality
        possible_moves = [move for move in self.generate_moves(self.m_board)]
        if not possible_moves:
            return get_score(self.m_board, ourColor)
        
        sorted_moves = self.evaluate_and_sort_moves(possible_moves, ourColor)

        if ourColor == self.m_chess_type:  # If it's our turn (Maximizing player)
            max_eval = Defines.MININT
            for move in sorted_moves:
                # Make a hypothetical move
                hypothetical_move = StoneMove()
                hypothetical_move.positions[0].x = move.x
                hypothetical_move.positions[0].y = move.y
                hypothetical_move.positions[1].x = move.x
                hypothetical_move.positions[1].y = move.y
                make_move(self.m_board, hypothetical_move, ourColor)
                
                # Recurse with reduced depth
                eval = self.alpha_beta_search(depth - 1, alpha, beta, 1 if ourColor == 2 else 2, bestMove, hypothetical_move)
                
                # Undo the hypothetical move
                unmake_move(self.m_board, hypothetical_move)
                
                # Update best move if needed
                if eval > max_eval:
                    max_eval = eval
                    bestMove.positions[0].x = move.x
                    bestMove.positions[0].y = move.y
                    bestMove.positions[1].x = move.x
                    bestMove.positions[1].y = move.y
                
                # Alpha-beta pruning
                alpha = max(alpha, eval)
                if beta <= alpha:
                    break

            return max_eval

        else:  # If it's opponent's turn (Minimizing player)
            min_eval = Defines.MAXINT
            for move in sorted_moves:
                # Make a hypothetical move
                hypothetical_move = StoneMove()
                hypothetical_move.positions[0].x = move[0]
                hypothetical_move.positions[0].y = move[1]
                hypothetical_move.positions[1].x = move[0]
                hypothetical_move.positions[1].y = move[1]
                make_move(self.m_board, hypothetical_move, ourColor)
                
                # Recurse with reduced depth
                eval = self.alpha_beta_search(depth - 1, alpha, beta, self.m_chess_type, bestMove, hypothetical_move)
                
                # Undo the hypothetical move
                unmake_move(self.m_board, hypothetical_move)
                
                # Alpha-beta pruning
                beta = min(beta, eval)
                if beta <= alpha:
                    break

            return min_eval


        
    def check_first_move(self):
        for i in range(1,len(self.m_board)-1):
            for j in range(1, len(self.m_board[i])-1):
                if(self.m_board[i][j] != Defines.NOSTONE):
                    return False
        return True
        
    # def find_possible_move(self):
    #     for i in range(1,len(self.m_board)-1):
    #         for j in range(1, len(self.m_board[i])-1):
    #             if(self.m_board[i][j] == Defines.NOSTONE):
    #                 return (i,j)
    #     return (-1,-1)
    
    def generate_moves(self, board):
        # Generate all possible moves for the given player on the board.
        # For simplicity, this function will return all empty positions as possible moves.
        moves = []
        for i in range(1, Defines.GRID_NUM - 1):  # Exclude the border
            for j in range(1, Defines.GRID_NUM - 1):  # Exclude the border
                if board[i][j] == Defines.NOSTONE:
                    moves.append(StonePosition(i, j))
        return moves
    
    



def flush_output():
    import sys
    sys.stdout.flush()