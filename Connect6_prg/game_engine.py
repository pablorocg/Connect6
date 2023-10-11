from defines import *
from tools import *
import sys
# from search_engine import SearchEngine
from search_algorithms import *
import time

class GameEngine:
    def __init__(self, name=Defines.ENGINE_NAME):
        if name and len(name) > 0:
            if len(name) < Defines.MSG_LENGTH:
                self.m_engine_name = name
            else:
                print(f"Too long Engine Name: {name}, should be less than: {Defines.MSG_LENGTH}")
        self.m_alphabeta_depth = 2
        self.m_board = t = np.zeros((Defines.GRID_NUM, Defines.GRID_NUM))
        self.init_game()
        
        if name == "MiniMax":
            self.m_search_engine = MiniMax()
            self.m_alphabeta_depth = 1

        elif name == "MiniMaxParalelizado":
            self.m_search_engine = MiniMaxParalelizado()
            self.m_alphabeta_depth = 1

        elif name == "MiniMaxAlphaBeta":
            self.m_search_engine = MiniMaxAlphaBeta()
            self.m_alphabeta_depth = 6

        elif name == "NegaMaxAlphaBeta":
            self.m_search_engine = NegaMaxAlphaBeta()

        
        
        
        
        
        
        self.m_best_move = StoneMove()

    def init_game(self):
        init_board(self.m_board)

    def on_help(self):
        print(
            f"On help for GameEngine {self.m_engine_name}\n"
            " name        - print the name of the Game Engine.\n"
            " print       - print the board.\n"
            " exit/quit   - quit the game.\n"
            " black XXXX  - place the black stone on the position XXXX on the board.\n"
            " white XXXX  - place the white stone on the position XXXX on the board, X is from A to S.\n"
            " next        - the engine will search the move for the next step.\n"
            " move XXXX   - tell the engine that the opponent made the move XXXX,\n"
            "              and the engine will search the move for the next step.\n"
            " new black   - start a new game and set the engine to black player.\n"
            " new white   - start a new game and set it to white.\n"
            " depth d     - set the alpha beta search depth, default is 6.\n"
            " vcf         - set vcf search.\n"
            " unvcf       - set none vcf search.\n"
            " help        - print this help.\n")

    def run(self):
        msg = ""
        self.on_help()
        while True:
            msg = input().strip()
            log_to_file(msg)
            if msg == "name":
                print(f"name {self.m_engine_name}")
            elif msg == "exit" or msg == "quit":
                break
            elif msg == "print":
                # print_board(self.m_board, self.m_best_move)
                show_m_board(self.m_board)
                print(f"Best move: {move2msg(self.m_best_move)}")
                print(f"Chess type: {self.m_chess_type}")
            elif msg == "vcf":
                self.m_vcf = True
            elif msg == "unvcf":
                self.m_vcf = False
            elif msg.startswith("black"):
                self.m_best_move = msg2move(msg[6:])
                make_move(self.m_board, self.m_best_move, Defines.BLACK)
                self.m_chess_type = Defines.BLACK
            elif msg.startswith("white"):
                self.m_best_move = msg2move(msg[6:])
                make_move(self.m_board, self.m_best_move, Defines.WHITE)
                self.m_chess_type = Defines.WHITE
            elif msg == "next":
                self.m_chess_type = 3 - self.m_chess_type# ^ 3
                if self.search_a_move(self.m_chess_type, self.m_best_move):
                    make_move(self.m_board, self.m_best_move, self.m_chess_type)
                    msg = f"move {move2msg(self.m_best_move)}"
                    print(msg)
                    flush_output()
            elif msg.startswith("new"):# Comando para iniciar una nueva partida

                self.init_game()# Inicializa el tablero

                if msg[4:] == "black":# Si el comando es new black
                    self.m_best_move = msg2move("JJ")# Se asigna la mejor jugada
                    make_move(self.m_board, self.m_best_move, Defines.BLACK)# Se hace la jugada
                    self.m_chess_type = Defines.BLACK# Se asigna el color de la ficha
                    msg = "move JJ"
                    print(msg)
                    flush_output()# Se imprime la jugada

                else:
                    
                    self.m_chess_type = Defines.WHITE # Si el comando es new white

            elif msg.startswith("move"):# Comando para hacer una jugada

                self.m_best_move = msg2move(msg[5:])# Se asigna la mejor jugada

                make_move(self.m_board, self.m_best_move, self.m_chess_type)# Se hace la jugada

                if is_win_by_premove(self.m_board, self.m_best_move):# Si la jugada es ganadora
                    print("We lost!")

                if self.search_a_move(self.m_chess_type, self.m_best_move):# Se busca la mejor jugada
                    msg = f"move {move2msg(self.m_best_move)}"# Se asigna el mensaje
                    make_move(self.m_board, self.m_best_move, self.m_chess_type)# Se hace la jugada
                    print(msg)
                    flush_output()
            elif msg.startswith("depth"):
                d = int(msg[6:])
                if 0 < d < 10:
                    self.m_alphabeta_depth = d
                print(f"Set the search depth to {self.m_alphabeta_depth}.\n")
            elif msg == "help":
                self.on_help()
        return 0

    def search_a_move(self, ourColor, bestMove):
        score = 0
        start = 0
        end = 0

        start = time.perf_counter()
        self.m_search_engine.before_search(self.m_board, self.m_chess_type, self.m_alphabeta_depth)
        score, bestMove = self.m_search_engine.get_best_move()
        self.m_best_move = bestMove
        # score = self.m_search_engine.alpha_beta_search(self.m_alphabeta_depth, Defines.MININT, Defines.MAXINT, ourColor, bestMove, bestMove)
        
        end = time.perf_counter()

        print(f"AB Time:\t{end - start:.3f}")
        # print(f"Node:\t{self.m_search_engine.m_total_nodes}\n")
        print(f"Score:\t{score:.3f}")
        return True

def flush_output():
    sys.stdout.flush()

# Create an instance of GameEngine and run the game
if __name__ == "__main__":
    game_engine = GameEngine("MiniMaxAlphaBeta")
    game_engine.run()