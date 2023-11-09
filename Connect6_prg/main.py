from game_engine import GameEngine
import sys
import argparse

def main():
    parser = argparse.ArgumentParser(description='Start the game engine with a specified algorithm and chromosome.')
    
    parser.add_argument('--algorithm', type=str, default='MiniMaxAlphaBeta', 
                        help='Specify the algorithm to use. Default is MiniMaxAlphaBeta.')
    
    parser.add_argument('--chromosome', type=str, default=None, 
                        help='Specify the chromosome as a comma-separated list of numbers.')
    
    args = parser.parse_args()
    
    # Convert the chromosome argument from comma-separated string to a list of integers
    if args.chromosome:
        args.chromosome = [int(value) for value in args.chromosome.split(',')]
    
    gameEngine = GameEngine(args.algorithm)
    gameEngine.run()
    gameEngine.run()

if __name__ == "__main__":
    main()