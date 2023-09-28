import numpy as np

class TicTacToe:
    def __init__(self):
        self.row_count = 3
        self.column_count = 3
        self.action_size = self.row_count * self.column_count

    def __repr__(self):
        return "TicTacToe"

    def get_initial_state(self):
        return np.zeros((self.row_count, self.column_count))
    
    def get_next_state(self, state, action, player):
        row = action // self.column_count
        column = action % self.column_count
        state[row][column] = player
        return state
    
    def get_valid_moves(self, state):
        return (state.reshape(-1) == 0).astype(np.uint8)
    
    def check_win(self, state, action):

        if action == None:
            return False

        row = action // self.column_count
        column = action % self.column_count
        player = state[row][column]
        
        return (
            np.sum(state[row, :]) == player * self.column_count 
            or np.sum(state[:, column]) == player * self.row_count
            or np.sum(np.diag(state)) == player * self.row_count
            or np.sum(np.diag(np.flip(state, axis=0))) == player * self.row_count
        )
    
    def get_value_and_terminated(self, state, action):
        if self.check_win(state, action):# game is terminated and win
            return 1, True
        elif np.sum(state == 0) == 0:# game is terminated but draw
            return 0, True
        else:# game is not terminated
            return 0, False
        
    def get_opponent(self, player):
        return -player
    
    def get_opponent_value(self, value):
        return -value
    
    def change_perspective(self, state, player):
        return state * player
    
    def get_encoded_state(self, state):
        """
        Codificacion de la matriz como un vector de 3 canales en el que cada canal representa
            Canal 0: Casillas ocupadas por el jugador -1
            Canal 1: Casillas no ocupadas por ningun jugador
            Canal 2: Casillas ocupadas por el jugador 1
        """
        encoded_state = np.stack(
            (state ==-1, state == 0, state == 1)
        ).astype(np.float32)
        
        return encoded_state
    
    def get_heuristic_value(self, state, action, player):
        """
        Devuelve el valor heuristico de un estado teniendo en cuenta el numero de amenazas y posibles victorias

        """
        row = action // self.column_count
        column = action % self.column_count
        state[row][column] = player

        # Comprobar filas y columnas
        value = 0
        for i in range(self.row_count):
            if np.sum(state[i, :]) == player * self.column_count:
                value += 1
            if np.sum(state[:, i]) == player * self.row_count:
                value += 1

        # Comprobar todas las diagonales
        if np.sum(np.diag(state)) == player * self.row_count:
            value += 1
        if np.sum(np.diag(np.fliplr(state))) == player * self.row_count:
            value += 1

        return value
    