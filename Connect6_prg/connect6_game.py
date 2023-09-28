import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

class Connect6:
    """
    Clase que representa el entorno del juego de Conecta 6
    """

    def __init__(self):
        """
        Connect (m, n, k, p, q)
        - m: número de filas del tablero
        - n: número de columnas del tablero
        - k: número de fichas en línea para ganar
        - p: número de fichas que se pueden colocar en un turno
        - q: número de fichas que se pueden mover en el primer turno


        Inicializa el objeto Connect6 con los siguientes atributos:
        - row_count: número de filas del tablero
        - column_count: número de columnas del tablero
        - win_condition: número de fichas en línea para ganar
        - action_size: número de acciones posibles
        - turno: número de turnos transcurridos
        
        """
        self.row_count = 19
        self.column_count = 19
        self.win_condition = 6

        self.action_size = self.row_count * self.column_count
        self.turno = 0

    def __repr__(self):
        """
        Devuelve una representación en cadena del objeto Connect6
        """
        return "Connect6"

    def get_initial_state(self):
        """
        Devuelve el estado inicial del juego (tablero vacío)
        """
        return np.zeros((self.row_count, self.column_count))
    
    def get_next_state(self, state, action, player):
        """
        Devuelve el estado resultante de aplicar una acción a un estado
        """
        row = action // self.column_count
        column = action % self.column_count
        state[row][column] = player
        return state
    
    def get_valid_moves(self, state):
        """
        Devuelve un vector de 0s y 1s indicando las acciones válidas
        """
        return (state.reshape(-1) == 0).astype(np.uint8)
    
    def check_win(self, state, action):
        """
        Comprueba si un jugador ha ganado el juego (6 en línea)
        """
        if action == None:
            return False

        row = action // self.column_count
        column = action % self.column_count
        
        # Simular la acción en el tablero
        sim_state = state.copy()
        sim_state[row][column] = 1 

        # Comprobar si hay 6 en línea
        # Comprobar filas
        for i in range(self.row_count):
            for j in range(self.column_count - self.win_condition + 1):
                if np.all(sim_state[i][j:j+self.win_condition] == 1):
                    return True
                
        # Comprobar columnas
        for i in range(self.row_count - self.win_condition + 1):
            for j in range(self.column_count):
                if np.all(sim_state[i:i+self.win_condition, j] == 1):
                    return True
        
        # Comprobar todas las diagonales
        for i in range(self.row_count - self.win_condition + 1):
            for j in range(self.column_count - self.win_condition + 1):
                if np.all(np.diag(sim_state[i:i+self.win_condition, j:j+self.win_condition]) == 1):
                    return True
                if np.all(np.diag(np.fliplr(sim_state[i:i+self.win_condition, j:j+self.win_condition])) == 1):
                    return True
        return False
    
    def get_value_and_terminated(self, state, action):
        """
        Devuelve el valor y si el juego ha terminado
        """
        if self.check_win(state, action):# game is terminated and win
            return 1, True
        elif np.sum(state == 0) == 0:# game is terminated but draw
            return 0, True
        else:# game is not terminated
            return 0, False
        
    def get_opponent(self, player):
        """
        Devuelve el jugador contrario
        """
        return -player
    
    def get_opponent_value(self, value):
        """
        Devuelve el valor contrario
        """
        return -value
    
    def change_perspective(self, state, player):
        """
        Cambia la perspectiva del tablero, es decir, poner jugador a 1 y oponente a -1
        """
        return state * player
    
    def get_encoded_state(self, state):
        """
        Codificación de la matriz como un vector de 3 canales en el que cada canal representa:
        - Canal 0: casillas ocupadas por el jugador -1
        - Canal 1: casillas no ocupadas por ningún jugador
        - Canal 2: casillas ocupadas por el jugador 1
        """
        encoded_state = np.stack(
            (state ==-1, state == 0, state == 1)
        ).astype(np.float32)
        
        return encoded_state

    def show_state(self, state):
        """
        Muestra el estado del juego
        """

        fontdict = {'fontsize': 10,
                    'fontweight' : 50}
        # Crear un colormap personalizado en el que el negro = 1, blanco = -1 y naranja = 0.
        # Define los colores
        colors = [(1, 1, 1), (1, 0.8, 0.4), (0, 0, 0)]

        # Define los puntos de referencia y crea el colormap
        n_bins = 3  # Número de colores en el colormap
        cmap_name = "custom_colormap"
        cm = mcolors.LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bins)

        # Normaliza los valores para que estén en el rango [0, 1]
        norm = mcolors.Normalize(vmin=-1, vmax=1)

        annot = np.arange(0, self.action_size)
        annot = annot.reshape(self.row_count, self.column_count)

        plt.figure(figsize=(6,6))
        # Mostrar movimientos sobre el tablero
        plt.imshow(state, cmap=cm, norm=norm) 
        
        # Mostrar anotaciones sobre la imagen
        for i in range(self.row_count):
            for j in range(self.column_count):
                plt.text(j, i, annot[i][j], ha="center", va="center", color="green", fontdict=fontdict)
 
        plt.axis('off')
        plt.tight_layout()
        plt.show()







        
# --------------------------------------------------------------------------------------------------------------------
    def gaussian(self, x, mu, sigma):
        coefficient = 1 / (np.sqrt(2 * np.pi * sigma**2))
        exponent = -((x - mu)**2) / (2 * sigma**2)
        return coefficient * np.exp(exponent)

    def get_gaussian_grid(self):
        """
        Devuelve una distribución gaussiana bidimensional del tamaño del tablero de juego
        """

        valores = np.arange(self.column_count)

        # Definir la media y la desviación estándar
        mu = valores.mean()
        sigma = valores.std() - 4.75

        grid = np.zeros((self.row_count, self.column_count))
        for i in range(self.row_count):
            for j in range(self.column_count):
                grid[i, j] = self.gaussian(x = i, mu = mu, sigma = sigma) * self.gaussian(x = j, mu = mu, sigma = sigma)

        # Normalizar el grid entre 0 y 1
        grid = (grid - grid.min()) / (grid.max() - grid.min())
        
        return grid

    def simulate_game(self):
        """
        Simula una partida aleatoria en la que comienza el j1 (negras) colocando 1 sola ficha y a partir de ahi ambos jugadores colocan 2 fichas por turno hasta que uno de los dos gana.
        """
        player = 1
        winner = 0
        state = self.get_initial_state()
        
        while True:
            print('Turno: ', self.turno, 'Jugador: ', player)
            if self.turno == 0:
                
                # Escoger accion para el 1º turno teniendo en cuenta las probabilidades de la distribucion gaussiana
                valid_moves = self.get_valid_moves(state)
                prob_grid = self.get_gaussian_grid()

                prob_grid = prob_grid.reshape((self.action_size))
                prob_grid = prob_grid * valid_moves
                prob_grid = prob_grid / prob_grid.sum()
                action = np.random.choice(self.action_size, p = prob_grid)
                print('Action: ', action)
                #movimiento
                state = self.get_next_state(state, action, player)
                #fin de turno
                player = self.get_opponent(player)
                self.turno += 1
                
            else:# turno > 0
                
                # mov i (1º o 2º) del turno

                # seleccionar accion aleatoria
                valid_moves = self.get_valid_moves(state)
                action = np.random.choice(np.where(valid_moves == 1)[0])
                print('     Action 1: {}'.format(action))
                
                # Comprobar si la accion es ganadora
                if self.check_win(state, action):
                    print('Movimiento ganador: ', action)
                    state = self.get_next_state(state, action, player)
                    winner = player
            
                else: # si no es ganadora, aplicar accion y continuar turno
                    state = self.get_next_state(state, action, player)
            
                
                    # fin de turno
                    self.turno += 1
                    player = self.get_opponent(player)
            
            if winner != 0:
                break
        return state, winner, action
        
                
        
               
                
                

                
    

    def detect_threats(self, state, player):
        """
        Algorithm to count the number of threats in one line of the board in the game of Connect6.
        """
        num_threats = 0
        sim_state = state.copy()
        threat_map = np.zeros((self.row_count, self.column_count))#Score each position on the board
        threat_positions = []

        # HORIZONTAL THREATS

        # Paso 1: For each row, slide a window of 6 positions from the left to the right.
        for row in range(0, self.row_count):
            
            if np.any(sim_state[row] != 0):# Comprobar si la fila (row) esta compuesta exclusivamente por ceros, si no, no tiene sentido comprobarla
                
                for column in range(self.column_count - self.win_condition + 1):

                    window_direct = state[row, column:column+self.win_condition]
                    window_reverse = np.flip(window_direct)

                    # comprobar si la ventana (window_direct) esta compuesta exclusivamente por ceros
                    if np.any(window_direct != 0):
                        # Comenzar a comprobar si hay amenazas
                        print('Row: ', row)
                        print('Window dir: ', window_direct)
                        print('Window rev: ', window_reverse)

                        # Contar los ceros que hay en la ventana (window_direct)
                        num_player = np.count_nonzero(window_direct == player)
                        num_zeros = np.count_nonzero(window_direct == 0)
                        num_opponent = np.count_nonzero(window_direct == -player)
                        
                        # player threats
                        if num_player >=4 and num_zeros >= 1:
                            
                            player_threats += 1
                            threat_positions.append((row, column + np.where(window_direct == 0)[0][0]))
                            
                            # Marcar la posicion vacia del tablero donde se encuentra la amenaza



                        
                        print('Num player: ', num_player)
                        print('Num zeros: ', num_zeros)
                        print('Num opponent: ', num_opponent)

                        # comprobar si la ventana (window_direct) contiene al menos 4 fichas seguidas de un 0
        print('Player threat: ', threat_positions)
        return num_threats, threat_map

    

# Paso 1: For a line, slide a window of 6 positions from the left to the right.
# Paso 2: Repeat the following step for each sliding window.
# Paso 3: If the window contains neither white stones (-1) nor marked squares (0) and at least four black stones (1), add one more threat and mark the rightmost empty square. The window satisfying the condition is called a threat window.


        
        


# # Crear un estado de prueba
# player = 1 # Black player
# c6 = Connect6()
# # state, winner, action = c6.simulate_game()

# # # Mostrar el estado del tablero
# # print('El ganador es: ', winner)
# # print('Ultima accion: ', action)
# # c6.show_state(state)

# state = c6.get_initial_state()

# # Colocar algunas fichas en el tablero
# for i in range(4):
#     state = c6.get_next_state(state, 180 + i, 1)
#     state = c6.get_next_state(state, 190 + i*i, 1)
# # state = c6.get_next_state(state, , 1)
# # state = c6.get_next_state(state, , 1)


# # Calcular el valor heuristico del estado
# num_threats, threat_map = c6.detect_threats(state, 1)
# print('Numero de amenazas: ', num_threats)

# # # Mostrar el estado del tablero
# c6.show_state(state)

# c6.show_state(threat_map)


    

