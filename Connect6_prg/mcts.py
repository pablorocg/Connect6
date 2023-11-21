import tools as tl
import random
import math
import numpy as np

 
import tools as tl  # Asegúrate de que 'tools' tenga las funciones necesarias para Connect6
import random
import math
import numpy as np

class Nodo:
    def __init__(self, estado, padre=None, movimiento=None, color=None):
        self.estado = estado  # Tablero de juego
        self.padre = padre  # Nodo padre
        self.hijos = []  # Lista de nodos hijos
        self.visitas = 0  # Número de visitas
        self.victorias = 0  # Número de victorias
        self.movimiento = movimiento  # Movimiento que llevó a este nodo
        self.movimientos_inexplorados = tl.get_available_moves(estado)  # Movimientos que no han sido explorados
        self.color = color  # Color del jugador que hizo el movimiento

    def es_terminal(self):
        return tl.check_winner(self.estado) != 0

    def es_completamente_expandido(self):
        return len(self.movimientos_inexplorados) == 0

    def seleccionar_nodo_con_ucb1(self):
        log_visitas_padre = math.log(self.visitas)
        valor_ucb1 = lambda n: (n.victorias / n.visitas) + math.sqrt(2 * log_visitas_padre / n.visitas)
        return max(self.hijos, key=valor_ucb1)

    def expansion(self):
        movimiento = self.movimientos_inexplorados.pop()
        nuevo_estado = tl.make_move(self.estado, movimiento, self.color)
        nuevo_color = 3 - self.color  # Cambiar el color para el siguiente jugador
        hijo = Nodo(nuevo_estado, self, movimiento, nuevo_color)
        self.hijos.append(hijo)
        return hijo

    def simulacion(self):
        estado_actual = self.estado.copy()
        jugador_actual = self.color
        while True:
            if tl.check_winner(estado_actual) != 0:
                break
            movimientos_posibles = tl.get_available_moves(estado_actual)
            if len(movimientos_posibles) == 0:
                break
            movimiento = random.choice(movimientos_posibles)
            estado_actual = tl.make_move(estado_actual, movimiento, jugador_actual)
            jugador_actual = 3 - jugador_actual
        return tl.check_winner(estado_actual)

    def retropropagacion(self, resultado):
        nodo_actual = self
        while nodo_actual is not None:
            nodo_actual.visitas += 1
            if nodo_actual.color == resultado:
                nodo_actual.victorias += 1
            nodo_actual = nodo_actual.padre


class MCTS:
    def __init__(self, estado_inicial, color_inicial):
        self.raiz = Nodo(estado_inicial, color=color_inicial)

    def buscar(self, iteraciones):
        for _ in range(iteraciones):
            nodo = self.raiz
            while not nodo.es_terminal():
                if not nodo.es_completamente_expandido():
                    nodo = nodo.expansion()
                else:
                    nodo = nodo.seleccionar_nodo_con_ucb1()
            resultado = nodo.simulacion()
            nodo.retropropagacion(resultado)

# Uso del MCTS
estado_inicial = np.zeros((21, 21))  # Tablero inicial vacío
estado_inicial = tl.init_board(estado_inicial)  # Inicializar el tablero
print(estado_inicial)
print(tl.check_winner(estado_inicial))  # Comprobar que el juego no ha terminado
mcts = MCTS(estado_inicial, 1)  # Comienza el jugador 1
mcts.buscar(1000)  # Realizar 1000 iteraciones
print(mcts.raiz.visitas)  # Imprimir el número de visitas de la raíz
