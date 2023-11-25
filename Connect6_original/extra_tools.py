# Visualizacion del tablero
# import matplotlib.pyplot as plt
# import matplotlib.colors as mcolors
# import numpy as np

def show_m_board(m_board):
    """
    Visualizacion del estado del juego
    """

    col = 'ABCDEFGHIJKLMNOPQRS'
    fil = 'SRQPONMLKJIHGFEDCBA'

    # Array de anotaciones de 19x19
    annot = []
    for l1 in fil:
        fila = []
        for l2 in col:
            fila.append(l2 + l1)
        annot.append(fila)

    #Obtener el tablero sin los bordes
    m_board_rep = np.array(m_board)
    m_board_rep = m_board_rep[1:20,1:20]
    
    # Reemplazar los 2 por -1
    m_board_rep = np.where(m_board_rep == 2, -1, m_board_rep)
    
    fontdict = {'fontsize': 10,
                'fontweight' : 50}
    # Crear un colormap personalizado en el que el negro = 1, blanco = -1 y naranja = 0.
    # Define los colores
    colors = [(1, 1, 1), (1, 0.8, 0.4), (0, 0, 0)]

    # Crea el colormap
    n_bins = 3  
    cmap_name = "custom_colormap"
    cm = mcolors.LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bins)

    # Normaliza los valores para que estén en el rango [0, 1]
    norm = mcolors.Normalize(vmin=-1, vmax=1)

    plt.figure(figsize=(6,6))
    # Mostrar movimientos sobre el tablero
    plt.imshow(m_board_rep, cmap=cm, norm=norm) 
    
    # Mostrar anotaciones sobre la imagen
    for i in range(19):
        for j in range(19):
            plt.text(j, i, annot[i][j], ha="center", va="center", color="green", fontdict=fontdict)

    plt.axis('off')
    plt.tight_layout()
    plt.show()