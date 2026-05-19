# Funciones del grupo 3D
#Martin Cuestas 466/24
#Manuel Selen Oliveros 501/24
#Leonardo Dominguez 285/22
# Carga de paquetes necesarios para graficar
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd # Para leer archivos
import geopandas as gpd # Para hacer cosas geográficas
import seaborn as sns # Para hacer plots lindos
import networkx as nx # Construcción de la red en NetworkX
import scipy
from scipy.linalg import solve_triangular

museos = gpd.read_file('https://raw.githubusercontent.com/MuseosAbiertos/Leaflet-museums-OpenStreetMap/refs/heads/principal/data/export.geojson')
barrios = gpd.read_file('https://cdn.buenosaires.gob.ar/datosabiertos/datasets/ministerio-de-educacion/barrios/barrios.geojson')
G_layout = {i:v for i,v in enumerate(zip(museos.to_crs("EPSG:22184").get_coordinates()['x'],museos.to_crs("EPSG:22184").get_coordinates()['y']))}

def construye_adyacencia(D,m):
    # Función que construye la matriz de adyacencia del grafo de museos
    # D matriz de distancias, m cantidad de links por nodo
    # Retorna la matriz de adyacencia como un numpy.
    D = D.copy()
    l = [] # Lista para guardar las filas
    for fila in D: # Recorriendo las filas, anexamos vectores lógicos
      c = [] # Lista para guardar las columnas de una fila
      i = 0 # Contador de museos considerados para cada nodo para evitar que se agreguen más de m-museos
      for columna in fila: # Recorriendo la columna de la fila examinamos cada museo para dicha fila
        if (columna<=fila[np.argsort(fila)[m]] and i<m+1):
          c.append(True) # Si el nodo esta a una a una distancia menor o igual a la del m-esimo más cercano, lo agregamos a lista
          i=i+1 # Actualizamos el contador
        else:
          c.append(False) #Si el nodo no esta a una distancia menor o igual a la del m-esimo más cercano, no lo agregamos a lista
      l.append(c) #Agregamos la fila
    A = np.asarray(l).astype(int) # Convertimos a entero
    np.fill_diagonal(A,0) # Borramos diagonal para eliminar autolinks
    return(A)

D = museos.to_crs("EPSG:22184").geometry.apply(lambda g: museos.to_crs("EPSG:22184").distance(g)).round().to_numpy()

##########################FUNCIONES TP 1##########################

def calcularLU(matriz):
    # Función que calcula el LU de una matriz
    # matriz: una matriz cuadrada (NxN)
    # Retorna la factorización LU a través de una lista con dos matrices L y U de NxN.
    L, U = [],[]
    matriz_copia = matriz.copy() #Copio matriz
    n=matriz.shape[0] #la matriz es cuadrada

    for k in range(n-1):
        # Tomamos los elementos de la columna del pivoteo y los dividimos por el elemento de la diagonal.
        # Ahora tenemos el factor multiplicativo de cada fila
        matriz_copia[k+1:, k] = matriz_copia[k+1:, k] / matriz_copia[k, k]

        # A cada fila debajo del pivoteo le restamos el producto externo del factor multiplicativo x la fila del pivoteo,
        # (excepto la columna del pivoteo para guardanos el factor multiplicativo)
        matriz_copia[k+1:, k+1:] -= np.outer(matriz_copia[k+1:, k], matriz_copia[k, k+1:])

    #Obtenemos L y U
    L = np.tril(matriz_copia,-1) + np.eye(matriz.shape[0]) #obtenemos tringular inferior con diagonal 0 y le sumamos una diagonal de 1
    U = np.triu(matriz_copia) #obtenemos diagonal superior incluyendo la diagonal
    return [L, U]

def calcula_matriz_C(A):
    # Función para calcular la matriz de trancisiones C
    # A: Matriz de adyacencia
    # Retorna la matriz C

    grados = np.sum(A, axis=1) #vemos el grado de cada fila
    K = np.diag(grados)  #La diagonal K con los grados de cada fila
    inversa_de_k = np.diag(1/grados) #La inversa de K (seria el inverso multiplicativo de cada uno)
    transpuesta_A = A.transpose()

    C = transpuesta_A@inversa_de_k # Calculamos C multiplicando la transpuesta de A y la inversa de K

    return C

def calcula_pagerank(A,alfa):
    # Función para calcular PageRank usando LU
    # A: Matriz de adyacencia
    # d: coeficientes de damping
    # Retorna: Un vector p con los coeficientes de page rank de cada museo
    C = calcula_matriz_C(A)
    N = np.shape(A)[0] # Obtenemos el número de museos N a partir de la estructura de la matriz A
    M = (N/alfa) * (np.eye(N) - (1-alfa) * C)
    L= calcularLU(M)[0] # Calculamos descomposición LU a partir de C y d
    U = calcularLU(M)[1]
    b = (N/alfa) * np.ones(N)  # Vector de 1s, multiplicado por el coeficiente correspondiente usando d y N.
    Up = scipy.linalg.solve_triangular(L,b,lower=True) # Primera inversión usando L
    p = scipy.linalg.solve_triangular(U,Up) # Segunda inversión usando U
    return p


def calcula_matriz_C_continua(D):
    # Función para calcular la matriz de trancisiones C
    # D: Matriz de distancia
    # Retorna la matriz C en versión continua
    D = D.copy()
    F = 1/D
    np.fill_diagonal(F,0)
    Kinv = np.diag(1/(np.sum(F, axis=1))) # Calcula inversa de la matriz K, que tiene en su diagonal la suma por filas de F
    C = F@Kinv # Calcula C multiplicando F y Kinv
    return C

def calcula_B(C,cantidad_de_visitas):
    # Recibe la matriz T de transiciones, y calcula la matriz B que representa la relación entre el total de visitas y el número inicial de visitantes
    # suponiendo que cada visitante realizó cantidad_de_visitas pasos
    # C: Matirz de transiciones
    # cantidad_de_visitas: Cantidad de pasos en la red dado por los visitantes. Indicado como r en el enunciado
    # Retorna:Una matriz B que vincula la cantidad de visitas w con la cantidad de primeras visitas v
    B = np.zeros_like(C)
    for i in range(cantidad_de_visitas):
      B+=np.linalg.matrix_power(C, i)
        # Sumamos las matrices de transición para cada cantidad de pasos
    return B

def resolver_v(B,w):
  # Función que resuelve el vector v de la ecuación v = B^(⁻1)w con LU
  # B: Matriz calculada a través de la sumatoria de los exponentes de C hasta r-1 pasos
  # w: Vector con el número total de visitas que recibió cada museo
  # Retorna: Un vector v con el número de personas de al museo
  L= calcularLU(B)[0] # Calculamos descomposición LU a partir de C y d
  U = calcularLU(B)[1] # Calculamos descomposición LU a partir de C y d
  Uv = scipy.linalg.solve_triangular(L,w,lower=True) # Primera inversión usando L
  v = scipy.linalg.solve_triangular(U,Uv,lower=False) # Segunda inversión usando U
  return v

def calcula_Norma_1_matriz(matriz):
  # Función que calcula la máxima suma de absolutos por cada columna de una matriz, su norma 1
  max = 0
  filas = matriz.shape[0]
  columnas = matriz.shape[1]
  for j in range(columnas):
    suma = 0
    for i in range(filas):
      suma += abs(matriz[i,j])
    if suma >= max:
      max = suma
  return max

def calcula_Norma_1_vector(vector):
  #Función que calcula la norma 1 de un vector
  res = 0
  for i in vector:
    res += abs(i)
  return res

def calcular_inversa(matriz):
    #Función que calcula la inversa de una matriz
    #Matriz: Una matriz cuadrada inversible
    #Retorna: La inversión de la matriz
    dimension = matriz.shape[0]
    LU_de_matriz = calcularLU(matriz) #Calculamos el LU de matriz
    L_de_matriz = LU_de_matriz[0]
    U_de_matriz = LU_de_matriz[1]
    Identidad = np.identity(dimension) #Identidad para buscar inversa por propiedad: Matriz . Matriz(inversa) = Identidad
    Y = solve_triangular(L_de_matriz, Identidad, lower=True) # Paso intermedio para resolver sistema triangular por LU
    Inversa_de_matriz = solve_triangular(U_de_matriz, Y, lower=False) #Paso final para resolver sistema mediante la triangulación UY

    return Inversa_de_matriz

def calcular_cond_1(matriz):
    #Función que calcula el número condicional de norma 1 de una matriz
    inv_matriz = calcular_inversa(matriz) #Calculamos la inversa de la matriz
    norma_matriz = calcula_Norma_1_matriz(matriz) #Calculamos la norma 1 de la matriz
    norma_inv_matriz = calcula_Norma_1_matriz(inv_matriz) #Calculamos la norma 1 de la inversa de la matriz
    cond = norma_matriz * norma_inv_matriz #Multiplicamos ambas normas para sacar la condición
    return cond

##Funciones de visualización y graficación

def visualizador(matriz_A,alfa, cant_principales): #Función para construir los distintos mapas
    G = nx.from_numpy_array(matriz_A)
    factor_escala = 1e4 # Escalamos los nodos 10 mil veces para que sean bien visibles
    fig, ax = plt.subplots(figsize=(10, 10)) # Visualización de la red en el mapa
    ax.set_title(f"Mapa de la red escalada por PageRank - Top {cant_principales} nodos\n(Alfa = {alfa})", fontsize=14) #Titulación de cada mapa basada en sus alfa y m
    ax.axis("off")  # Ocultamos ejes para una mejor visualización
    barrios.to_crs("EPSG:22184").boundary.plot(color='gray',ax=ax) # Graficamos Los barrios
    pr = calcula_pagerank(matriz_A,alfa) #PageRank
    pr = pr/pr.sum() # Normalizamos para que sume 1
    Nprincipales = cant_principales # Cantidad de principales
    principales = np.argsort(pr)[-Nprincipales:] # Identificamos a los N principales
    labels = {n: str(n) if i in principales else "" for i, n in enumerate(G.nodes)} # Nombres para esos nodos
    nx.draw_networkx(G,G_layout,node_size = pr*factor_escala, ax=ax,with_labels=False) # Graficamos red
    plt.tight_layout()
    plt.show()

def b3():
    #Función que gráfica las diferentes redes conectando a cada museo con sus m vecinos más cercanos variando m entre 1, 3, 5 y 10 
    #usando α=1/5
    ms = [1, 3, 5, 10]
    alfa = 0.2
    factor_escala = 1e4  # Escala para los tamaños de nodo

    fig, axs = plt.subplots(2, 2, figsize=(15, 15))  # 2x2 subplot donde iran las redes
    axs = axs.flatten() 

    for idx, Mprincipales in enumerate(ms):
        ax = axs[idx]

        A = construye_adyacencia(D, ms[idx]) # Creamos grafo desde la matriz de adyacencia correspondiente al m
        G = nx.from_numpy_array(A)

        pr = calcula_pagerank(A, alfa) # Calculam PageRank
        pr = pr / pr.sum() # Normalizamos el PageRank

        principales = np.argsort(pr)[-Mprincipales:] # Identificamos a los m museos principales
        labels = {n: str(n) if i in principales else "" for i, n in enumerate(G.nodes)}

        # Título del subplot
        ax.set_title(f"Top {ms[idx]} nodos por PageRank\n(Alfa = {alfa})", fontsize=12)
        ax.axis("off")

        barrios.to_crs("EPSG:22184").boundary.plot(color='gray', ax=ax)

        nx.draw_networkx(
            G, G_layout,
            node_size=pr * factor_escala,
            ax=ax,
            with_labels=False
        )

    plt.tight_layout()
    plt.show()

def arte_oriental():
    #Función que gráfica las diferentes redes conectando a cada museo con sus m vecinos más cercanos variando m entre 1, 3, 5 y 10 
    #usando α=1/5 y centrando al nodo del Museo Nacional de Arte Oriental y sus vecinos (entrantes y salientes)
  ms = [1, 3, 5, 10]
  alfa = 0.2
  factor_escala = 1e4

  fig, axs = plt.subplots(2, 2, figsize=(15, 15)) # 2x2 subplot donde iran las redes
  axs = axs.flatten()

  # Índice del Museo Nacional de Arte Oriental
  id_museo_oriental = museos[museos['name'] == 'Museo Nacional de Arte Oriental'].index[0]

  for idx, Nprincipales in enumerate(ms):
      ax = axs[idx]

      # Matriz de adyacencia y grafo dirigido
      A = construye_adyacencia(D, ms[idx]) # Creamos grafo desde la matriz de adyacencia correspondiente al m
      G = nx.from_numpy_array(A, create_using=nx.DiGraph)

      pr = calcula_pagerank(A, alfa) # Calculam PageRank
      pr = pr / pr.sum() # Normalizamos el PageRank

      entrantes = [n for n in G.predecessors(id_museo_oriental)] #Registramos los vecinos entrantes
      salientes = [n for n in G.successors(id_museo_oriental)] #Registramos los vecinos salientes

      colores_nodos = [] #Registramos como colorear cada nodo
      alphas_nodos = []
      for i in G.nodes:
          if i == id_museo_oriental: #Museo Nacional de Arte Oriental de rojo
              colores_nodos.append("red")
              alphas_nodos.append(1.0)
          elif i in entrantes: #Vecinos entrantes de azul
              colores_nodos.append("blue")
              alphas_nodos.append(1.0)
          elif i in salientes: #Vecinos salientes de verde
              colores_nodos.append("green")
              alphas_nodos.append(1.0)
          else:
              colores_nodos.append("lightgray") #Demás nodos de gris
              alphas_nodos.append(1.0)

      
      edge_colors = [] #Colores de aristas
      edge_alphas = []
      for u, v in G.edges():
          if (u == id_museo_oriental and v in salientes) or (v == id_museo_oriental and u in entrantes): #Aristas a estudiar en negro
              edge_colors.append("black")
              edge_alphas.append(1.0)
          else:
              edge_colors.append("gray") #Demás aristas en gris con baja opacidad para no
              edge_alphas.append(0.8)

      ax.set_title(f"Top {ms[idx]} nodos por PageRank\n(Alfa = {alfa})", fontsize=12)
      ax.axis("off")

      barrios.to_crs("EPSG:22184").boundary.plot(color='gray', ax=ax)

      # Dibujar nodos
      nx.draw_networkx_nodes(
          G, G_layout,
          node_size=pr * factor_escala,
          node_color=colores_nodos,
          alpha=alphas_nodos,
          ax=ax
      )

      # Dibujar aristas
      nx.draw_networkx_edges(
          G, G_layout,
          edge_color=edge_colors,
          alpha=edge_alphas,
          arrows=False,
          ax=ax
      )
      texto = f"Entrantes: {len(entrantes)}\nSalientes: {len(salientes)}"
      ax.text(0.05, 0.95, texto, transform=ax.transAxes,
      verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7))

  # Anotar cantidad de entrantes y salientes en el plot

  
  plt.tight_layout()
  plt.show()

def patologia():
  ms = [1, 3, 5, 10]
  alfa = 0.2
  factor_escala = 1e4

  fig, axs = plt.subplots(2, 2, figsize=(15, 15))
  axs = axs.flatten()

  # Índice del Museo Nacional de Patología
  id_museo_patologia = museos[museos['name'] == 'Museo de Patología'].index[0]

  for idx, Nprincipales in enumerate(ms):
      ax = axs[idx]

      # Matriz de adyacencia y grafo dirigido
      A = construye_adyacencia(D, ms[idx])
      G = nx.from_numpy_array(A, create_using=nx.DiGraph)

      # PageRank normalizado
      pr = calcula_pagerank(A, alfa)
      pr = pr / pr.sum()

      # Entrantes y salientes del museo
      entrantes = [n for n in G.predecessors(id_museo_patologia)]
      salientes = [n for n in G.successors(id_museo_patologia)]

      # Colores de nodos
      colores_nodos = []
      alphas_nodos = []
      for i in G.nodes:
          if i == id_museo_patologia:
              colores_nodos.append("red")
              alphas_nodos.append(1.0)
          elif i in entrantes:
              colores_nodos.append("blue")
              alphas_nodos.append(1.0)
          elif i in salientes:
              colores_nodos.append("green")
              alphas_nodos.append(1.0)
          else:
              colores_nodos.append("lightgray")
              alphas_nodos.append(1.0)

      # Colores de aristas
      edge_colors = []
      edge_alphas = []
      for u, v in G.edges():
          if (u == id_museo_patologia and v in salientes) or (v == id_museo_patologia and u in entrantes):
              edge_colors.append("black")
              edge_alphas.append(1.0)
          else:
              edge_colors.append("gray")
              edge_alphas.append(0.8)

      ax.set_title(f"Top {ms[idx]} nodos por PageRank\n(Alfa = {alfa})", fontsize=12)
      ax.axis("off")

      barrios.to_crs("EPSG:22184").boundary.plot(color='gray', ax=ax)

      # Dibujar nodos
      nx.draw_networkx_nodes(
          G, G_layout,
          node_size=pr * factor_escala,
          node_color=colores_nodos,
          alpha=alphas_nodos,
          ax=ax
      )

      # Dibujar aristas
      nx.draw_networkx_edges(
          G, G_layout,
          edge_color=edge_colors,
          alpha=edge_alphas,
          arrows=False,
          ax=ax
      )
      texto = f"Entrantes: {len(entrantes)}\nSalientes: {len(salientes)}"
      ax.text(0.05, 0.95, texto, transform=ax.transAxes,
      verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7))
  plt.tight_layout()
  plt.show()

def c3():
    alfas = [6/7, 4/5, 2/3, 1/2, 1/3, 1/5, 1/7]
    factor_escala = 1e4
    fig, axs = plt.subplots(nrows=2, ncols=4, figsize=(20, 10))
    axs = axs.flatten()

    matriz_A = construye_adyacencia(D, 5)
    G = nx.from_numpy_array(matriz_A)
    barrios_proj = barrios.to_crs("EPSG:22184")  # Solo lo hacemos una vez

    for idx, alfa in enumerate(alfas):
        ax = axs[idx]
        ax.set_title(f"Top 5 nodos\nAlfa = {alfa:.2f}", fontsize=10)
        ax.axis("off")
        barrios_proj.boundary.plot(color='gray', ax=ax)

        pr = calcula_pagerank(matriz_A, alfa)
        pr = pr / pr.sum()

        principales = np.argsort(pr)[-5:]
        labels = {n: str(n) if i in principales else "" for i, n in enumerate(G.nodes)}

        nx.draw_networkx(
            G,G_layout,
            node_size=pr * factor_escala,
            ax=ax,
            with_labels=False
        )

    for ax in axs[len(alfas):]: # Como sobran subplots (porque son 8 y usamos 7), apagamos el último
        ax.axis("off")

plt.tight_layout()
plt.show()

def obtener_pagerank_variando_m(D):
    # Función para calcular el pagerank variando m en base a la matriz de distancia
    # D: Matriz distancia
    # Retorna: los 3 museos con más pagerank de cada m y sus pageranks normalizados
    #Nos guardamos los 3 museos principales y todos los pr en cada iteracion (o sea, por cada m tomamos los 3 principales)
    museos_principales = []
    pr_normalizados_totales = []
    for m in range(1,11):
        matriz_A = construye_adyacencia(D,m)
        pr = calcula_pagerank(matriz_A ,1/5) #Vemos el pagerank variando m
        pr = pr/pr.sum() #Normalizamos valores
        Nprincipales = 3 # Cantidad de Museos principales
        principales = np.argsort(pr)[-Nprincipales:] # Identificamos a los 3 museos principales
        museos_principales.append(principales)
        pr_normalizados_totales.append(pr)

    museos_principales = np.unique(np.array(museos_principales)) #Pasamos a un array y sacamos los repetidos
    pr_normalizados_totales = np.array(pr_normalizados_totales)


    return museos_principales, pr_normalizados_totales

def graficar_pagerank_variando_m(D):
    # Función para gráficar el pagerank en base a la matriz de distancia
    # D: Matriz distancia
    # Retorna: el gráfico de los museos con pagerank variando m
    #Nos guardamos los 3 museos principales y todos los pr en cada iteracion (o sea, por cada m tomamos los 3 principales)
    museos_principales, pr_normalizados_totales = obtener_pagerank_variando_m(D)
    pr_seleccionados = []
    #Seleccionamos el pr a lo largo de cada m de cada museo principal
    for indices_museos in museos_principales:
        pr_seleccionados.append(pr_normalizados_totales[:, indices_museos])

    plt.figure(figsize=(15, 12))
    colores = cm.get_cmap('tab10', len(museos_principales))
    # Graficamos todos los museos seleccionados
    for i, museo_id in enumerate(museos_principales):
        plt.plot(range(1, 11), pr_seleccionados[i], label=f'Museo {museo_id}: {museos["name"][museo_id]}', color=colores(i))

    # Leyenda y referencias del mapa

    plt.title('Variación del PageRank de los Museos Principales variando m (m=1 a m=10)')
    plt.xlabel('m [cantidad de vecinos]')
    plt.ylabel('PageRank Normalizado')
    plt.xticks(range(1, 11))
    plt.grid(True)
    plt.legend(fontsize=8, loc="best")
    plt.show()

def obtener_pagerank_variando_alfa(D):
    # Función para gráficar el pagerank en base a la matriz de distancia variando el alfa
    # D: Matriz distancia
    # Retorna: los 3 museos con más alfa de cada m y sus pageranks normalizados
    museos_principales = []
    pr_normalizados_totales = []
    for alfa in range(1,11):
        matriz_A = construye_adyacencia(D,5)
        pr = calcula_pagerank(matriz_A ,1/alfa) #Vemos el pagerank variando alfa
        pr = pr/pr.sum() #Normalizamos valores
        Nprincipales = 3 # Cantidad de Museos principales
        principales = np.argsort(pr)[-Nprincipales:] # Identificamos a los 3 museos principales
        museos_principales.append(principales)
        pr_normalizados_totales.append(pr)

    museos_principales = np.unique(np.array(museos_principales)) #Pasamos a un array y sacamos los repetidos
    pr_normalizados_totales = np.array(pr_normalizados_totales)


    return museos_principales, pr_normalizados_totales

def graficar_pagerank_variando_alfa(D):
    # Función para gráficar el pagerank en base a la matriz de distancia variando el alfa
    # D: Matriz distancia
    # Retorna: el gráfico de los museos con pagerank variando alfa
    museos_principales, pr_normalizados_totales = obtener_pagerank_variando_alfa(D)
    pr_seleccionados = []
    #Seleccionamos el pr a lo largo de cada alfa de cada museo principal
    for indices_museos in museos_principales:
        pr_seleccionados.append(pr_normalizados_totales[:, indices_museos])

    plt.figure(figsize=(15, 12))
    # Graficamos todos los museos seleccionados
    for i, museo_id in enumerate(museos_principales):
        plt.plot(range(1, 11), pr_seleccionados[i], label=f'Museo {museo_id}: {museos["name"][museo_id]}')

    plt.title(r'Variación del PageRank de los Museos Principales variando $\alpha$ ($\alpha$=1 a $\alpha$=1/10)')
    plt.xlabel(r'$\frac{1}{\alpha}$ [factor de amortiguamiento]')
    plt.ylabel('PageRank Normalizado')
    plt.xticks(range(1, 11))
    plt.grid(True)
    plt.legend(fontsize=8, loc="best")
    plt.show()

##########################FUNCIONES TP 2##########################

def calcular_matriz_K(A):
    # Calcula la matriz de grados K (grados de cada fila)
    grados = np.sum(A, axis=1) #vemos el grado de cada fila
    K = np.diag(grados)  #La diagonal K con los grados de cada fila
    return K


def calcula_L(A):
    # La función recibe la matriz de adyacencia A y calcula la matriz laplaciana
    K = calcular_matriz_K(A)
    L = K - A
    return L


def E(A):
  #Retorna sumatoria de todos los grados de la matriz A
  return np.sum(A)

def calcula_R(A):
    #La función recibe la matriz de adyacencia A y devuelve la matriz de modularidad
    P =  np.zeros(A.shape) #matriz de ceros del mismo tamaño de la matriz cuadrada A
    K = calcular_matriz_K(A) #Obtenemos la matriz diagonal de grados
    grados = np.diag(K) #Obtenemos un vector con los grados
    E2 = E(A) #Los grados totales de la matriz A
    for i in range (0, A.shape[0], 1):
        for j in range (0, A.shape[1],1):
            P[i][j] = (grados[i]*grados[j])/E2
    R = A-P #matriz modular
    return R

def calcula_lambda(L,v):
    #Dada una matriz laplaciana y un autovector de la matriz original, devuelve la minimizacion
    lambdadon = 1/4*(v.transpose()@L@v)
    return lambdadon

def calcula_Q(R,v):
   #Dada una matriz de modularidad y un autovector de la matriz original, devuelve la maximizacion
    Q = (v.transpose()@R@v)
    return Q

def metpot1(A,tol=1e-8,maxrep=np.inf):
   # Recibe una matriz A y calcula su autovalor de mayor módulo, con un error relativo menor a tol y-o haciendo como mucho maxrep repeticiones
   v = 2 * np.random.rand(A.shape[1]) - 1 # Generamos un vector de partida aleatorio, entre -1 y 1, CONSULTAR QUE ONDA
   v = v/np.linalg.norm(v, 2) # Lo normalizamos
   v1 = A@v # Aplicamos la matriz una vez
   v1 = v1/np.linalg.norm(v1, 2) # normalizamos
   l = (v@A@v)/(v@v) # Calculamos el autovector estimado Ver que onda v.transpose()
   l1 = (v1@A@v1)/(v1@v1) # Y el estimado en el siguiente paso
   nrep = 0 # Contador
   while np.abs(l1-l)/np.abs(l) > tol and nrep < maxrep: # Si estamos por debajo de la tolerancia buscada
      v = v1 # actualizamos v y repetimos
      l = l1
      v1 = A@v # Aplicamos la matriz una vez
      v1 = v1/np.linalg.norm(v1, 2) # normalizamos
      l1 = (v1@A@v1)/(v1@v1) # calculamos autovector
      nrep += 1 # Un pasito mas
   if not nrep < maxrep:
      print('MaxRep alcanzado')
   l = (v1@A@v1)/(v1@v1) # Calculamos el autovalor
   return v1,l,nrep<maxrep

def calcula_Norma_2_vector(v):
  #Función que calcula la norma 2 de un vector
  res = 0
  for i in v:
    res += i**2
  return np.sqrt(res)

def deflaciona(A,tol=1e-8,maxrep=np.inf):
    # Recibe la matriz A, una tolerancia para el método de la potencia, y un número máximo de repeticiones. Devuelve una matriz cuyo autovalor asociado a su autovector 1 es 0
    v1,l1,_ = metpot1(A,tol,maxrep) # Buscamos primer autovector con método de la potencia
    deflA = A - l1*(np.linalg.outer(v1,v1)/(v1.transpose()@v1)) # Sugerencia, usar la funcion outer de numpy
    return deflA

def metpotI(A,mu,tol=1e-8,maxrep=np.inf):
    # Retorna el primer autovalor de la inversa de A + mu * I, junto a su autovector y si el método convergió.
    matriz = A+(mu*np.eye(A.shape[0])) #matriz A + matriz diagonal con sus valores en mu
    inversa = calcular_inversa(matriz)
    return metpot1(inversa,tol=tol,maxrep=maxrep) #Obtenemos 1er autovalor de la inversa (es decir, el autovalor más chico de la matriz A+mu)

def metpotI2(A,mu,tol=1e-8,maxrep=np.inf):
   # Recibe la matriz A, y un valor mu y retorna el segundo autovalor y autovector de la matriz A,
   # suponiendo que sus autovalores son positivos excepto por el menor que es igual a 0
   # Retorna el segundo autovector, su autovalor, y si el metodo llegó a converger.
   X = A+(mu*np.eye(A.shape[0])) # Calculamos la matriz A shifteada en mu
   iX = calcular_inversa(X) # La invertimos
   defliX = deflaciona(iX) # La deflacionamos
   v,l,_ =  metpot1(defliX) # Buscamos su segundo autovector
   l = 1/l # Reobtenemos el autovalor correcto
   l -= mu
   return v,l,_

def metpot2(A,v1,l1,tol=1e-8,maxrep=np.inf):
   #Recibe una matriz A, y primer autovalor y autovector del mismo, y devuelve el segundo autovalor y autovector más grande
   deflA = deflaciona(A,tol,maxrep)
   return metpot1(deflA,tol,maxrep)

def laplaciano_iterativo(A,niveles,nombres_s=None):
    # Recibe una matriz A, una cantidad de niveles sobre los que hacer cortes, y los nombres de los nodos
    # Retorna una lista con conjuntos de nodos representando las comunidades.
    # La función debe, recursivamente, ir realizando cortes y reduciendo en 1 el número de niveles hasta llegar a 0 y retornar.
    if nombres_s is None: # Si no se proveyeron nombres, los asignamos poniendo del 0 al N-1
        nombres_s = range(A.shape[0])
    if A.shape[0] == 1 or niveles == 0: # Si llegamos al último paso, retornamos los nombres en una lista
        return([nombres_s])
    else: # Sino:
        L = calcula_L(A) # Recalculamos el L
        v, l, _ = metpotI2(L, mu=0.01) # Encontramos el segundo autovector de L
        # Recortamos A en dos partes, la que está asociada a el signo positivo de v y la que está asociada al negativo

        Ap = A[v > 0][:, v > 0] # Asociado al signo positivo
        Am = A[v < 0][:, v < 0] # Asociado al signo negativo

        return(
                laplaciano_iterativo(Ap,niveles-1,
                                     nombres_s=[ni for ni,vi in zip(nombres_s,v) if vi>0]) +
                laplaciano_iterativo(Am,niveles-1,
                                     nombres_s=[ni for ni,vi in zip(nombres_s,v) if vi<0])
                )


def modularidad_iterativo(A=None,R=None,nombres_s=None):
    # Recibe una matriz A, una matriz R de modularidad, y los nombres de los nodos
    # Retorna una lista con conjuntos de nodos representando las comunidades.

    if A is None and R is None:
        print('Dame una matriz')
        return(np.nan)
    if R is None:
        R = calcula_R(A)
    if nombres_s is None:
        nombres_s = range(R.shape[0])
    # Acá empieza lo bueno
    if R.shape[0] == 1: # Si llegamos al último nivel
        return([nombres_s])
    else:
        v,l,_ = metpot1(R) # Primer autovector y autovalor de R
        # Modularidad Actual:
        Q0 = np.sum(R[v>0,:][:,v>0]) + np.sum(R[v<0,:][:,v<0])
        if Q0<=0 or all(v>0) or all(v<0): # Si la modularidad actual es menor a cero, o no se propone una partición, terminamos
            return([nombres_s])
        else:
            ## Hacemos como con L, pero usando directamente R para poder mantener siempre la misma matriz de modularidad
            Rp = R[v > 0][:, v > 0] # Parte de R asociada a los valores positivos de v
            Rm = R[v < 0][:, v < 0] # Parte asociada a los valores negativos de v
            vp,lp,_ = metpot1(Rp)  # autovector principal de Rp
            vm,lm,_ = metpot1(Rm) # autovector principal de Rm

            # Calculamos el cambio en Q que se produciría al hacer esta partición
            Q1 = 0
            if not all(vp>0) or all(vp<0):
               Q1 = np.sum(Rp[vp>0,:][:,vp>0]) + np.sum(Rp[vp<0,:][:,vp<0])
            if not all(vm>0) or all(vm<0):
                Q1 += np.sum(Rm[vm>0,:][:,vm>0]) + np.sum(Rm[vm<0,:][:,vm<0])
            if Q0 >= Q1: # Si al partir obtuvimos un Q menor, devolvemos la última partición que hicimos
                return([[ni for ni,vi in zip(nombres_s,v) if vi>0],[ni for ni,vi in zip(nombres_s,v) if vi<0]])
            else:
                # Sino, repetimos para los subniveles
                return(
                     modularidad_iterativo(R=Rp, nombres_s=[ni for ni, vi in zip(nombres_s, v) if vi > 0]) +
                     modularidad_iterativo(R=Rm, nombres_s=[ni for ni, vi in zip(nombres_s, v) if vi < 0])
                     )


def simetrizacion(A):
  #Dada una matriz, lo transformamos en simetrica
  A1 = np.ceil(0.5*(A+A.transpose()))
  return A1

##Funciones de visualización y graficación


def construye_matriz_de_adyacencia_simetrica(D,m):
  #Dado m = cantidad de vecinos
  #dado D las distancias a los museos
  #Devuelve matriz simetrica
  matriz_adyacencia = construye_adyacencia(D,m)
  matriz_adyacencia = simetrizacion(matriz_adyacencia)
  return matriz_adyacencia

def itera_usando_laplaciano(matriz_simetrica, iteraciones):
    #Dado matriz simetrica y la cantidad de iteraciones
    #Devuelve array de comunidades de nodos
  return laplaciano_iterativo(matriz_simetrica, iteraciones)


def asignacion_colores(comunidades):
  #Dado un array de comunidades de colores, devuelve un array donde en cada indice del nodo se le asigna un color
  paleta_colores = ["red", "blue", "green", "orange", "purple", "brown", "pink", "cyan", "black", "grey",
                    "olive", "lime", "teal", "navy", "maroon", "gold", "silver", "violet", "indigo", "crimson"] #Array con colores
  nodos_coloreados = [" "] * len(museos) #Creamos un array de strings vacios segun la cantidad de nodos/museos
  # Usamos un diccionario donde en cada indice i corresponde a la comunidad y sus respectivos nodos
  for i, nodos in enumerate(comunidades):
    if len(nodos)>0:
      #Si la comunidad tiene nodos, entonces le asignamos a cada nodo de esa comunidad el mismo color
      for nodo in nodos:
        nodos_coloreados[nodo] = paleta_colores[i] #AL array de strings vacios le asignamos ahora un color de la paleta según en que comunidad pertenece
  return nodos_coloreados

def construccion_mapa(matriz, array_coloracion, titulo):
  #Dado una matriz de adyacencia y un array de nodos coloreados
  #Devuelve un mapa coloreado
  G = nx.from_numpy_array(matriz) # Construimos la red a partir de la matriz de adyacencia
  # Construimos un layout a partir de las coordenadas geográficas
  G_layout = {i:v for i,v in enumerate(zip(museos.to_crs("EPSG:22184").get_coordinates()['x'],museos.to_crs("EPSG:22184").get_coordinates()['y']))}
  fig, ax = plt.subplots(figsize=(15, 15)) # Visualización de la red en el mapa
  barrios.to_crs("EPSG:22184").boundary.plot(color='gray',ax=ax) # Graficamos Los barrios
  nx.draw_networkx(G,G_layout,ax=ax) # Graficamos los museos
  nx.draw_networkx_nodes(
            G,
            G_layout,
            ax=ax,
            node_color=array_coloracion,
        )
  plt.title(titulo)
  plt.show()

def visualizador_laplaciano(D, m, cortes):
  #Dado la cantidad de vecinos, cortes y mapa de distancias
  #Crea visualizacion del mapa usando matriz laplaciano
  matriz = construye_matriz_de_adyacencia_simetrica(D,m)
  comunidades_iteradas = laplaciano_iterativo(matriz, cortes)
  array_coloracion = asignacion_colores(comunidades_iteradas)
  construccion_mapa(matriz, array_coloracion, f"Laplaciano: Visualización de Comunidades (m={m}, Particiones={cortes}) Comunidades = {len(comunidades_iteradas)}")



def visualizador_modularidad(D, m):
  #Dado la cantidad de vecinos y mapa de distancias
  #Crea visualizacion del mapa usando modularidad
  matriz = construye_matriz_de_adyacencia_simetrica(D,m)
  comunidades_iteradas = modularidad_iterativo(matriz)
  array_coloracion = asignacion_colores(comunidades_iteradas)
  construccion_mapa(matriz, array_coloracion, f"Modularidad: Visualización de Comunidades (m={m}) Comunidades = {len(comunidades_iteradas)}")

def comparar_mapas_comunidades(D, m, niveles):
    #Dado una matriz de adyacencia simetrica y cantidad de vecinos y niveles a realizar
    #Devuelve dos mapas, a la izquierda un mapa de comunidades usando el laplaciano iterando hasta la cantidad de cores
    #a la derecha un mapa de comunidades utilizando modularidad
    fig, axs = plt.subplots(1, 2, figsize=(30, 15))

    # Generamos las 4 combinaciones
    configuraciones = [
        ("Laplaciano", m, niveles, axs[0]),
        ("Modularidad", m, None, axs[1])
    ] #Asignamos la posicion de cada mapa


    for metodo, m, niveles, ax in configuraciones:
        matriz = construye_matriz_de_adyacencia_simetrica(D, m)

        # Según el metodo usado, creamos el mapa correspondiente y su titulo
        if metodo == "Laplaciano":
            comunidades = laplaciano_iterativo(matriz, niveles)
            titulo = f"{metodo}: m={m}, niveles={niveles}, comunidades={len(comunidades)}"
        else:
            comunidades = modularidad_iterativo(matriz)
            titulo = f"{metodo}: m={m}, comunidades={len(comunidades)}"

        array_coloracion = asignacion_colores(comunidades) #Obtenemos un array con cada nodo coloreado según su comunidad

        G = nx.from_numpy_array(matriz) # Graficamos los museos con sus vecinos
        G_layout = {
            i: v for i, v in enumerate(
                zip(
                    museos.to_crs("EPSG:22184").get_coordinates()['x'],
                    museos.to_crs("EPSG:22184").get_coordinates()['y']
                )
            )
        }

        barrios.to_crs("EPSG:22184").boundary.plot(color='gray', ax=ax) # Graficamos Los barrios
        nx.draw_networkx(
            G,
            G_layout,
            ax=ax,
            node_color=array_coloracion,
            with_labels=False,
            node_size=100
        )
        ax.set_title(titulo, fontsize=14)
        ax.axis('off')

    plt.tight_layout()
    plt.show()
