# UNIVERSIDAD DE COLIMA
# INGENIERÍA EN COMPUTACIÓN INTELIGENTE
# REDES NEURONALES
# 7°D
# AUTOR: CRISTIAN ARMANDO LARIOS BRAVO





# Importación de librerías.
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt



plt.ion()

# Mapas auto-organizados o SOM.
class SOM:
    # Constructor.
    def __init__(self, n, m, dim, tasa_aprendizaje=0.1, sigma=1.0):
        # Inicialización de los pesos.
        self.weights = np.random.rand(n, m, dim)
        # self.weights = 2 * np.random.rand(n, m, dim) - 1
        # Inicialización de la tasa de aprendizaje.
        self.tasa_aprendizaje = tasa_aprendizaje
        # Inicialización del radio.
        self.sigma = sigma # Que afecta qué tan amplio es el ajuste de los pesos alrededor de la neurona ganadora.

    # Función de entrenamiento.
    def train(self, data, epocas):
        # Número de datos.
        n = data.shape[0]

        print(f"Pesos iniciales:")
        # print(self.weights)
        for i in range(len(self.weights)):
            for j in range(len(self.weights)):
                print(f"Neurona ({i}, {j}): {self.weights[i, j]}")

        # Entrenamiento.
        for epoca in range(epocas):
            # Tasa de aprendizaje.
            lr = self.tasa_aprendizaje * (1 - epoca / epocas) # Se reduce la tasa de aprendizaje a medida que avanza el entrenamiento.
            # Radio de vecindad.
            s = self.sigma * (1 - epoca / epocas)
            # Selección de un dato aleatorio.
            i = np.random.randint(n)
            # Dato.
            x = data[i]
            # Cálculo de la distancia euclidiana.
            distancias = np.linalg.norm(self.weights - x, axis=2)
            # Neurona ganadora (BMU).
            bmu = np.unravel_index(np.argmin(distancias), distancias.shape) # Se elige la que tenga la menor distancia.
            # Actualización de los pesos.
            for i in range(self.weights.shape[0]):
                for j in range(self.weights.shape[1]):
                    d = np.linalg.norm(np.array([i, j]) - np.array(bmu)) # Distancia entre la neurona ganadora y la neurona actual.
                    # Actualización de los pesos.
                    if d <= s:  # Si la distancia es menor o igual al radio.
                        h = np.exp(-d**2 / (2 * s**2)) # Función de vecindad.
                        self.weights[i, j] += lr * h * (x - self.weights[i, j]) # El cambio en los pesos es proporcional a la tasa de aprendizaje y la diferencia entre el vector de entrada "x" y los pesos actuales de la neurona
            if epoca % 10 == 0:
                # Gráficar la malla de neuronas.
                # plt.scatter(data[:, 0], data[:, 1], c=df["Species"].astype("category").cat.codes)
                for i in range(self.weights.shape[0]):
                    for j in range(self.weights.shape[1]):
                        if i > 0:
                            plt.plot([self.weights[i, j, 0], self.weights[i - 1, j, 0]], [self.weights[i, j, 1], self.weights[i - 1, j, 1]], c="black")
                        if j > 0:
                            plt.plot([self.weights[i, j, 0], self.weights[i, j - 1, 0]], [self.weights[i, j, 1], self.weights[i, j - 1, 1]], c="black")
                plt.scatter(self.weights[:, :, 0], self.weights[:, :, 1], c="red")
                
                # Graficar la neurona ganadora
                bmu_x, bmu_y = self.weights[bmu[0], bmu[1], 0], self.weights[bmu[0], bmu[1], 1]
                plt.scatter(bmu_x, bmu_y, c="blue", marker="x", s=100, label="Neurona Ganadora")
                
                plt.title(f"Época {epoca + 1}")
                # plt.show()
                plt.draw()
                # plt.pause(0.001)
                plt.pause(0.1)
                plt.clf()
                    
    # Función de asignación.
    # Asigna a cada vector de entrada una neurona ganadora
    def asignar(self, data):
        # Número de datos.
        n = data.shape[0]
        asignaciones = np.zeros(n)
        for i in range(n):
            x = data[i]
            distancias = np.linalg.norm(self.weights - x, axis=2)
            bmu = np.unravel_index(np.argmin(distancias), distancias.shape)
            asignaciones[i] = bmu[0] * self.weights.shape[1] + bmu[1]
        return asignaciones

opc = str(input("¿Cuál dataset desea utilizar?\n1. Iris.csv\n2. Bill_authentication.csv\nIngrese la opción: "))
if opc == "1":
    # Lectura de datos.
    df = pd.read_csv("C:/Users/Cristian/Programacion/Python/Redes_Neuronales/Iris.csv")
    # df = pd.read_csv("C:/Programacion/Python/Redes_Neuronales/Iris.csv")
    # Datos.
    data = df.iloc[:, 1:5].values  # Se eliminan las columnas de identificación. (Id, Species)
    # Normalización de los datos.
    data = (data - data.mean(axis=0)) / data.std(axis=0)
    # Número de dimensiones.
    dim = data.shape[1]
    
    # Número de neuronas.
    """
    Cálculo de neuronas:
        Número de neuronas = 5 * sqrt(n)
        Siguiendo la regla de 5 * sqrt(n) para el número de neuronas.
        Aproximadamente se ocupan 61 neuronas para un dataset de 150 datos.
        Se usará una cuadrícula de 8x8.
    """
    n = 8
    m = 8
elif opc == "2":
    # Lectura de datos.
    df = pd.read_csv("C:/Users/Cristian/Programacion/Python/Redes_Neuronales/bill_authentication.csv")
    # Datos.
    data = df.iloc[:, 0:4].values  # Se eliminan las columnas de identificación. (Class)
    # Normalización de los datos.
    data = (data - data.mean(axis=0)) / data.std(axis=0)
    # Número de dimensiones.
    dim = data.shape[1]
    # Cálculo de neuronas. => 5 * sqrt(n) => 5 * sqrt(1372) => 5 * 37 => 185
    # Cuadrícula ideal (185) =  14x14 (196)
    n = 14
    m = 14

# Solicitar al usuario la tasa de aprendizaje, vecindad inicial y el número de épocas.
tasa_aprendizaje = float(input("Ingrese la tasa de aprendizaje: "))
sigma = float(input("Ingrese la vecindad inicial (sigma): "))
epocas = int(input("Ingrese el número de épocas: "))

if sigma == "":
    sigma = m / 2

# Creación del modelo.
model = SOM(n, m, dim, tasa_aprendizaje=tasa_aprendizaje, sigma=sigma)
# Entrenamiento.
model.train(data, epocas)
# Asignación.
asignaciones = model.asignar(data)
# Mostrar los pesos finales.
print("\nPesos finales:")
for i in range(n):
    for j in range(m):
        print(f"Neurona ({i}, {j}): {model.weights[i, j]}")
# print(model.weights)

plt.ioff()
# plt.pause(5)

if opc == "1":
    # Graficar mapa inicial.
    plt.scatter(data[:, 0], data[:, 1], c=df["Species"].astype("category").cat.codes)
    plt.title("Mapa inicial")
    plt.show()

    # Gráfica del mapa auto-organizado.
    plt.scatter(asignaciones, np.arange(data.shape[0]), c=df["Species"].astype("category").cat.codes)
    plt.xlabel("Neurona")
    plt.ylabel("Dato")
    plt.title("Mapa auto-organizado")
    plt.show()

    # Graficar la malla de neuronas.
    for i in range(n):
        for j in range(m):
            if i > 0:
                plt.plot([model.weights[i, j, 0], model.weights[i - 1, j, 0]], [model.weights[i, j, 1], model.weights[i - 1, j, 1]], c="black")
            if j > 0:
                plt.plot([model.weights[i, j, 0], model.weights[i, j - 1, 0]], [model.weights[i, j, 1], model.weights[i, j - 1, 1]], c="black")
    plt.scatter(model.weights[:, :, 0], model.weights[:, :, 1], c="red")
    plt.title("Malla de neuronas")
    plt.show()
elif opc == "2":
    # Graficar mapa inicial.
    plt.scatter(data[:, 0], data[:, 1], c=df["Class"].astype("category").cat.codes)
    plt.title("Mapa inicial")
    plt.show()
    
    # Gráfica del mapa auto-organizado.
    plt.scatter(asignaciones, np.arange(data.shape[0]), c=df["Class"].astype("category").cat.codes)
    plt.xlabel("Neurona")
    plt.ylabel("Dato")
    plt.title("Mapa auto-organizado")
    plt.show()
    
    # Graficar la malla de neuronas.
    # plt.scatter(data[:, 0], data[:, 1], c=df["Species"].astype("category").cat.codes)
    for i in range(n):
        for j in range(m):
            if i > 0:
                plt.plot([model.weights[i, j, 0], model.weights[i - 1, j, 0]], [model.weights[i, j, 1], model.weights[i - 1, j, 1]], c="black")
            if j > 0:
                plt.plot([model.weights[i, j, 0], model.weights[i, j - 1, 0]], [model.weights[i, j, 1], model.weights[i, j - 1, 1]], c="black")
    plt.scatter(model.weights[:, :, 0], model.weights[:, :, 1], c="red")
    plt.title("Malla de neuronas")
    plt.show()