# UNIVERSIDAD DE COLIMA
# INGENIERÍA EN COMPUTACIÓN INTELIGENTE
# REDES NEURONALES
# 7°D
# AUTOR: CRISTIAN ARMANDO LARIOS BRAVO

# Importación de librerías.
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

#! FUNCIONES DE ACTIVACIÓN

# - 1. Sigmoide (un solor valor)
def sigmoide(x):
    return 1/(1 + np.exp(-x))

# - 1.1. Derivada de la función sigmoide
def sigmoide_derivada(x):
    return x * (1 - x)

# - 2. ReLU
def  relu(x):
    return np.maximum(0, x)

# - 2.1. Derivada de la función ReLU
def relu_derivada(x):
    return np.where(x>=0, 1, 0)

# - 3. Tangente Hiperbólica (tanh)
def tanh(x):
    return np.tanh(x)
# - 3.1. Derivada de la función Tangente Hiperbólica
def tanh_derivada(x):
    return 1 - np.tanh(x)**2

# - Escalon
def escalon(x):
    return np.where(x>=0, 1, 0)

# - 4. Softmax
# def softmax(x):
#     return np.exp(x) / np.sum(np.exp(x), axis=0)
def softmax(x):
    exp_values = np.exp(x - np.max(x))
    return exp_values / np.sum(exp_values, axis=0)


# - 4.1 Derivada de la función Softmax
def softmax_derivada(x):
    s = x.reshape(-1, 1) # Asegurar que es un vector columna
    return np.diagflat(s) - np.dot(s, s.T) # np.dot(s, s.T) es la multiplicación de matrices

#! FeddForward
def FeedForward(Entrada, numCapas, pesos, fnActivacionCapas):
    a = [(Entrada)]
    # a = Entrada.values
        
    for i in range(numCapas -1):
        if (i < numCapas - 1):
            z = np.dot(pesos[i], a[-1])
            # z = np.dot(pesos[i], a[-1]) + biasCapas[i]
            
            if fnActivacionCapas[i] == "1":
                a.append(sigmoide(z))
            elif fnActivacionCapas[i] == "2":
                a.append(relu(z))
            elif fnActivacionCapas[i] == "3":
                a.append(tanh(z))
            elif fnActivacionCapas[i] == "4":
                a.append(softmax(z))
    return a


#? Cargar el dataset.
df = pd.read_csv("C:/Users/Cristian/Programacion/Python/Redes_Neuronales/Iris.csv")
# df = pd.read_csv("C:/Programacion/Python/Redes_Neuronales/Iris.csv")
df = df.drop(['Id'], axis=1)

#? Discretización de los datos[Clases]. (0, 1)
# Cambiar las etiquetas de las especies a números.
df["Species"].unique()
df["Species"].value_counts()
df["Species"] = df["Species"].map(
    {"Iris-setosa": 0,
     "Iris-versicolor": 1,
     "Iris-virginica": 2})
df["Species"].unique()
df["Species"].value_counts()


#? Datos de entrenamiento y prueba.
X = df.copy()
X.drop(['Species'], axis=1, inplace=True)
y = df.copy()
y = y['Species'] 

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

###^ 1.- Definir la topologia de la red neuronal
####&      1.1. Cantidad de capas de la Red Neuronal
# numCapas = int(input("Ingrese la cantidad de capas ocultas de la red neuronal: ")) + 2
# numCapas = 4
numCapas = 3
print(f"La red neuronal tiene {numCapas} capas.")

####&       1.2. Cantidad de Neuronas en cada capa
# numNeuronas = [4, 5, 6, 3]
numNeuronas = [4, 3, 3]
# numNeuronas = []
# for i in range(numCapas):
#     neuronas = int(input(f"Ingrese la cantidad de neuronas en la capa {i+1}:"))
#     numNeuronas.append(neuronas)
print(f"Cantidad de neuronas por capa: {numNeuronas}")

# Bias por cada capa.
np.random.seed(0)
biasCapas = []
for i in range(numCapas):
    bias = np.random.rand(numNeuronas[i])
    biasCapas.append(bias)
print(f"Bias por capa:")
for i in biasCapas:
    print(i)

####&       1.3 Función de Activación en cada capa (ocultas y salida)
# fnActivacionCapas = []
# fnActivacionCapas =['1', '1', '1']
fnActivacionCapas =['1', '4']
# for i in range(numCapas-1):
#     print(f"\nFUNCIONES DE ACTIVACIÓN: \n \
#         1.- Sigmoide\n \
#         2.- ReLU \n \
#         3.- Tangente Hiperbólica.\n \
#         4.- Softmax")
#     fn = input(f"Ingrese la función de activación de la capa {i+2}: ")
#     fnActivacionCapas.append(fn)
print(f"Función de activación en cada capa: {fnActivacionCapas}")


###^ 2.- Implementar FeedForward
####& 2.1 Crear la matriz de pesos
np.random.seed(0)
pesos = []
numCaracteristicas = df.shape[1]-1

# U(-1; 1)
for i in range(numCapas-1):
    w = np.random.uniform(-1, 1, (numNeuronas[i], numNeuronas[i+1]))
    pesos.append(w)

# for i in range(numCapas-1):
#     w = np.random.rand(numNeuronas[i], numNeuronas[i+1])
#     pesos.append(w)

print("Matriz de pesos:")
for i in pesos:
    print(i)
    
####& 2.2 Aplicar FeedForward y obtener una salida

pesosT = []
for i in pesos:
    pesosT.append(i.T)
    
# Error cuadrático medio
def ECM(y, y_pred):
    return np.mean((y - y_pred)**2)

# valoresx = x_train.values[:50, :]
valoresx = x_train.values

num_epocas = 1000
tasa_aprendizaje = 0.5

def backpropagation(a, y_real, pesos, fnActivacionCapas, tasa_aprendizaje):
    deltas = []
    error = a[-1] - y_real
    
    # Calcular delta para la capa de salida
    if fnActivacionCapas[-1] == "1":
        delta = error * sigmoide_derivada(a[-1])
    elif fnActivacionCapas[-1] == "2":
        delta = error * relu_derivada(a[-1])
    elif fnActivacionCapas[-1] == "3":
        delta = error * tanh_derivada(a[-1])
    elif fnActivacionCapas[-1] == "4":
        delta = error * softmax_derivada(a[-1])
    
    deltas.append(delta)
    
    # Propagar hacia atrás
    # Calcular delta para las capas ocultas
    for i in range(len(a)-2, 0, -1):
        if fnActivacionCapas[i-1] == "1":
            delta = np.dot(pesos[i].T, deltas[-1]) * sigmoide_derivada(a[i])
        elif fnActivacionCapas[i-1] == "2":
            delta = np.dot(pesos[i].T, deltas[-1]) * relu_derivada(a[i])
        elif fnActivacionCapas[i-1] == "3":
            delta = np.dot(pesos[i].T, deltas[-1]) * tanh_derivada(a[i])
        elif fnActivacionCapas[i-1] == "4":
            delta = np.dot(pesos[i].T, deltas[-1]) * softmax_derivada(a[i])
        
        deltas.append(delta)
    
    deltas.reverse()

    # Actualizar pesos
    # convertir el vector deltas[i] en un vector columna.
    # convertir el vector a[i] en un vector fila.
    for i in range(len(pesos)):
        pesos[i] -= tasa_aprendizaje * np.dot(deltas[i].reshape(-1, 1), a[i].reshape(1, -1))

    
    return pesos

# Entrenamiento
for epoca in range(num_epocas):
    for i in range(len(valoresx)):
        # Propagación hacia adelante
        salida_obtenida = FeedForward(valoresx[i], numCapas, pesosT, fnActivacionCapas)
        
        # Propagación hacia atrás
        pesosT = backpropagation(salida_obtenida, y_train.values[i], pesosT, fnActivacionCapas, tasa_aprendizaje)
        
        if epoca % 100 == 0:
            total_error = 0
            for i in range(len(valoresx)):
                salida_obtenida = FeedForward(valoresx[i], numCapas, pesosT)
                total_error += ECM(y_train.values[i], salida_obtenida[-1])
            # print(f"Epoca {epoca}, Error Total: {total_error}")

# Prueba
print()
valoresp = x_test.values
total_error = 0
predicciones = []
for i in range(len(valoresp)):
    salida_obtenida = FeedForward(valoresp[i], numCapas, pesosT)
    predicciones.append(np.argmax(salida_obtenida[-1]))
    total_error += ECM(y_test.values[i], salida_obtenida[-1])
    print(f"Salida: {salida_obtenida[-1]}")
    print(f"Predicción: {np.argmax(salida_obtenida[-1])}")
    print(f"Esperado: {y_test.values[i]}")
    print(f"ECM: {ECM(y_test.values[i], salida_obtenida[-1])}")
    
# Exactitud
predicciones = np.array(predicciones)
exactitud = np.mean(predicciones == y_test.values)
print(f"Exactitud del modelo: {exactitud * 100:.2f}%")
# print(f"Exactitud en el conjunto de prueba: {exactitud}")