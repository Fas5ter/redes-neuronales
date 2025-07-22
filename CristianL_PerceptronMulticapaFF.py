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
#   - Sigmoide (un solor valor)
def sigmoide(x):
    return 1/(1 + np.exp(-x))

def sigmoide2(x):
    for i in range(len(x)):
        x[i] = 1 / (1 + np.exp(-x[i]))

# - ReLU
def  relu(x):
    return np.maximum(0, x)

# - Tangente Hiperbólica (tanh)
def tanh(x):
    return np.tanh(x)

# - Escalon
def escalon(x):
    return np.where(x>=0, 1, 0)

# - Softmax
def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)

# - Mish
def mish(x):
    return x * np.tanh(np.log(1 + np.exp(x)))

#! FeedForward
def FeedForward(Entrada, numCapas, pesos):
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
                a.append(mish(z))
            elif fnActivacionCapas[i] == "5":
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
numCapas = int(input("Ingrese la cantidad de capas ocultas de la red neuronal: ")) + 2
# numCapas = 3
print(f"La red neuronal tiene {numCapas} capas.")

####&       1.2. Cantidad de Neuronas en cada capa
# numNeuronas = [4, 3, 3]
numNeuronas = []
for i in range(numCapas):
    neuronas = int(input(f"Ingrese la cantidad de neuronas en la capa {i+1}:"))
    numNeuronas.append(neuronas)
print(f"Cantidad de neuronas por capa: {numNeuronas}")

# Bias por cada capa.
np.random.seed(0)
# biasCapas = []
# for i in range(numCapas):
#     bias = np.random.rand(numNeuronas[i])
#     biasCapas.append(bias)
# print(f"Bias por capa:")
# for i in biasCapas:
#     print(i)

####&       1.3 Función de Activación en cada capa (ocultas y salida)
fnActivacionCapas = []
# fnActivacionCapas =['1', '1']
for i in range(numCapas-1):
    print(f"\nFUNCIONES DE ACTIVACIÓN: \n \
        1.- Sigmoide\n \
        2.- ReLU \n \
        3.- Tangente Hiperbólica.\n \
        4.- Mish.\n \
        5.- Softmax")
    fn = input(f"Ingrese la función de activación de la capa {i+2}: ")
    fnActivacionCapas.append(fn)
print(f"Función de activación en cada capa: {fnActivacionCapas}")


###^ 2.- Implementar FeedForward
####& 2.1 Crear la matriz de pesos
np.random.seed(0)
pesos = []
numCaracteristicas = df.shape[1]-1
for i in range(numCapas-1):
    w = np.random.rand(numNeuronas[i], numNeuronas[i+1])
    pesos.append(w)

print("Matriz de pesos:")
for i in pesos:
    print(i)
    
####& 2.2 Aplicar FeedForward y obtener una salida

pesosT = []
for i in pesos:
    pesosT.append(i.T)
    
# valoresx = x_train.values[:50, :]
valoresx = x_train.values
errores = 0
for i in range(len(valoresx)):
    x = FeedForward(x_train.values[i], numCapas, pesosT)
    indiceAlto = np.argmax(x[-1])
    print(f"Salida: {x[-1]}")
    print(f"Clase Obtenida (Yob): {indiceAlto}")
    print(f"Clase Esperada (Yd): {y_train.values[i]}")
    print()