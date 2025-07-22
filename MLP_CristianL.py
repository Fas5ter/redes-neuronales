
#! EL BUENO
# UNIVERSIDAD DE COLIMA
# INGENIERÍA EN COMPUTACIÓN INTELIGENTE
# REDES NEURONALES
# 7°D
# AUTOR: CRISTIAN ARMANDO LARIOS BRAVO


# Importación de librerías.
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

#! FUNCIONES DE ACTIVACIÓN

# 1. Sigmoide
def sigmoide(x):
    return 1 / (1 + np.exp(-x))

# 1.1. Derivada de la función sigmoide
def sigmoide_derivada(x):
    return x * (1 - x)

# 2. ReLU
def relu(x):
    return np.maximum(0, x)

# 2.1. Derivada de la función ReLU
def relu_derivada(x):
    return np.where(x > 0, 1, 0)

# 3. Tangente Hiperbólica (tanh)
def tanh_func(x):
    return np.tanh(x)

# 3.1. Derivada de la función Tangente Hiperbólica
def tanh_derivada(x):
    return 1 - np.tanh(x)**2

# 4. Softmax
def softmax(x):
    exp_values = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_values / np.sum(exp_values, axis=1, keepdims=True)

#! Función de pérdida: Entropía Cruzada
def entropia_cruzada(y_real, y_pred):
    m = y_real.shape[0]
    # Añadir 1e-15 para evitar log(0)
    log_likelihood = -np.log(y_pred[range(m), y_real] + 1e-15)
    return np.sum(log_likelihood) / m

#! Función para convertir etiquetas a One-Hot Encoding
def one_hot(y, num_clases):
    one_hot_matrix = np.zeros((y.size, num_clases))
    one_hot_matrix[np.arange(y.size), y] = 1
    return one_hot_matrix

#! FeedForward + Backpropagation
def backpropagation(X, y_real, num_capas, pesos, sesgos, funciones_activacion, tasa_aprendizaje=0.01):
    #* Propagación hacia adelante
    activaciones = [X]
    z_values = []
    
    for i in range(num_capas - 1):
        z = np.dot(activaciones[-1], pesos[i]) + sesgos[i]
        z_values.append(z)
        
        if funciones_activacion[i] == "1":
            a = sigmoide(z)
        elif funciones_activacion[i] == "2":
            a = relu(z)
        elif funciones_activacion[i] == "3":
            a = tanh_func(z)
        elif funciones_activacion[i] == "4":
            a = softmax(z)
        activaciones.append(a)

    #* Retropropagación
    deltas = []
    m = X.shape[0]
    
    # Convertir y_real a One-Hot Encoding
    y_one_hot = one_hot(y_real, pesos[-1].shape[1])
    
    # Error en la capa de salida
    error_salida = activaciones[-1] - y_one_hot
    deltas.append(error_salida)
    
    # Retropropagación para capas ocultas
    for i in reversed(range(num_capas - 2)):
        if funciones_activacion[i] == "1":
            delta = np.dot(deltas[-1], pesos[i+1].T) * sigmoide_derivada(activaciones[i+1])
        elif funciones_activacion[i] == "2":
            delta = np.dot(deltas[-1], pesos[i+1].T) * relu_derivada(z_values[i])
        elif funciones_activacion[i] == "3":
            delta = np.dot(deltas[-1], pesos[i+1].T) * tanh_derivada(activaciones[i+1])
        elif funciones_activacion[i] == "4":
            # Para softmax con entropía cruzada, la derivada ya está simplificada
            delta = np.dot(deltas[-1], pesos[i+1].T)  # No multiplicar por softmax_derivada
        deltas.append(delta)
    
    deltas.reverse()
    
    # Actualización de pesos y sesgos
    for i in range(num_capas - 1):
        pesos[i] -= tasa_aprendizaje * np.dot(activaciones[i].T, deltas[i]) / m
        sesgos[i] -= tasa_aprendizaje * np.sum(deltas[i], axis=0, keepdims=True) / m
    
    return pesos, sesgos

#? Cargar el dataset.
df = pd.read_csv("C:/Users/Cristian/Programacion/Python/Redes_Neuronales/Iris.csv")
# df = pd.read_csv("C:/Programacion/Python/Redes_Neuronales/Iris.csv")
df = df.drop(['Id'], axis=1)

#? Discretización de los datos [Clases]. (0, 1, 2)
df["Species"] = df["Species"].map(
    {"Iris-setosa": 0,
     "Iris-versicolor": 1,
     "Iris-virginica": 2}
)

#? Datos de entrenamiento y prueba.
X = df.drop(['Species'], axis=1).values
y = df['Species'].values

# Normalizar los datos
# escalador = StandardScaler()
# X = escalador.fit_transform(X)

# Dividir en entrenamiento y prueba
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

###^ 1.- Definir la topología de la red neuronal
####&      1.1. Cantidad de capas de la Red Neuronal
num_capas = int(input("Ingrese la cantidad de capas (incluyendo capa de entrada y salida): "))
print(f"La red neuronal tiene {num_capas} capas.")

####&       1.2. Cantidad de Neuronas en cada capa
num_neuronas = []
for i in range(num_capas):
    if i == 0:
        neuronas = x_train.shape[1]
        num_neuronas.append(neuronas)
    elif i == num_capas -1:
        neuronas = len(np.unique(y_train))
        num_neuronas.append(neuronas)
    else:
        neuronas = int(input(f"Ingrese la cantidad de neuronas en la capa {i+1}: "))
        num_neuronas.append(neuronas)
print(f"Cantidad de neuronas por capa: {num_neuronas}")

####&       1.3 Función de Activación en cada capa (ocultas y salida)
funciones_activacion = []
for i in range(num_capas -1):
    print(f"\nFUNCIONES DE ACTIVACIÓN para la capa {i+1} a {i+2}:")
    print("1.- Sigmoide")
    print("2.- ReLU")
    print("3.- Tangente Hiperbólica")
    print("4.- Softmax (solo para la capa de salida)")
    fn = input(f"Ingrese la función de activación para la capa {i+2}: ")
    while fn not in ['1', '2', '3', '4']:
        print("Entrada inválida. Por favor, ingrese 1, 2, 3 o 4.")
        fn = input(f"Ingrese la función de activación para la capa {i+2}: ")
    funciones_activacion.append(fn)
print(f"Funciones de activación por capa: {funciones_activacion}")

# 1.- Inicializar los pesos y sesgos con valores aleatorios pequeños.
pesos = []
sesgos = []
for i in range(num_capas -1):
    w = np.random.uniform(-1, 1, (num_neuronas[i], num_neuronas[i+1]))
    b = np.zeros((1, num_neuronas[i+1]))
    pesos.append(w)
    sesgos.append(b)

print("Pesos y sesgos iniciales:")
print(f"{pesos}\n")
print(f"{sesgos}\n")


# Parámetros de entrenamiento
# epocas = 1000
# tasa_aprendizaje = 0.01
while True:
    epocas = str(input("Ingrese la cantidad de épocas: "))
    tasa_aprendizaje = str(input("Ingrese la tasa de aprendizaje: "))
    
    # Validar que sean números
    if epocas.isnumeric() and tasa_aprendizaje.replace(".", "", 1).isdigit():
        epocas = int(epocas)
        tasa_aprendizaje = float(tasa_aprendizaje)
        break
    else:
        print("Entrada inválida. Por favor, ingrese un número entero para las épocas y un número decimal para la tasa de aprendizaje.")

print()

# Entrenamiento
for epoca in range(epocas):
    for i in range(x_train.shape[0]):
        # Propagación hacia atrás para cada muestra
        X_sample = x_train[i].reshape(1, -1) # Convertir a matriz fila
        y_sample = y_train[i]
        pesos, sesgos = backpropagation(X_sample, y_sample, num_capas, pesos, sesgos, funciones_activacion, tasa_aprendizaje)
    
    # Imprimir la pérdida cada 100 épocas
    if (epoca+1) % 100 == 0:
        # Propagación hacia adelante para todo el conjunto de entrenamiento
        activaciones = [x_train]
        for j in range(num_capas -1):
            z = np.dot(activaciones[-1], pesos[j]) + sesgos[j]
            if funciones_activacion[j] == "1":
                a = sigmoide(z)
            elif funciones_activacion[j] == "2":
                a = relu(z)
            elif funciones_activacion[j] == "3":
                a = tanh_func(z)
            elif funciones_activacion[j] == "4":
                a = softmax(z)
            activaciones.append(a)
        y_pred = activaciones[-1]
        # Cálculo de la función de perdida
        pérdida = entropia_cruzada(y_train, y_pred)
        print(f"Época {epoca+1}, Pérdida: {pérdida}")

print()

# Prueba de la red neuronal
y_pred_prueba = []
for i in range(x_test.shape[0]):
    a = x_test[i].reshape(1, -1)
    for j in range(num_capas -1):
        z = np.dot(a, pesos[j]) + sesgos[j]
        if funciones_activacion[j] == "1":
            a = sigmoide(z)
        elif funciones_activacion[j] == "2":
            a = relu(z)
        elif funciones_activacion[j] == "3":
            a = tanh_func(z)
        elif funciones_activacion[j] == "4":
            a = softmax(z)
    predicción = np.argmax(a, axis=1)[0]
    y_pred_prueba.append(predicción)
    print(f"Predicción: {predicción} - Real: {y_test[i]}")

# Calcular exactitud
exactitud = np.mean(np.array(y_pred_prueba) == y_test)
print(f"Exactitud en el conjunto de prueba: {exactitud * 100:.2f}%")

print("\n\n")

print("PESOS Y SESGOS FINALES")
print(f"{pesos}\n")
print(f"{sesgos}\n")