
# UNIVERSIDAD DE COLIMA
# INGENIERÍA EN COMPUTACIÓN INTELIGENTE
# REDES NEURONALES
# 7°D
# AUTOR: CRISTIAN ARMANDO LARIOS BRAVO
# Redes Neuronales de Base Radial o RBF.

# Importación de librerías.
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split

#! FUNCIONES DE ACTIVACIÓN

#* Capa oculta
def f_activacion(opc, x, c, var):
    if opc == "1":
        return np.exp(-np.linalg.norm(x - c)**2 / (2 * var**2)) # Gausiana
    elif opc == "2":
        return np.sqrt(1 + np.linalg.norm(x - c)**2 / var**2) # Multicuadrática
    elif opc == "3":
        return 1 / np.sqrt(1 + np.linalg.norm(x - c)**2 / var**2) # Multicuadrática inversa

#* Capa de salida
def f_salida(opc, x):
    if opc == "1":
        return 1 / (1 + np.exp(-x)) # Sigmoide
    elif opc == "2":
        exp_values = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_values / np.sum(exp_values, axis=1, keepdims=True) # Softmax
    elif opc == "3":
        return np.tanh(x) # Tangente hiperbólica

def error_cuadratico_medio(y_real, y_pred):
    return np.mean((y_real - y_pred)**2)


encoder = LabelEncoder()

opc = str(input("¿Cuál dataset desea utilizar?\n1. Iris.csv\n2. Bill_authentication.csv\nIngrese la opción: "))
# neuronasO = int(input("Ingrese el número de neuronas en la capa oculta: "))
tasa_aprendizaje = float(input("Ingrese la tasa de aprendizaje: "))
epocas = int(input("Ingrese el número de épocas: "))
f_capa_oculta = str(input("¿Qué función de activación desea utilizar en la capa oculta?\n1. Gausiana\n2. Multicuadrática\n3. Multicuadrática inversa\nIngrese la opción: "))
f_capa_salida = str(input("¿Qué función de activación desea utilizar en la capa de salida?\n1. Sigmoide\n2. Softmax\n3. Tangente hiperbólica\nIngrese la opción: "))

if opc == "1":
    np.random.seed(2)
    #? Cargar el dataset.
    df = pd.read_csv("C:/Users/Cristian/Programacion/Python/Redes_Neuronales/Iris.csv")
    # df = pd.read_csv("C:/Programacion/Python/Redes_Neuronales/Iris.csv")
    df = df.drop(['Id'], axis=1)

    #? Discretización de los datos[Clases]. (0, 1, 2)
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
    

    x_train, x_test, y_train, y_test = train_test_split(X, y ,test_size=0.2, random_state=42)

    # Aplicación de KMeans.
    kmeans = KMeans(n_clusters=3, random_state=0).fit(x_train)
    print(f"\nCentroides:\n{kmeans.cluster_centers_}")

    neuronasO = 3 # número de neuronas en la capa de salida
    neuronasS = 3 # número de neuronas en la capa de salida

    # Seudosalidas (3x120)
    Z = np.zeros((neuronasO, x_train.shape[0]))

    # Inicialización de los pesos (-1, 1).
    pesos = np.random.uniform(-1, 1, (neuronasS, neuronasO))

    # Valores deseados
    valor_deseado = OneHotEncoder().fit_transform(y_train.values.reshape(-1, 1)).toarray()

    # Cálculo de las activaciones de la capa oculta
    var = 1  # Varianza fija
    for i in range(neuronasO):
        for j in range(x_train.shape[0]):
            Z[i, j] = f_activacion(f_capa_oculta, x_train.iloc[j], kmeans.cluster_centers_[i], var)

    # Ajuste de pesos usando descenso de gradiente
    for epoca in range(epocas):
        # Calcular la salida para todos los datos de entrenamiento
        salida = f_salida(f_capa_salida, np.dot(Z.T, pesos.T))
        error_total = error_cuadratico_medio(valor_deseado, salida) 
        
        # Actualización de pesos (regla delta)
        error = valor_deseado - salida
        pesos += tasa_aprendizaje * np.dot(error.T, Z.T)
        # if error_total < 0.001:
        #     break

    # Evaluación del modelo
    Z_test = np.zeros((neuronasO, x_test.shape[0]))
    for i in range(neuronasO):
        for j in range(x_test.shape[0]):
            Z_test[i, j] = f_activacion(f_capa_oculta, x_test.iloc[j], kmeans.cluster_centers_[i], var)

    # Predicción para los datos de prueba
    salida_test = f_salida(f_capa_salida, np.dot(Z_test.T, pesos.T))
    predicciones = np.argmax(salida_test, axis=1)

    # Exactitud
    exactitud = np.mean(predicciones == y_test)
    print(f"\nExactitud en datos de prueba: {exactitud * 100:.2f}%\n")
    print(f"Pesos finales:\n{pesos}")
    
if opc == "2":
    
    np.random.seed(0)
    #? Cargar el dataset.
    df = pd.read_csv("C:/Users/Cristian/Programacion/Python/Redes_Neuronales/Bill_authentication.csv")
    # df = pd.read_csv("C:/Programacion/Python/Redes_Neuronales/Bill_authentication.csv")

    #? Datos de entrenamiento y prueba.
    X = df.copy()
    X.drop(['Class'], axis=1, inplace=True)
    y = df.copy()
    y = y['Class']

    # x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=165)

    # Aplicación de KMeans.
    kmeans = KMeans(n_clusters=2, random_state=0).fit(x_train)
    print(f"Centroides:\n{kmeans.cluster_centers_}")

    neuronasO = 2 # número de neuronas en la capa oculta
    neuronasS = 2 # número de neuronas en la capa de salida

    # Seudosalidas (2x960)
    Z = np.zeros((neuronasO, x_train.shape[0]))

    # Inicialización de los pesos (-1, 1).
    pesos = np.random.uniform(-1, 1, (neuronasS, neuronasO))

    # Valores deseados
    valor_deseado = OneHotEncoder().fit_transform(y_train.values.reshape(-1, 1)).toarray()

    # Cálculo de las activaciones de la capa oculta usando la función gaussiana
    var = 1 
    for i in range(neuronasO):
        for j in range(x_train.shape[0]):
            Z[i, j] = f_activacion(f_capa_oculta, x_train.iloc[j], kmeans.cluster_centers_[i], var)

    # Ajuste de pesos usando descenso de gradiente
    for epoca in range(epocas):
        # Calcular la salida para todos los datos de entrenamiento
        salida = f_salida(f_capa_salida, np.dot(Z.T, pesos.T))
        error_total = error_cuadratico_medio(valor_deseado, salida) 
        
        # Actualización de pesos (regla delta)
        error = valor_deseado - salida
        pesos += tasa_aprendizaje * np.dot(error.T, Z.T)
        error_total = error_cuadratico_medio(valor_deseado, salida)
        
        # if (epoca+1) % 100 == 0:
        #     print(f"Época {epoca+1}, Pérdida: {error_total}")
        # if error_total < 0.01:
        #     break
    
    # Evaluación del modelo
    Z_test = np.zeros((neuronasO, x_test.shape[0]))
    for i in range(neuronasO):
        for j in range(x_test.shape[0]):
            Z_test[i, j] = f_activacion(f_capa_oculta, x_test.iloc[j], kmeans.cluster_centers_[i], var)

    # Predicción para los datos de prueba
    salida_test = f_salida(f_capa_salida, np.dot(Z_test.T, pesos.T))
    predicciones = np.argmax(salida_test, axis=1)

    # Exactitud
    exactitud = np.mean(predicciones == y_test)
    print(f"\nExactitud en datos de prueba: {exactitud * 100:.2f}%\n")
    print(f"Pesos finales:\n{pesos}")