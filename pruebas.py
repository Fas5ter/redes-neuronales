import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from prettytable import PrettyTable
import os

os.environ["LOKY_MAX_CPU_COUNT"] = "11" 

# Funciones de activación para la capa oculta
def gaussiana(r, varianza):
    return np.exp(-r*2 / (2 * varianza*2))

def multicuadratica(r):
    return np.sqrt(1 + r**2)

def multicuadratica_inversa(r):
    return 1 / np.sqrt(1 + r**2)

# Funciones de activación para la capa de salida
def sigmoide(x):
    return 1 / (1 + np.exp(-x))

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

def tanh(x):
    return np.tanh(x)

# Cálculo de la varianza
def calcular_varianzas_por_centroide(centroides):
    varianzas = []
    for i, centroide in enumerate(centroides):
        # Calcular distancias a otros centroides
        distancias = [np.linalg.norm(centroide - otros) for j, otros in enumerate(centroides) if i != j]
        # Ordenar distancias y calcular la media de las dos más cercanas
        distancias.sort()
        varianza = np.sqrt(distancias[0] * distancias[1])
        varianzas.append(varianza)
    return varianzas

# Seleccionar las funciones de activación
def seleccionar_Funciones_Activacion():
    print("\nElige la función de activación para la capa oculta:")
    print("     1. Gaussiana")
    print("     2. Multicuadrática")
    print("     3. Multicuadrática Inversa")
    eleccion_oculta = input("Función de activación para la capa oculta: ")

    if eleccion_oculta == "1":
        funcion_oculta = gaussiana
    elif eleccion_oculta == "2":
        funcion_oculta = multicuadratica
    elif eleccion_oculta == "3":
        funcion_oculta = multicuadratica_inversa
    else:
        print("Selección inválida, usando Gaussiana por defecto.")
        funcion_oculta = gaussiana

    print("\nElige la función de activación para la capa de salida:")
    print("     1. Sigmoide")
    print("     2. Softmax")
    print("     3. Tangente Hiperbólica")
    eleccion_salida = input("Función de activación para la capa de salida: ")

    if eleccion_salida == "1":
        funcion_salida = sigmoide
    elif eleccion_salida == "2":
        funcion_salida = softmax
    elif eleccion_salida == "3":
        funcion_salida = tanh
    else:
        print("Selección inválida, usando Sigmoide por defecto.")
        funcion_salida = sigmoide

    return funcion_oculta, funcion_salida

# Función para cargar y preprocesar los datos
def cargar_preprocesar_datos(df):
    # Verificar si hay Id y extraer las entradas
    inputs = np.array(df.iloc[:, 1:-1]) if df.columns[0] == "Id" else np.array(df.iloc[:, :-1])
    
    # Normalización de los inputs
    escala = StandardScaler()
    inputs = escala.fit_transform(inputs)
    
    # Mapear las clases
    clases_esperadas = pd.factorize(df.iloc[:, -1])[0]
    
    # Codificación one-hot
    num_clases = len(np.unique(clases_esperadas))
    clases_esperadas_one_hot = np.eye(num_clases)[clases_esperadas]
    
    # Dividir en conjunto de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(inputs, clases_esperadas_one_hot, test_size=0.2)

    return X_train, X_test, y_train, y_test

# Función para aplicar la capa oculta RBF
def capa_oculta_rbf(X, centroides, funcion_activacion, varianzas=None):
    activaciones_ocultas = np.zeros((X.shape[0], len(centroides)))

    for i, centroide in enumerate(centroides):
        # Calcular distancias entre los inputs y los centroides
        distancias = np.linalg.norm(X - centroide, axis=1)
        # Si la función es Gaussiana, usamos la varianza
        if varianzas is not None and funcion_activacion == gaussiana:
            activaciones_ocultas[:, i] = funcion_activacion(distancias, varianzas[i])
        else:  # Para otras funciones que no requieran varianza
            activaciones_ocultas[:, i] = funcion_activacion(distancias)

    return activaciones_ocultas

# Función para entrenar el modelo RBF
def entrenar_rbf(X_train, y_train, centroides, funcion_oculta, funcion_salida, tasa_aprendizaje, epocas, varianzas):
    
    # Activaciones de la capa oculta
    activaciones_ocultas = capa_oculta_rbf(X_train, centroides, funcion_oculta, varianzas)
    
    # Inicializar pesos aleatorios para la capa de salida
    pesos_salida = np.random.randn(y_train.shape[1], activaciones_ocultas.shape[1])

    # Lista de errores por época
    errores_totales = [] 

    # Entrenamiento
    for epoca in range(epocas):
        # Calcular salidas para las activaciones ocultas
        salidas = funcion_salida(np.dot(pesos_salida, activaciones_ocultas.T)).T 
        # Calcular el error
        errores = salidas - y_train 
        # Actualizar pesos (descenso del gradiente)
        pesos_salida -= tasa_aprendizaje * np.dot(errores.T, activaciones_ocultas) 

        # Calcular el error cuadrático medio para la época actual y almacenarlo
        error_total = np.mean(np.square(errores))
        errores_totales.append(error_total)

    # Graficar el error total por época
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, epocas + 1), errores_totales, linestyle='-')
    plt.title('Error por Época')
    plt.xlabel('Épocas')
    plt.ylabel('Error Cuadrático Medio (MSE)')
    plt.grid(True)
    plt.show()

    return pesos_salida

# Función para evaluar el modelo RBF
def evaluar_rbf(X_test, y_test, centroides, funcion_oculta, funcion_salida, pesos_salida, varianzas):
    # Activaciones de la capa oculta
    activaciones_ocultas = capa_oculta_rbf(X_test, centroides, funcion_oculta, varianzas)
    clases_generadas = []

    # Clasificación 
    for i in range(X_test.shape[0]):
        salida_oculta = activaciones_ocultas[i]
        salida = funcion_salida(np.dot(pesos_salida, salida_oculta))
        clases_generadas.append(np.argmax(salida))

    # Precisión y Error Cuadrático Medio
    precision = np.mean(np.argmax(y_test, axis=1) == clases_generadas)
    mse = np.mean((y_test - np.eye(y_test.shape[1])[clases_generadas])**2)
    
    return precision, mse

# Función main
def main():
    # Cargar y preprocesar los datos
    dataset = input("Nombre del archivo CSV: ")
    df = pd.read_csv(dataset)
    X_train, X_test, y_train, y_test = cargar_preprocesar_datos(df)

    # Pedir número de neuronas en la capa oculta
    num_neuronas = int(input("\nNúmero de neuronas en la capa oculta (centroides): "))
    
    # Seleccionar funciones de activación
    funcion_oculta, funcion_salida = seleccionar_Funciones_Activacion()

    # Pedir tasa de aprendizaje y número de épocas
    tasa_aprendizaje = float(input("\nTasa de Aprendizaje: "))
    epocas = int(input("\nNúmero de Épocas: "))

    # Crear los centroides usando KMeans
    kmeans = KMeans(n_clusters=num_neuronas)
    kmeans.fit(X_train)
    centroides = kmeans.cluster_centers_

    # Calcular las varianzas
    varianzas = calcular_varianzas_por_centroide(centroides)

    # Entrenar el modelo pasando las varianzas calculadas
    pesos_salida = entrenar_rbf(X_train, y_train, centroides, funcion_oculta, funcion_salida, tasa_aprendizaje, epocas, varianzas)

    # Evaluar el modelo pasando las varianzas calculadas
    precision, mse = evaluar_rbf(X_test, y_test, centroides, funcion_oculta, funcion_salida, pesos_salida, varianzas)
    
    # Mostrar resultados
    print(f"\nPrecisión: {precision * 100:.2f}%")
    print(f"Error Cuadrático Medio: {mse:.4f}")
    
    # Mostrar tabla con clases esperadas y generadas
    tabla = PrettyTable()
    # tabla.field_names = ["Clases Esperadas", "Clases Generadas"]
    # for i in range(len(y_test)):
    #     tabla.add_row([np.argmax(y_test[i]), np.argmax(y_test[i])])
    # print(tabla)

if __name__ == "__main__":
    main()