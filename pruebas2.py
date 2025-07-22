import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from prettytable import PrettyTable

# Funciones de activación para la capa oculta
def gaussiana(r, varianza):
    return np.exp(-r**2 / (2 * varianza**2))

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
        distancias = [np.linalg.norm(centroide - otros) for j, otros in enumerate(centroides) if i != j]
        distancias.sort()
        varianza = np.sqrt(distancias[0] * distancias[1])
        varianzas.append(varianza)
    return varianzas

# Función para aplicar la capa oculta RBF
def capa_oculta_rbf(X, centroides, funcion_activacion, varianzas=None):
    activaciones_ocultas = np.zeros((X.shape[0], len(centroides)))
    for i, centroide in enumerate(centroides):
        distancias = np.linalg.norm(X - centroide, axis=1)
        if varianzas is not None and funcion_activacion == gaussiana:
            activaciones_ocultas[:, i] = funcion_activacion(distancias, varianzas[i])
        else:
            activaciones_ocultas[:, i] = funcion_activacion(distancias)
    return activaciones_ocultas

# Función para entrenar el modelo RBF
def entrenar_rbf(X_train, y_train, centroides, funcion_oculta, funcion_salida, tasa_aprendizaje, epocas, varianzas):
    activaciones_ocultas = capa_oculta_rbf(X_train, centroides, funcion_oculta, varianzas)
    pesos_salida = np.random.uniform(-1, 1, (y_train.shape[1], activaciones_ocultas.shape[1]))
    errores_totales = []

    for epoca in range(epocas):
        salidas = funcion_salida(np.dot(pesos_salida, activaciones_ocultas.T)).T
        errores = salidas - y_train
        pesos_salida -= tasa_aprendizaje * np.dot(errores.T, activaciones_ocultas)
        error_total = np.mean(np.square(errores))
        errores_totales.append(error_total)

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
    activaciones_ocultas = capa_oculta_rbf(X_test, centroides, funcion_oculta, varianzas)
    clases_generadas = []
    for i in range(X_test.shape[0]):
        salida_oculta = activaciones_ocultas[i]
        salida = funcion_salida(np.dot(pesos_salida, salida_oculta))
        clases_generadas.append(np.argmax(salida))
    precision = np.mean(np.argmax(y_test, axis=1) == clases_generadas)
    mse = np.mean((y_test - np.eye(y_test.shape[1])[clases_generadas])**2)
    return precision, mse

# Función principal
def main():
    dataset = input("Nombre del archivo CSV: ")
    df = pd.read_csv(dataset)
    
    X = np.array(df.iloc[:, :-1])
    y = pd.factorize(df.iloc[:, -1])[0]
    y = OneHotEncoder().fit_transform(y.reshape(-1, 1)).toarray()
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    num_neuronas = int(input("Número de neuronas en la capa oculta (centroides): "))
    kmeans = KMeans(n_clusters=num_neuronas)
    kmeans.fit(X_train)
    centroides = kmeans.cluster_centers_
    varianzas = calcular_varianzas_por_centroide(centroides)
    
    tasa_aprendizaje = float(input("Tasa de Aprendizaje: "))
    epocas = int(input("Número de Épocas: "))

    funcion_oculta = gaussiana
    funcion_salida = softmax
    
    pesos_salida = entrenar_rbf(X_train, y_train, centroides, funcion_oculta, funcion_salida, tasa_aprendizaje, epocas, varianzas)
    
    precision, mse = evaluar_rbf(X_test, y_test, centroides, funcion_oculta, funcion_salida, pesos_salida, varianzas)
    print(f"\nPrecisión: {precision * 100:.2f}%")
    print(f"Error Cuadrático Medio: {mse:.4f}")

    tabla = PrettyTable()
    tabla.field_names = ["Clases Esperadas", "Clases Generadas"]
    for i in range(len(y_test)):
        tabla.add_row([np.argmax(y_test[i]), np.argmax(y_test[i])])
    print(tabla)

if __name__ == "__main__":
    main()
