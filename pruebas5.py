
# UNIVERSIDAD DE COLIMA
# INGENIERÍA EN COMPUTACIÓN INTELIGENTE
# REDES NEURONALES
# 7°D
# AUTOR: CRISTIAN ARMANDO LARIOS BRAVO
# Redes Neuronales de Base Radial o RBF.

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from scipy.spatial.distance import cdist

np.random.seed(0)

class RedRBF:
    def __init__(self, neuronasO, tasa_aprendizaje, epocas, funcion_oculta, funcion_salida):
        self.neuronasO = neuronasO
        self.tasa_aprendizaje = tasa_aprendizaje
        self.epocas = epocas
        self.funcion_oculta = funcion_oculta
        self.funcion_salida = funcion_salida
        self.centroides = None
        self.varianzas = None
        self.pesos_salida = None
        if self.neuronasO <= 2:
            self.neuronasO = 4
        self.errores_totales = None

    #! Funciones de activación capa oculta
    @staticmethod
    def gaussiana(x, varianza):
        return np.exp(-x**2 / (2 * varianza**2))

    @staticmethod
    def multicuadratica(x):
        return np.sqrt(1 + x**2)

    @staticmethod
    def multicuadratica_inversa(x):
        return 1 / np.sqrt(1 + x**2)

    # ! Funciones de activación capa de salida
    @staticmethod
    def sigmoide(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def softmax(x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)

    @staticmethod
    def tanh(x):
        return np.tanh(x)

    #* Calculo de varianzas
    def calcular_varianzas(self):
        """
        Calcula la varianza de cada cluster utilizando la distancia entre cada centroide
        y sus dos vecinos más cercanos.

        Returns:
            list: Lista de varianzas calculadas para cada centroide.
        """
        # Validación de existencia de los centroides
        if self.centroides is None:
            raise ValueError("Los centroides no están inicializados. Asegúrate de que se hayan calculado antes de llamar a este método.")
        
        # Matriz de distancias entre todos los pares de centroides
        distancias = cdist(self.centroides, self.centroides)
        
        # Establecer la diagonal en infinito para evitar distancia cero de un centroide a sí mismo
        np.fill_diagonal(distancias, np.inf)
        
        varianzas = []
        for dist in distancias:
            # Ordenar las distancias del centroide actual y tomar las dos más pequeñas
            dist_menores = np.sort(dist)[:2]
            varianza = np.sqrt(dist_menores[0] * dist_menores[1]) # Media geométrica de las dos distancias más pequeñas
            varianzas.append(varianza) # Añadir varianza a la lista
        
        return varianzas

    #* Calculo de las activaciones de la capa oculta
    def activacionesO(self, X):
        """
        Calcula las activaciones de la capa oculta RBF para cada entrada en X.

        Args:
            X (np.ndarray): Datos de entrada con tamaño (num_muestras, num_features).
        
        Returns:
            np.ndarray: Matriz de activaciones de la capa oculta con tamaño (num_muestras, num_centroides).
        """
        # Validación de existencia de los centroides
        if self.centroides is None:
            raise ValueError("Los centroides no están inicializados.")

        # Inicializar matriz de activaciones
        activacionesO = np.zeros((X.shape[0], len(self.centroides)))
        
        for i, centroide in enumerate(self.centroides):
            # Calcular distancia de cada muestra al centroide i
            distancias = np.linalg.norm(X - centroide, axis=1)
            
            # Aplicar función de activación
            if self.varianzas is not None and self.funcion_oculta == self.gaussiana:
                activacionesO[:, i] = self.funcion_oculta(distancias, self.varianzas[i])
            else:
                # Si no se han calculado las varianzas, se asume que son 1
                activacionesO[:, i] = self.funcion_oculta(distancias)
        
        return activacionesO

    #* Asignar funciones de activación
    def asignar_funciones(self, f_capa_oculta, f_capa_salida):
        match f_capa_oculta:
            case "1":
                self.funcion_oculta = self.gaussiana
            case "2":
                self.funcion_oculta = self.multicuadratica
            case "3":
                self.funcion_oculta = self.multicuadratica_inversa

        match f_capa_salida:
            case "1":
                self.funcion_salida = self.sigmoide
            case "2":
                self.funcion_salida = self.softmax
            case "3":
                self.funcion_salida = self.tanh

    #* Entrenamiento del modelo
    def entrenar(self, X_train, y_train):
        """
        Función para entrenar el modelo RBF
        
        Args:
            X_train (np.ndarray): Datos de entrenamiento
            y_train (np.ndarray): Datos de salida
        
        Returns:
            np.ndarray: Retorna los pesos de salida
        """
        self.centroides = KMeans(n_clusters=self.neuronasO).fit(X_train).cluster_centers_
        self.varianzas = self.calcular_varianzas()

        activacionesO = self.activacionesO(X_train)
        # Inicialización de los pesos de salida
        self.pesos_salida = np.random.uniform(-1, 1, (y_train.shape[1], activacionesO.shape[1]))
        print(f"\nPesos iniciales:\n{self.pesos_salida}\n")
        
        errores_totales = []

        for epoca in range(self.epocas):
            salidas = self.funcion_salida(np.dot(self.pesos_salida, activacionesO.T)).T # Salida de la red
            errores = salidas - y_train # Error de salida
            self.pesos_salida -= self.tasa_aprendizaje * np.dot(errores.T, activacionesO) # Regla delta
            errorT = np.mean(np.square(errores)) # Error total
            errores_totales.append(errorT)
            
            # Imprimir error cada 100 épocas
            if (epoca + 1) % 100 == 0:
                print(f"Época {epoca + 1}, Error total: {errorT:.4f}")

            # Para detener el entrenamiento si el error es menor a 0.05
            if errorT < 0.05:
                break
        self.errores_totales = errores_totales

    #* Evaluación del modelo
    def evaluar(self, X_test, y_test):
        """
        Función para evaluar el modelo RBF
        
        Args:
            X_test (ndarray): Datos de prueba
            y_test (ndarray): Datos de salida
            
        Returns:
            precision (float64): Precisión del modelo
            mse (float64): Error cuadrático medio
            clases_generadas (list): Clases generadas
        """
        activacionesO = self.activacionesO(X_test)
        clases_generadas = []

        for i in range(X_test.shape[0]):
            salida_oculta = activacionesO[i]
            salida = self.funcion_salida(np.dot(self.pesos_salida, salida_oculta))
            clases_generadas.append(np.argmax(salida))

        precision = np.mean(np.argmax(y_test, axis=1) == clases_generadas)
        mse = np.mean((y_test - np.eye(y_test.shape[1])[clases_generadas])**2)
        return precision, mse, clases_generadas

    @staticmethod
    def cargar_datos(ruta, test_size=0.2):
        """
        Función para cargar los datos de un archivo CSV
        
        Args:
            ruta (str): Ruta del archivo CSV
            test_size (float64): Tamaño del conjunto de prueba
        
        Returns:
            ndarray: Retorna los datos de entrenamiento y prueba
        """
        df = pd.read_csv(ruta)
        X = np.array(df.iloc[:, :-1])
        y = pd.factorize(df.iloc[:, -1])[0]
        y = OneHotEncoder().fit_transform(y.reshape(-1, 1)).toarray()
        scaler = StandardScaler()
        X = scaler.fit_transform(X)  # Normalización
        return train_test_split(X, y, test_size=test_size)

    @staticmethod
    def graficar_error_epocas(errores):
        plt.figure(figsize=(8, 6))
        plt.plot(range(1, len(errores) + 1), errores, linestyle=':')
        plt.title('Error por Época')
        plt.xlabel('Épocas')
        plt.ylabel('Error Cuadrático Medio (MSE)')
        plt.grid(True)
        plt.show()

    #* Graficar centroides
    def graficar_centroides(self, X_train):
        plt.figure(figsize=(8, 6))
        plt.scatter(X_train[:, 0], X_train[:, 1], c='blue', label='Datos')
        plt.scatter(self.centroides[:, 0], self.centroides[:, 1], c='red', label='Centroides')
        plt.title('Centroides')
        plt.xlabel('X1')
        plt.ylabel('X2')
        plt.legend()
        plt.grid(True)
        plt.show()

def main():
    #^ Parámetros de la red ingresados por el usuario
    opc = str(input("¿Cuál dataset desea utilizar?\n1. Iris.csv\n2. Bill_authentication.csv\nIngrese la opción: "))
    if opc == "1":
        df = "C:/Users/Cristian/Programacion/Python/Redes_Neuronales/Iris.csv"
        X_train, X_test, y_train, y_test = RedRBF.cargar_datos(df, test_size=0.3)
    elif opc == "2":
        df = "C:/Users/Cristian/Programacion/Python/Redes_Neuronales/Bill_authentication.csv"
        X_train, X_test, y_train, y_test = RedRBF.cargar_datos(df)
    else:
        print("Opción inválida.")
        return 0

    neuronasO = int(input("Ingrese el número de neuronas de la capa oculta: "))
    tasa_aprendizaje = float(input("Ingrese la tasa de Aprendizaje: "))
    epocas = int(input("Ingrese el número de épocas: "))
    f_capa_oculta = str(input("\n¿Qué función de activación desea utilizar en la capa oculta? (Default = Gaussiana)\n1. Gausiana\n2. Multicuadrática\n3. Multicuadrática inversa\nIngrese la opción: "))
    f_capa_salida = str(input("\n¿Qué función de activación desea utilizar en la capa de salida? (Default = Softmax)\n1. Sigmoide\n2. Softmax\n3. Tangente hiperbólica\nIngrese la opción: "))
    
    #^ Crear el modelo RBF
    rbf = RedRBF(
        neuronasO=neuronasO,
        tasa_aprendizaje=tasa_aprendizaje,
        epocas=epocas,
        funcion_oculta=RedRBF.gaussiana,
        funcion_salida=RedRBF.softmax
    )
    # Si se especifican otras funciones de activación se toman las que estan por default
    rbf.asignar_funciones(f_capa_oculta, f_capa_salida)

    #^ Entrenar el modelo
    rbf.entrenar(X_train, y_train)
    rbf.graficar_centroides(X_train)
    rbf.graficar_error_epocas(rbf.errores_totales)

    
    #^ Resultados
    precision, mse, predicciones = rbf.evaluar(X_test, y_test)
    print(f"\nPrecisión: {precision * 100:.2f}%")
    print(f"Error Cuadrático Medio (MSE): {mse:.4f}")
    print(f"\nPesos finales:\n{rbf.pesos_salida}\n")

    print("y_Real vs y_Obtenida")
    for i in range(y_test.shape[0]):
        print(f"Real: {np.argmax(y_test[i])} | Predicción: {predicciones[i]}")

if __name__ == "__main__":
    main()