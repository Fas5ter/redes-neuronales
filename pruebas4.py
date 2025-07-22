import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

class RedRBF:
    def __init__(self, neuronas_ocultas, tasa_aprendizaje, epocas, funcion_oculta, funcion_salida):
        self.neuronas_ocultas = neuronas_ocultas
        self.tasa_aprendizaje = tasa_aprendizaje
        self.epocas = epocas
        self.funcion_oculta = funcion_oculta
        self.funcion_salida = funcion_salida
        self.centroides = None
        self.varianzas = None
        self.pesos_salida = None
    
    #! Funciones de activación capa oculta
    @staticmethod
    def gaussiana(r, varianza):
        return np.exp(-r**2 / (2 * varianza**2))

    @staticmethod
    def multicuadratica(r):
        return np.sqrt(1 + r**2)

    @staticmethod
    def multicuadratica_inversa(r):
        return 1 / np.sqrt(1 + r**2)

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
    
    def calcular_varianzas(self, centroides):
        """
        Función para calcular la varianza de los clusters
        
        Args:
            centroides (_type_): Centros de los clusters
        
        Returns:
            _type_: Retorna la varianza de los clusters
        """
        
        varianzas = []
        for i, centroide in enumerate(centroides):
            distancias = [np.linalg.norm(centroide - otros) for j, otros in enumerate(centroides) if i != j]
            distancias.sort()
            varianza = np.sqrt(distancias[0] * distancias[1])
            varianzas.append(varianza)
        return varianzas
    
    def capa_oculta_rbf(self, X):
        """
        Función para aplicar la capa oculta RBF
        
        Args:
            X (_type_): Datos de entrada
            centroides (_type_): Centros de los clusters
            funcion_activacion (_type_): Función de activación
            varianzas (_type_): Varianza de los clusters
        
        Returns:
            _type_: Retorna las activaciones ocultas
        """
        activaciones_ocultas = np.zeros((X.shape[0], len(self.centroides)))
        for i, centroide in enumerate(self.centroides):
            distancias = np.linalg.norm(X - centroide, axis=1)
            if self.varianzas is not None and self.funcion_oculta == self.gaussiana:
                activaciones_ocultas[:, i] = self.funcion_oculta(distancias, self.varianzas[i])
            else:
                activaciones_ocultas[:, i] = self.funcion_oculta(distancias)
        return activaciones_ocultas

    def entrenar(self, X_train, y_train):
        """
        Función para entrenar el modelo RBF
        
        Args:
            X_train (_type_): Datos de entrenamiento
            y_train (_type_): Datos de salida
        
        Returns:
            _type_: Retorna los pesos de salida
        """
        self.centroides = KMeans(n_clusters=self.neuronas_ocultas).fit(X_train).cluster_centers_
        
        # Gráfica de los centroides
        plt.figure(figsize=(8, 6))
        plt.scatter(X_train[:, 0], X_train[:, 1], c='blue', label='Datos')
        plt.scatter(self.centroides[:, 0], self.centroides[:, 1], c='red', label='Centroides')
        plt.title('Centroides')
        plt.xlabel('X1')
        plt.ylabel('X2')
        plt.legend()
        plt.grid(True)
        plt.show()
        
        self.varianzas = self.calcular_varianzas(self.centroides)
        
        activaciones_ocultas = self.capa_oculta_rbf(X_train)
        self.pesos_salida = np.random.uniform(-1, 1, (y_train.shape[1], activaciones_ocultas.shape[1]))
        errores_totales = []

        # Entrenamiento
        for epoca in range(self.epocas):
            salidas = self.funcion_salida(np.dot(self.pesos_salida, activaciones_ocultas.T)).T
            errores = salidas - y_train 
            self.pesos_salida -= self.tasa_aprendizaje * np.dot(errores.T, activaciones_ocultas)
            error_total = np.mean(np.square(errores))
            errores_totales.append(error_total)

        # plt.figure(figsize=(8, 6))
        # plt.plot(range(1, self.epocas + 1), errores_totales, linestyle='-')
        # plt.title('Error por Época')
        # plt.xlabel('Épocas')
        # plt.ylabel('Error Cuadrático Medio (MSE)')
        # plt.grid(True)
        # plt.show()
        return 
        
    def evaluar(self, X_test, y_test):
        """
        Función para evaluar el modelo RBF
        
        Args:
            X_test (_type_): Datos de prueba
            y_test (_type_): Datos de salida
            
        Returns:
            _type_: Retorna la precisión y el error cuadrático medio
        """
        activaciones_ocultas = self.capa_oculta_rbf(X_test)
        clases_generadas = []

        for i in range(X_test.shape[0]):
            salida_oculta = activaciones_ocultas[i]
            salida = self.funcion_salida(np.dot(self.pesos_salida, salida_oculta))
            clases_generadas.append(np.argmax(salida))

        precision = np.mean(np.argmax(y_test, axis=1) == clases_generadas)
        mse = np.mean((y_test - np.eye(y_test.shape[1])[clases_generadas])**2)
        return precision, mse, clases_generadas

    @staticmethod
    def cargar_datos(ruta):
        df = pd.read_csv(ruta)
        X = np.array(df.iloc[:, :-1]) # Agarrar todas las columnas menos la última
        y = pd.factorize(df.iloc[:, -1])[0] # Agarrar la última columna
        y = OneHotEncoder().fit_transform(y.reshape(-1, 1)).toarray() # Codificación OneHot
        return train_test_split(X, y, test_size=0.2)

def main():
    # Elección del dataset
    opc = str(input("¿Cuál dataset desea utilizar?\n1. Iris.csv\n2. Bill_authentication.csv\nIngrese la opción: "))
    if opc == "1":
        df = "C:/Users/Cristian/Programacion/Python/Redes_Neuronales/Iris.csv"
        # df = "C:/Programacion/Python/Redes_Neuronales/Iris.csv"
    if opc == "2":
        df = "C:/Users/Cristian/Programacion/Python/Redes_Neuronales/Bill_authentication.csv"
        # df = "C:/Programacion/Python/Redes_Neuronales/Bill_authentication.csv"
    
    # Cargar datos
    X_train, X_test, y_train, y_test = RedRBF.cargar_datos(df)

    # Parámetros de la red
    neuronas = int(input("Ingresa el número de neuronas en la capa oculta: "))
    tasa_aprendizaje = float(input("Ingresa la tasa de Aprendizaje: "))
    epocas = int(input("Ingresa el número de épocas: "))
    f_capa_oculta = str(input("¿Qué función de activación desea utilizar en la capa oculta? (Default = Gaussiana)\n1. Gausiana\n2. Multicuadrática\n3. Multicuadrática inversa\nIngrese la opción: "))
    f_capa_salida = str(input("¿Qué función de activación desea utilizar en la capa de salida? (Default = Softmax)\n1. Sigmoide\n2. Softmax\n3. Tangente hiperbólica\nIngrese la opción: "))
    
    # Crear y entrenar la red
    rbf = RedRBF(
        neuronas_ocultas=neuronas,
        tasa_aprendizaje=tasa_aprendizaje,
        epocas=epocas,
        funcion_oculta=RedRBF.gaussiana,
        funcion_salida=RedRBF.softmax
    )
    
    # Asignar funciones de activación
    match f_capa_oculta:
        case "1":
            rbf.funcion_oculta = RedRBF.gaussiana
        case "2":
            rbf.funcion_oculta = RedRBF.multicuadratica
        case "3":
            rbf.funcion_oculta = RedRBF.multicuadratica_inversa
    
    match f_capa_salida:
        case "1":
            rbf.funcion_salida = RedRBF.sigmoide
        case "2":
            rbf.funcion_salida = RedRBF.softmax
        case "3":
            rbf.funcion_salida = RedRBF.tanh
    
    rbf.entrenar(X_train, y_train)
    
    # Evaluar la red
    precision, mse, predicciones = rbf.evaluar(X_test, y_test)
    print(f"\nPrecisión: {precision * 100:.2f}%")
    print(f"Error Cuadrático Medio: {mse:.4f}")
    print(f"\nPesos finales:\n{rbf.pesos_salida}\n")
    
    # plt.plot(range(y_test.shape[0]), rbf.funcion_salida(np.dot(rbf.pesos_salida, rbf.capa_oculta_rbf(X_test).T).T), label='Valores de Salida')
    print("y_Real vs Y_Obtenida")
    for i in range(y_test.shape[0]):
        print(f"Real: {np.argmax(y_test[i])}, Predicción: {predicciones[i]}")

if __name__ == "__main__":
    main()
