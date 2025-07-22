import os
import time
import tabulate
import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import filedialog
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split

np.random.seed(24)

#? Limpieza de datos
def Limpieza(ruta):
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
    return train_test_split(X, y, test_size=0.2)

#? Ingreso de datos
def Data():
    while(True):
        try:
        #* Archivo
            root = tk.Tk()
            root.withdraw()
            print("Ingresa el data set:")
            df = filedialog.askopenfilename()
            data = pd.read_csv(df)
            print(f"\n El archivo seleccionado es: {df}\n")
            break
        except:
            print("Ingresa un valor válido.")
    return data

#? Ingreso de parámetros
def Parametros():
    while(True):
        try:
            print("\n\nFunciones de activación de la capa oculta:\n 1.- Gausiana\n 2.- Multicuadratica\n 3.- Multicuadratica inversa\n")
            func_oculta = input("Ingresa la función de activación: ")
            if int(func_oculta) > 0 and int(func_oculta) < 4:
                break
        except:
            print("Ingresa un valor válido.")
    while(True):
        try:
            print("\n\nFunciones de activación:\n 1.- Tangente hiperbólica\n 2.- Sigmoide\n 3.- Softmax\n")
            func_salida = input("Ingresa la función de activación: ")
            if int(func_salida) > 0 and int(func_salida) < 4:
                break
        except:
            print("Ingresa un valor válido.")
    while(True):
        try:
            epocas = int(input("Ingresa la cantidad de epocas máximas: "))
            if epocas > 0:
                break
        except:
            print("Ingresa un valor válido.")
    while(True):
        try:
            alpha = float(input("Ingresa la taza de aprendizaje: "))
            if alpha > 0 and alpha < 1:
                break
        except:
            print("Ingresa un valor válido.")
       
    return func_oculta, func_salida, epocas, alpha

#? Centros
def Centros(data: np.array, k:int):
    """Función que calcula los centros de los clusters y los clusters

    Args:
        data (np.array): Arreglo de datos
        k (int): Número de clusters

    Returns:
        np.array: Centros de los clusters
    """
    kmeans = KMeans(n_clusters=k, random_state=0).fit(data)
    return kmeans.cluster_centers_

#? Varianza por cluster
def Varianza(centros: np.ndarray) -> np.array:
    """Función que calcula la varianza de los clusters

    Args:
        centros (np.ndarray): Arreglo con los centros de cada cluster

    Returns:
        np.ndarray: Varianza de los clusters
    """        
    #* Calculo de la varianza del centro con los dos centros más cercanos (distancia externa)
    #? En caso de que solo hayan dos centros
    dists = np.linalg.norm(centros[:, np.newaxis] - centros, axis=2)
    np.fill_diagonal(dists, np.inf)  # Evita la distancia cero consigo mismo
    var_externa = np.min(dists, axis=1)
    return np.sqrt(var_externa / 2)

#? Funciones de activación capa oculta
def Func_oculta(funcion: str, data: np.array, centros: np.ndarray, varianza: np.array) -> np.array:
    """Función que calcula las seudosalidas por medio
    de la función de activación seleccionada.

    Args:
        funcion (str): Función de activación.
        data (np.array): Arreglo de datos.
        centros (np.array): Arreglo con los centros.
        varianza (np.array): Arreglo con las varianzas.

    Returns:
        np.array: Arreglo con las seudosalidas.
    """
    r = np.linalg.norm(data[:, np.newaxis] - centros, axis=2)
    
    # Inicializar matriz de activaciones
    # activacionesO = np.zeros((data.shape[0], len(centros)))

    # for i, centroide in enumerate(centros):
    #     # Calcular la distancia de cada muestra al centroide i
    #     r = np.linalg.norm(data - centroide, axis=1)
        
    #     # Aplicar function de activacion
    #     if funcion == "1": # 1.-Gausiana
    #         activacionesO[:, i] = np.exp(-r**2/(2*varianza[i]**2))
    #     elif funcion == "2": # 2.-Multicuadrática
    #         activacionesO[:, i] = np.sqrt(1 + r**2)
    #     elif funcion == "3": # 3.-Multicuadrática inversa
    #         activacionesO[:, i] = 1 / np.sqrt(1 + r**2)
        
    # return activacionesO

    match funcion:
        case "1": # 1.-Gausiana
            return np.exp(-r**2/(2*varianza**2))
        case "2": # 2.-Multicuadrática
            return np.sqrt(1 + r**2)
        case "3": # 3.-Multicuadrática inversa
            return 1 / np.sqrt(1 + r**2)
        case _:
            return np.array([0])

#? Funciones de activación capa salida
def Funcion_salida(num : np.array, funcion : str) -> np.array:
    """Función que evalúa la función de activación.

    Args:
        num (np.array): Matriz a evaluar.
        funcion (str): Función a evaluar.
        
    Returns:
        np.array: Valores de salida de la función de activación.
    """
    match funcion:
        case "1":
            return np.tanh(num) # 3.-Tangente hiperbólica
        case "2":
            return 1/(1+np.exp(-num)) # 4.-Sigmoide
        case "3":
            exp_values = np.exp(num - np.max(num, axis=1, keepdims=True))
            return exp_values / np.sum(exp_values, axis=1, keepdims=True) # 5.- Softmax
        case _:
            return np.array([0])

#? Derivadas de funciones de activación
def derivada_funcion_activacion(num: np.array, funcion : str) -> np.array:
    """Función que evalúa la derivada de la función de activación.

    Args:
        num (np.array): Matriz a evaluar.
        funcion (str): Función a evaluar.
        
    Returns:
        np.array: Valores de la derivada de la función de activación.
    """
    match funcion:
        case "1":
            return 1 - np.tanh(num)**2
        case "2":
            return num * (1 - num)
        case _:
            
            return np.array([0])  

def División_datos(data: np.array, clase: pd.Series):
    """Función que divide los datos en entrenamiento y prueba

    Args:
        data (np.array): Arreglo de datos.
        clase (pd.Series): Clases.

    Returns:
        np.array: Datos de entrenamiento.
        np.array: Datos de prueba.
        pd.Series: Clases de entrenamiento.
        pd.Series: Clases de prueba.
    """
    X_train, X_test, y_train, y_test = train_test_split(data, clase, test_size=0.2)
    X_train = np.array(X_train)
    y_train = np.eye(total_clases)[pd.factorize(y_train)[0]]
    return X_train, X_test, y_train, y_test

def Entrenamiento(seudo: np.array, y_train: np.array, epocas: int, alpha: float, func_salida: str):
    """Función que entrena la red neuronal

    Args:
        seudo (np.array): Seudo salidas.
        y_train (np.array): Clases reales de entrenamiento.
        epocas (int): Número máximo de épocas.
        alpha (float): Tas de aprendizaje.
        func_salida (str): Función de activación de la capa de salida.

    Returns:
        _type_: Sesgos y pesos finales.
    """
    
#! Inicialización de pesos y bias
    #? Inicialización de pesos iniciales
    w = np.random.uniform(-1, 1, (3,len(seudo[1])))
    #? Inicialización de bias
    bias = np.zeros((1, 3))

    print("\nPesos iniciales: \n", w)        
    print("Bias: \n", bias)
    
#! Entrenamiento
    for epoca in range(epocas):
        #* Calculo de la salida de la función de activación
        salida = np.dot(seudo, w.T) + bias
        salida_final = Funcion_salida(salida, func_salida)
        
        #* Calculo del error
        error = y_train - salida_final
        #* Calculo de la derivada de la función de activación
        if func_salida != "3":
            delta = error * derivada_funcion_activacion(salida_final, func_salida)
        else:
            delta = error
        
        #* Actualización de pesos y bias
        w += alpha * np.dot(delta.T, seudo)
        bias += alpha * np.sum(delta, axis=0)
        
        #* Calculo del error cuadrático medio
        error_epoca = np.mean(error**2)
        
        if (epoca+1) % 100 == 0:
            print(f"Época {epoca+1}, Pérdida: {error_epoca}")
            
        if error_epoca < 0.01:
            break
        
#! Impresión de resultados
    print("\n\nPesos finales: \n", w)
    print("Bias final: \n", bias)
    return w, bias

def Prueba(w: np.array, bias: np.array, X_test: np.array, y_test: pd.Series, func_oculta: str, func_salida: str, centros: np.array, var: np.array):
    """Función que prueba la red neuronal

    Args:
        w (np.array): Pesos finales.
        bias (np.array): Sesgos finales.
        X_test (np.array): Datos de prueba.
        y_test (pd.Series): Clases reales de prueba.
        func_oculta (str): Función de activación de la capa oculta.
        func_salida (str): Función de activación de la capa de salida.
        centros (np.array): Centros de los clusters.
        var (np.array): Varianza de los clusters.

    Returns:
        _type_: DataFrame con las clases reales y predichas, número de aciertos y predicciones.
    """
    
    #! Prueba
    resultado = pd.DataFrame()
    resultado["Real"] = y_test.values
    
    #* Calculo de salidas
    seudo_test = Func_oculta(func_oculta, X_test, centros, var)
    salida_test = np.dot(seudo_test, w.T) + bias
    salida_final_test = Funcion_salida(salida_test, func_salida)
    
    #* Predicciones
    pred_prueba = np.argmax(salida_final_test, axis=1)
    resultado["Predicción"] = pred_prueba
    
    #* Porcentaje de aciertos
    aciertos = (pred_prueba == y_test).sum()

    return resultado, aciertos, pred_prueba

if __name__ == "__main__":
    os.system('cls')
#! Datos default
    # df = 'C:/Users/nydia/Documents/7D/Redes neuronales/bill_authentication.csv'
    df = "C:/Users/Cristian/Programacion/Python/Redes_Neuronales/Iris.csv"
    data = pd.read_csv(df)
    func_salida = "2"
    func_oculta = "1"
    epocas = 1000
    alpha = 0.01

#! Ingreso y limpieza de datos   
    # data = Data()
    # data, total_clases, clase = Limpieza(data)

#! Ingreso de parámetros
    # func_oculta, func_salida, epocas, alpha = Parametros()

#! División de datos
    X_train, X_test, y_train, y_test = Limpieza(df)
    
#! Calculo de centros, varianza y seudo salidas
    centros = Centros(X_train, 3)
    print("Centros: \n", centros)
    var = Varianza(centros)
    print("\n\nVarianza: \n", var)
    seudo = Func_oculta(func_oculta, X_train, centros, var)
    # print("\n\nSeudo salidas: \n", seudo)

#! Entrenamiento
    w, bias = Entrenamiento(seudo, y_train, epocas, alpha, func_salida)
    
#! Prueba
    # resultado, aciertos, pred_prueba = Prueba(w, bias, X_test, y_test, func_oculta, func_salida, centros, var)

#! Impresión de resultados
    # print("\nPrueba: \n")
    # print(tabulate.tabulate(resultado, headers='keys', tablefmt='pretty'))
    
    # print(f"\nAciertos: {aciertos}/{len(pred_prueba)}")
    # print(f"\nPorcentaje de aciertos: {aciertos/len(pred_prueba)*100}%")
    
        