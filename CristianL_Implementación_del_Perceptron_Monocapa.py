# # Implementación del Perceptron Monocapa
# # Autor: Cristian Armando Larios Bravo
# # Asignatura: Redes Neuronales





# Importación de librerías.
import numpy as np
import pandas as pd

# ### Iris.csv

# Cargar el dataset.
print("Dataset Iris.csv")
df = pd.read_csv("C:/Users/Cristian/Programacion/Python/Redes_Neuronales/Iris.csv")
# df = pd.read_csv("C:/Programacion/Python/Redes_Neuronales/Iris.csv")


df = df.drop(['Id'], axis=1)
# #### Discretización

# Discretización de los datos[Clases]. (0, 1)
# Cambiar las etiquetas de las especies a números.
df["Species"].unique()
df["Species"].value_counts()
df["Species"] = df["Species"].replace({"Iris-setosa": 0, "Iris-versicolor": 1, "Iris-virginica": 2})
df["Species"].unique()
df["Species"].value_counts()

# Eliminar la especie Iris-virginica
df = df.drop(df[df["Species"] == 2].index)

df["Species"].value_counts()

# #### Entrenamiento

# 4 pesos aleatorios para 4 características.
np.random.seed(0)
weights = np.random.rand(4, 1)
# weights = [1,2,3,4]
# weights = [0.1, 0.15, 0.2, 0.25]
print(weights)

# Función de activación.
def activacion(suma):
    return 1 if suma >= 0 else 0


# Funcion de Error.
def error(y, y_pred):
    return y - y_pred


def sumaMuchos (*args):
    suma = 0
    for arg in args:
        suma += arg
    return suma


# Función de Entrenamiento del perceptron.
def entrenamiento(entradas, pesos, taza_aprendizaje, epocas):
    aprox = 0
    epoca = 0
    for epoch in range(epocas):
        print(f"Epoca: {epoch+1}")
        # print(f"Pesos actuales: {pesos}")
        ErrorAcum = 0
        if aprox == len(entradas):
                break
        aprox = 0
        # epoca +=1
        for i in range(len(entradas)):
            # Sumatoria
            Sumatoria = sumaMuchos((entradas.values[i][0]*pesos[0]),
                                   (entradas.values[i][1]*pesos[1]),
                                   (entradas.values[i][2]*pesos[2]),
                                   (entradas.values[i][3]*pesos[3]))
            
            # Función de activación
            y_pred = activacion(Sumatoria)
            
            # Función de error
            err = error(entradas.values[i][4], y_pred)
            auxError = taza_aprendizaje * err
            
            # w(k+1) = wk + (TasaAprendizaje * Error * Caracteristica)
            
            # Comparar si los pesos son iguales, para detener el entrenamiento
            if (pesos[0] + (taza_aprendizaje * err * entradas.values[i][0])) - pesos[0] <= 0.5\
                and (pesos[1] + (taza_aprendizaje * err * entradas.values[i][1])) - pesos[1] <= 0.5\
                and (pesos[2] + (taza_aprendizaje * err * entradas.values[i][2])) - pesos[2] <= 0.5\
                and (pesos[3] + (taza_aprendizaje * err * entradas.values[i][3])) - pesos[3] <= 0.5:
                aprox += 1
            
            # Error absoluto es 0

            pesos[0] = pesos[0] + (taza_aprendizaje * err * entradas.values[i][0])
            pesos[1] = pesos[1] + (taza_aprendizaje * err * entradas.values[i][1])
            pesos[2] = pesos[2] + (taza_aprendizaje * err * entradas.values[i][2])
            pesos[3] = pesos[3] + (taza_aprendizaje * err * entradas.values[i][3])
            
            # print(f"Pesos actuales: {pesos}")
            # ErrorAcum += int(auxError !=0.0)**2
            
            if aprox == len(entradas):
                break
            
        epoca += 1
    print(f"Epocas: {epoca+1}")
    print(f"Pesos finales: {pesos}")
    return pesos


# Datos de entrenamiento y prueba con libreria sklearn.
from sklearn.model_selection import train_test_split

X = df.copy()
# X = X.drop('Species', axis=1)
y = df.copy()
y = y['Species'] 

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
pesos = entrenamiento(x_train, weights, 0.3, 1000)


# !#### Prueba iris.csv


x = 0
for i in range(len(x_test)):
    Suma = sumaMuchos((x_test.values[i][0]*pesos[0]),(x_test.values[i][1]*pesos[1]),(x_test.values[i][2]*pesos[2]), (x_test.values[i][3]*pesos[3]))
    y_pred = activacion(Suma)
    err = error(x_test.values[i][4], y_pred)
    if err == 0:
        x += 1
    print(f"Sumatoria: {Suma} | Predicción: {y_pred} | Error: {err}")
    # print(f"Real: {x_test.values[i][4]} Predicción: {y_pred} Error: {err}")
print(f"Exactitud: {x/len(x_test)}")


str(input())
print("\n\n")
print("Dataset Bill_authentication.csv")

# ### Bill_authentication.csv

# Cargar el dataset.
# df2 = pd.read_csv("C:/Programacion/Python/Redes_Neuronales/bill_authentication.csv")
df2 = pd.read_csv("C:/Users/Cristian/Programacion/Python/Redes_Neuronales/bill_authentication.csv")

# #### Observaciones
# * No es necesario aplicar discretización, ya que los valores de las clases son 0 y 1


np.random.seed(5)
weights2 = np.random.rand(4, 1)


# #### Entrenamiento Bill Authentication
# Datos de entrenamiento y prueba con libreria sklearn.
from sklearn.model_selection import train_test_split

X2 = df2.copy()
y2 = df2.copy()
y2 = y2['Class'] 

x_train2, x_test2, y_train2, y_test2 = train_test_split(X2, y2 ,test_size=0.2, random_state=42)
pesos2 = entrenamiento(x_train2, weights2, 0.3, 1000)

# !#### Prueba Bill Authentication
x2 = 0
for i in range(len(x_test2)):
    Suma2 = sumaMuchos((x_test2.values[i][0]*pesos2[0]),(x_test2.values[i][1]*pesos2[1]),(x_test2.values[i][2]*pesos2[2]), (x_test2.values[i][3]*pesos2[3]))
    y_pred2 = activacion(Suma2)
    err2 = error(x_test2.values[i][4], y_pred2)
    if err2 == 0:
        x2 += 1
    print(f"Sumatoria: {Suma2} | Predicción: {y_pred2} | Error: {err2}")
print(f"Exactitud: {x2/len(x_test2)}")