
import pandas as pd
from sklearn.model_selection import train_test_split

def load_and_process_data():
    # Leer el archivo CSV (sin encabezado)
    df = pd.read_csv('data/dataset.csv', header=None)

    # Asignar la primera fila como los nombres de las columnas
    df.columns = df.iloc[0]
    df = df.drop(0).reset_index(drop=True)  # Eliminar la primera fila que se usó como encabezado

    # Verificar que la columna 'MEDV' (precio de la casa) está presente
    print(df.columns)

    # Seleccionar las características (X) y la etiqueta (y)
    X = df.drop('MEDV', axis=1)  # Eliminar la columna 'MEDV' para las características
    y = df['MEDV']  # La columna 'MEDV' es la etiqueta (precio)

    # Dividir el dataset en conjunto de entrenamiento y conjunto de prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test
