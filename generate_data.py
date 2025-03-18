import pandas as pd

# Leer el archivo CSV
df = pd.read_csv('data/dataset.csv', header=None)

# Mostrar las primeras filas del archivo para verificar la estructura
print(df.head())

# Suponiendo que la primera fila contiene los nombres de las columnas, asignamos la primera fila como encabezado
# y eliminamos esa fila del DataFrame
df.columns = df.iloc[0]  # Asignar la primera fila como nombres de las columnas
df = df.drop(0).reset_index(drop=True)  # Eliminar la primera fila (ahora que ya se usa como encabezado)

# Mostrar las primeras filas del dataframe para verificar que los encabezados se asignaron correctamente
print(df.head())

# Verificar las columnas
print(df.columns)
