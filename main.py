from src.modelo import entrenar_modelo
from src.preprocesamiento import load_and_process_data
from src.evaluacion import evaluar_modelo
import os

# Verificar si la carpeta 'output' existe, y si no, crearla
if not os.path.exists('output'):
    os.makedirs('output')

# Función principal que entrena y evalúa el modelo
def train_and_evaluate_model():
    # Cargar y procesar los datos
    X_train, X_test, y_train, y_test = load_and_process_data()

    # Entrenar el modelo Random Forest
    modelo = entrenar_modelo(X_train, y_train)

    # Hacer predicciones
    y_pred = modelo.predict(X_test)

    # Evaluar el modelo
    mse, r2 = evaluar_modelo(y_test, y_pred)

    # Mostrar los resultados
    print(f"Mean Squared Error: {mse}")
    print(f"R²: {r2}")

    # Guardar los resultados en el archivo "resultados.txt"
    with open("output/resultados.txt", "w") as file:
        file.write(f"Mean Squared Error: {mse}\n")
        file.write(f"R²: {r2}\n")

if __name__ == "__main__":
    train_and_evaluate_model()
