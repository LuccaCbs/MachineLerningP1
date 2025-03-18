from sklearn.ensemble import RandomForestRegressor

def entrenar_modelo(X_train, y_train):
    # Crear el modelo Random Forest
    modelo = RandomForestRegressor(n_estimators=250, random_state=42)

    # Entrenar el modelo
    modelo.fit(X_train, y_train)

    return modelo
