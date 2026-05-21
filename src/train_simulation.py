import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib

# Definir semilla para reproducibilidad
np.random.seed(42)

def generate_synthetic_data(num_samples=2000):
    """
    Genera un dataset sintético realista para el Proyecto Stellarium.
    Simula condiciones de observación astronómica en diferentes locaciones de Colombia.
    """
    print(f"Generando {num_samples} registros sintéticos...")
    
    # 1. Variables climáticas
    nubosidad = np.random.randint(0, 101, size=num_samples)  # Cobertura nubosa en %
    humedad = np.random.randint(20, 101, size=num_samples)    # Humedad relativa en %
    temperatura = np.random.uniform(8.0, 32.0, size=num_samples)  # °C
    velocidad_viento = np.random.uniform(0.0, 15.0, size=num_samples)  # m/s
    presion = np.random.uniform(980.0, 1025.0, size=num_samples)  # hPa
    
    # 2. Variables geográficas y de ubicación
    # Bortle Scale: 1 (prístino) a 9 (gran ciudad)
    # Ubicaciones predefinidas de simulación
    locations = ['Tatacoa', 'Villa de Leyva', 'Bogotá', 'Medellín', 'Cali', 'Guajira']
    loc_probs = [0.15, 0.15, 0.25, 0.20, 0.15, 0.10]
    selected_locs = np.random.choice(locations, size=num_samples, p=loc_probs)
    
    bortle_map = {'Tatacoa': 2, 'Villa de Leyva': 3, 'Bogotá': 8, 'Medellín': 7, 'Cali': 7, 'Guajira': 1}
    altitud_map = {'Tatacoa': 400, 'Villa de Leyva': 2149, 'Bogotá': 2640, 'Medellín': 1495, 'Cali': 1018, 'Guajira': 50}
    
    bortle = np.array([bortle_map[loc] for loc in selected_locs])
    altitud = np.array([altitud_map[loc] for loc in selected_locs])
    
    # Agregar un pequeño ruido a bortle y altitud para simular variabilidad de microzona
    bortle = np.clip(bortle + np.random.choice([-1, 0, 1], size=num_samples, p=[0.1, 0.8, 0.1]), 1, 9)
    altitud = altitud + np.random.normal(0, 30, size=num_samples)
    
    # 3. Variables astronómicas
    fase_lunar = np.random.uniform(0.0, 1.0, size=num_samples)  # 0: Nueva, 1: Llena
    # Cantidad de planetas visibles sobre el horizonte (0 a 5 de los brillantes principales)
    planetas_visibles = np.random.choice([0, 1, 2, 3, 4, 5], size=num_samples, p=[0.10, 0.20, 0.30, 0.25, 0.10, 0.05])
    
    # 4. Construcción de la Variable Objetivo: Calidad_Observacion (Binaria)
    # Lógica física de observabilidad:
    # Cielos despejados (< 30% nubosidad), humedad aceptable (< 80%), baja contaminación lumínica (Bortle < 5),
    # velocidad de viento estable (< 10 m/s) y fase lunar no muy brillante (< 0.5 o Luna Nueva)
    # Además, la presencia de "múltiples planetas visibles en el firmamento" (planetas_visibles >= 3)
    
    scores = (
        (100 - nubosidad) * 0.4 +
        (100 - humedad) * 0.15 +
        (10 - bortle) * 10 * 0.2 +
        (1.0 - fase_lunar) * 15 * 0.1 +
        (altitud / 3000) * 10 * 0.1 +
        (planetas_visibles * 2) * 0.05
    )
    
    # Agregar ruido aleatorio al score
    scores += np.random.normal(0, 5, size=num_samples)
    
    # Clasificación final: 1 (Excelente/Favorable) si score > 60 y nubosidad < 35% y Bortle < 6, de lo contrario 0 (Pobre/Inviable)
    calidad = np.where((scores > 58) & (nubosidad < 35) & (bortle < 6), 1, 0)
    
    # Crear DataFrame
    df = pd.DataFrame({
        'Ubicacion': selected_locs,
        'Altitud': altitud,
        'Bortle': bortle,
        'Nubosidad': nubosidad,
        'Humedad': humedad,
        'Temperatura': temperatura,
        'Velocidad_Viento': velocidad_viento,
        'Presion': presion,
        'Fase_Lunar': fase_lunar,
        'Planetas_Visibles': planetas_visibles,
        'Score_Visibilidad': scores,
        'Calidad_Observacion': calidad
    })
    
    return df

def train_model(df):
    """
    Entrena un clasificador Random Forest para predecir si una noche será óptima.
    """
    print("Entrenando el modelo de Machine Learning...")
    
    # Definir características (features) y variable objetivo (target)
    features = ['Altitud', 'Bortle', 'Nubosidad', 'Humedad', 'Temperatura', 'Velocidad_Viento', 'Presion', 'Fase_Lunar', 'Planetas_Visibles']
    X = df[features]
    y = df['Calidad_Observacion']
    
    # Dividir en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    
    # Inicializar y entrenar el modelo
    model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluar el modelo
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nExactitud del Modelo (Accuracy): {accuracy:.4f}")
    print("\nReporte de Clasificación:")
    print(classification_report(y_test, y_pred))
    
    # Importancia de las variables
    importances = model.feature_importances_
    for name, importance in zip(features, importances):
        print(f"Importancia de la variable '{name}': {importance:.4f}")
        
    return model

if __name__ == '__main__':
    # Asegurar existencia de carpetas
    os.makedirs('src', exist_ok=True)
    os.makedirs('data', exist_ok=True)
    
    # Generar y guardar el dataset limpio
    df = generate_synthetic_data()
    dataset_path = 'data/dataset_clean.csv'
    df.to_csv(dataset_path, index=False)
    print(f"Dataset guardado exitosamente en: {dataset_path}")
    
    # Entrenar modelo
    model = train_model(df)
    
    # Guardar el modelo entrenado
    model_path = 'src/stellarium_model.joblib'
    joblib.dump(model, model_path)
    print(f"Modelo serializado y guardado exitosamente en: {model_path}")
