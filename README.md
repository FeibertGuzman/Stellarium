# Proyecto Stellarium: IA y ML para la Observación Astronómica 🌌

[![Python Version](https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11-blue.svg)](https://www.python.org/)
[![Streamlit App](https://static.streamlit.io/badges/streamlit-badge-primary.svg)](https://share.streamlit.io/FeibertGuzman/Proyecto-Stellarium/main/src/app.py)
[![HTML](https://img.shields.io/badge/HTML-View-orange.svg)](https://github.com/FeibertGuzman/Proyecto-Stellarium/blob/main/index.html)
[![Notebook](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://github.com/FeibertGuzman/Proyecto-Stellarium/blob/main/notebook.ipynb)
[![Metodología](https://img.shields.io/badge/Metodolog%C3%ADa-CRISP--ML-purple.svg)](https://arxiv.org/abs/2103.04571)
[![Licencia](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

Este proyecto desarrolla una solución integral basada en **Inteligencia Artificial y Machine Learning** orientada a la observación del firmamento nocturno para aficionados y principiantes con telescopios básicos de montura manual.

El sistema recopila datos meteorológicos e históricos y los correlaciona con la posición de los cuerpos celestes mediante efemérides tridimensionales de la NASA para identificar y prescribir los momentos y locaciones idóneos (como el **Desierto de la Tatacoa**) donde coinciden **múltiples planetas visibles en el firmamento**.

---

## 📂 Estructura del Repositorio

```
Proyecto Stellarium/
├── data/
│   └── dataset_clean.csv                # Dataset generado por el proceso ETL/Simulación
├── src/
│   ├── app.py                           # Aplicación interactiva de Streamlit (9 módulos)
│   ├── train_simulation.py              # Script de generación de datos y entrenamiento de ML
│   └── stellarium_model.joblib          # Modelo Random Forest serializado y entrenado
├── index.html                           # Landing Page premium de presentación del proyecto
├── Proyecto_Stellarium_Talento_Tech.md  # Documento Maestro Académico (18 Pasos desarrollados)
├── requirements.txt                     # Archivo de dependencias del entorno de Python
└── README.md                            # Guía rápida del repositorio
```

---

## ⚡ Instalación y Ejecución Local

Siga los siguientes pasos para configurar y ejecutar la aplicación en su máquina local:

### 1. Clonar el repositorio y acceder a la carpeta
```bash
git clone https://github.com/FeibertGuzman/Stellarium.git
cd Stellarium
```

### 2. Crear y activar un entorno virtual (Recomendado)
En PowerShell (Windows):
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 3. Instalar las dependencias requeridas
```bash
pip install -r requirements.txt
```

### 4. Ejecutar el pipeline de datos y entrenamiento del modelo
Este paso generará el dataset sintético limpio y entrenará el clasificador **Random Forest**:
```bash
python src/train_simulation.py
```

### 5. Lanzar la aplicación interactiva de Streamlit
```bash
streamlit run src/app.py
```

---

## 🛠️ Tecnologías y Librerías Utilizadas

* **Procesamiento de Datos:** `pandas`, `numpy`
* **Machine Learning:** `scikit-learn` (Random Forest, GridSearchCV)
* **Cálculo Astronómico:** `skyfield`, `astropy` (Efemérides JPL de la NASA)
* **Frontend y Visualizaciones:** `streamlit`, `plotly`, `folium` (mapas interactivos), `streamlit-folium`
* **Estilos y Landing:** CSS Puro (Vanilla CSS), Lucide Icons, HTML5 Semántico

---

## 🤝 Contribuciones

- **Autor:** Feibert Alirio Guzmán Pérez
- ¡Se aceptan Pull Requests! Por favor abra un Pull Request con una descripción clara de los cambios, referencias a issues cuando aplique y, si es posible, pruebas o instrucciones para reproducir.
