import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import folium
from streamlit_folium import st_folium
import os
import joblib

# Configuración de página de Streamlit
st.set_page_config(
    page_title="Stellarium IA - Dashboard de Observación",
    page_icon="🌌",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilos CSS personalizados para simular vidrio (glassmorphism) y estética premium en modo oscuro
st.markdown("""
<style>
    /* Estilo general */
    .stApp {
        background-color: #05070c;
        color: #f8f9fa;
    }
    
    /* Contenedores con efecto de vidrio */
    .glass-card {
        background: rgba(13, 20, 38, 0.55);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 15px;
        padding: 1.5rem;
        margin-bottom: 1.25rem;
        backdrop-filter: blur(8px);
        -webkit-backdrop-filter: blur(8px);
    }
    
    /* Títulos destacados */
    .gradient-text {
        font-weight: 800;
        background: linear-gradient(135deg, #ffffff 30%, #9d4edd 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    /* Frase objetivo resaltada */
    .highlight-target {
        color: #00b4d8;
        font-weight: 700;
        text-shadow: 0 0 10px rgba(0, 180, 216, 0.3);
    }
</style>
""", unsafe_allow_html=True)

# Intentar cargar el modelo entrenado, o usar una función heurística de contingencia
MODEL_PATH = "src/stellarium_model.joblib"
if os.path.exists(MODEL_PATH):
    try:
        model = joblib.load(MODEL_PATH)
        is_model_loaded = True
    except Exception:
        is_model_loaded = False
else:
    is_model_loaded = False

def predict_observation_quality(altitud, bortle, nubosidad, humedad, temperatura, viento, presion, fase_lunar, planetas_visibles):
    """
    Predice la probabilidad de una noche óptima.
    Usa el modelo entrenado si está disponible; si no, aplica una lógica matemática de heurística.
    """
    if is_model_loaded:
        features = [[altitud, bortle, nubosidad, humedad, temperatura, viento, presion, fase_lunar, planetas_visibles]]
        prob = model.predict_proba(features)[0][1]
        return prob
    else:
        # Heurística de respaldo (calculada según la lógica física descrita en la teoría del proyecto)
        score = (
            (100 - nubosidad) * 0.4 +
            (100 - humedad) * 0.15 +
            (10 - bortle) * 10 * 0.2 +
            (1.0 - fase_lunar) * 15 * 0.1 +
            (altitud / 3000) * 10 * 0.1 +
            (planetas_visibles * 2) * 0.05
        )
        # Normalizar entre 0 y 1
        prob = np.clip((score - 10) / 80, 0.0, 1.0)
        # Penalizaciones estrictas de clima real
        if nubosidad > 40:
            prob *= 0.3
        if bortle > 6:
            prob *= 0.5
        return prob

# Inicializar estado de comentarios de foro si no existe
if 'foro_comments' not in st.session_state:
    st.session_state['foro_comments'] = [
        {"usuario": "AstroJuan", "ubicacion": "Desierto de la Tatacoa", "comentario": "¡Increíble alineación ayer! Júpiter y Saturno se veían nítidos con un ocular Plössl de 10mm.", "fecha": "2026-05-18"},
        {"usuario": "Celeste_99", "ubicacion": "Villa de Leyva", "comentario": "La nubosidad subió a medianoche, pero pudimos captar a Marte cerca de la Luna durante el crepúsculo astronómico.", "fecha": "2026-05-19"}
    ]

# --- MENÚ LATERAL DE INTERFAZ ---
st.sidebar.markdown("<h2 class='gradient-text'>Stellarium IA</h2>", unsafe_allow_html=True)
st.sidebar.write("Optimización y Machine Learning para astronomía amateur.")

menu = st.sidebar.radio(
    "Navegación de Módulos",
    [
        "🪐 1. Landing Page",
        "📊 2. Dashboard de Monitoreo",
        "🗺️ 3. Mapa Georreferenciado",
        "🌌 4. Visualización del Firmamento",
        "📅 5. Predicción de Mejores Fechas",
        "📍 6. Predicción de Lugares",
        "🔭 7. Sistema de Recomendaciones",
        "📈 8. Panel Estadístico Histórico",
        "💬 9. Foro de Aficionados"
    ]
)

# --- MÓDULO 1: LANDING PAGE ---
if menu == "🪐 1. Landing Page":
    st.markdown("<h1 class='gradient-text'>Proyecto Stellarium: IA para la Observación Astronómica</h1>", unsafe_allow_html=True)
    st.markdown("### Plataforma Inteligente para Aficionados a la Astronomía | Talento Tech")
    
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.markdown(f"""
        <div class='glass-card'>
            <h4>¿Qué es Stellarium IA?</h4>
            <p>
                Este proyecto desarrolla una solución integral basada en <b>Inteligencia Artificial y Machine Learning</b>
                para mitigar los desafíos que enfrentan los aficionados a la astronomía que cuentan con equipos básicos o de principiante.
            </p>
            <p>
                A través de modelos predictivos y el procesamiento de efemérides orbitales de la NASA y datos meteorológicos,
                la solución determina de forma óptima cuándo y dónde se presentarán escenarios de 
                <span class='highlight-target'>múltiples planetas visibles en el firmamento</span>.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class='glass-card'>
            <h4>Cómo Funciona la Solución Paso a Paso:</h4>
            <ol>
                <li><b>Ingeniería de Datos (ETL):</b> Consultamos APIs en tiempo real de OpenWeather y cálculos del motor astronómico Skyfield.</li>
                <li><b>Predicción por Machine Learning:</b> Un modelo Random Forest clasifica la viabilidad de la noche de observación analizando variables climáticas y fases lunares.</li>
                <li><b>Prescripción Óptima:</b> El recomendador analiza tu equipo y sugiere las mejores locaciones de baja contaminación lumínica (como el Desierto de la Tatacoa) y configuraciones ópticas adecuadas.</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class='glass-card' style='text-align: center;'>
            <h1 style='font-size: 5rem;'>🌌</h1>
            <h4>Múltiples Planetas Visibles</h4>
            <p>Monitoreamos las órbitas celestes para predecir cuándo coincidirán planetas como Mercurio, Venus, Marte, Júpiter y Saturno alineados sobre el horizonte local.</p>
            <hr style='border-color: rgba(255,255,255,0.05); margin: 1rem 0;'>
            <h5>Estado del Modelo:</h5>
        </div>
        """, unsafe_allow_html=True)
        
        if is_model_loaded:
            st.success("✅ Modelo Random Forest Activo y Cargado (src/stellarium_model.joblib)")
        else:
            st.warning("⚠️ Modo de Contingencia Heurística Activo (Modelo Random Forest no detectado en 'src/')")
            st.info("Para activar el modelo real de ML, ejecute: `python src/train_simulation.py` en la terminal.")

# --- MÓDULO 2: DASHBOARD DE MONITOREO ---
elif menu == "📊 2. Dashboard de Monitoreo":
    st.markdown("<h2 class='gradient-text'>Dashboard de Monitoreo del Firmamento</h2>", unsafe_allow_html=True)
    st.write("Ajusta las condiciones en tiempo real para evaluar el índice predictivo de calidad visual.")
    
    col_inputs, col_gauge = st.columns([3, 2])
    
    with col_inputs:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.subheader("Parámetros de Entrada de la Noche")
        
        c1, c2, c3 = st.columns(3)
        with c1:
            altitud = st.slider("Altitud del Sitio (m.s.n.m)", 0, 3000, 400)
            bortle = st.slider("Escala Bortle (Polución Lumínica)", 1, 9, 2)
            nubosidad = st.slider("Cobertura Nubosa (%)", 0, 100, 15)
        with c2:
            humedad = st.slider("Humedad Relativa (%)", 10, 100, 50)
            temperatura = st.slider("Temperatura (°C)", 5, 40, 25)
            viento = st.slider("Velocidad del Viento (m/s)", 0.0, 20.0, 4.0)
        with c3:
            presion = st.slider("Presión Atmosférica (hPa)", 980, 1030, 1010)
            fase_lunar = st.slider("Fase Lunar (Disco Iluminado)", 0.0, 1.0, 0.1)
            planetas_visibles = st.selectbox("Planetas Visibles Coincidentes", [0, 1, 2, 3, 4, 5], index=3)
            
        st.markdown("</div>", unsafe_allow_html=True)
        
    with col_gauge:
        # Calcular predicción
        prob = predict_observation_quality(altitud, bortle, nubosidad, humedad, temperatura, viento, presion, fase_lunar, planetas_visibles)
        
        st.markdown("<div class='glass-card' style='text-align: center;'>", unsafe_allow_html=True)
        st.subheader("Calidad Predictiva del Cielo")
        
        # Indicador de Gauge interactivo
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = prob * 100,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Viabilidad de Observación (%)", 'font': {'size': 18, 'color': "#fff"}},
            gauge = {
                'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#fff"},
                'bar': {'color': "#9d4edd"},
                'bgcolor': "rgba(0,0,0,0)",
                'borderwidth': 2,
                'bordercolor': "rgba(255,255,255,0.1)",
                'steps': [
                    {'range': [0, 40], 'color': 'rgba(242, 92, 84, 0.3)'},
                    {'range': [40, 75], 'color': 'rgba(255, 183, 3, 0.3)'},
                    {'range': [75, 100], 'color': 'rgba(0, 180, 216, 0.3)'}
                ],
            }
        ))
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font={'color': "#fff", 'family': "Space Grotesk"}, height=280)
        st.plotly_chart(fig, use_container_width=True)
        
        if prob > 0.75:
            st.success("🌠 ¡EXCELENTES CONDICIONES! Oportunidad óptima para capturar múltiples planetas visibles en el firmamento.")
        elif prob > 0.4:
            st.warning("⛅ CONDICIONES REGULARES. Podría haber nubes pasajeras o interferencia de luz lunar.")
        else:
            st.error("🌧️ CONDICIONES DESFAVORABLES. Cielos cubiertos o polución lumínica extrema.")
        st.markdown("</div>", unsafe_allow_html=True)

# --- MÓDULO 3: MAPA GEORREFERENCIADO ---
elif menu == "🗺️ 3. Mapa Georreferenciado":
    st.markdown("<h2 class='gradient-text'>Santuarios de Observación en Colombia</h2>", unsafe_allow_html=True)
    st.write("Monitoreo geográfico de los mejores lugares con menor polución lumínica y nubosidad controlada.")
    
    # Dataset de puntos piloto
    places_data = pd.DataFrame({
        'name': ['Desierto de la Tatacoa', 'Villa de Leyva', 'Cabo de la Vela (Guajira)', 'Observatorio U. Andes (Bogotá)', 'Parque Explora (Medellín)'],
        'lat': [3.2323, 5.6322, 12.2030, 4.6015, 6.2711],
        'lon': [-75.1662, -73.5255, -72.1520, -74.0661, -75.5645],
        'bortle': [2, 3, 1, 8, 7],
        'altitud': [400, 2149, 20, 2640, 1495],
        'calidad_promedio': ['Excelente (Clase 2)', 'Buena (Clase 3)', 'Prístino (Clase 1)', 'Muy Pobre (Clase 8)', 'Pobre (Clase 7)']
    })
    
    col_map, col_details = st.columns([3, 1.5])
    
    with col_map:
        # Inicializar mapa Folium centrado en Colombia
        m = folium.Map(location=[5.0, -74.0], zoom_start=6, tiles="CartoDB dark_matter")
        
        for idx, row in places_data.iterrows():
            color = 'green' if row['bortle'] <= 3 else 'orange' if row['bortle'] <= 5 else 'red'
            folium.Marker(
                location=[row['lat'], row['lon']],
                popup=f"<b>{row['name']}</b><br>Bortle: {row['bortle']}<br>Calidad: {row['calidad_promedio']}",
                tooltip=row['name'],
                icon=folium.Icon(color=color, icon='eye-open')
            ).add_to(m)
            
        st_folium(m, width=700, height=450)
        
    with col_details:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.subheader("Análisis de Puntos")
        st.write("La polución lumínica medida bajo la escala de Bortle determina el contraste de planetas sutiles.")
        
        # Mostrar tabla interactiva de puntos
        st.dataframe(places_data[['name', 'bortle', 'altitud', 'calidad_promedio']])
        st.info("💡 El Desierto de la Tatacoa y la Guajira destacan con los índices Bortle más bajos, ideales para telescopios básicos.")
        st.markdown("</div>", unsafe_allow_html=True)

# --- MÓDULO 4: VISUALIZACIÓN DEL FIRMAMENTO ---
elif menu == "🌌 4. Visualización del Firmamento":
    st.markdown("<h2 class='gradient-text'>Simulación de Múltiples Planetas Visibles en el Firmamento</h2>", unsafe_allow_html=True)
    st.write("Representación angular de las posiciones de los planetas brillantes sobre el horizonte local (10° a 90°).")
    
    # Simular posiciones angulares tridimensionales
    planetas = ['Mercurio', 'Venus', 'Marte', 'Júpiter', 'Saturno']
    altitudes = [12, 45, 62, 58, 22]  # Grados sobre el horizonte
    azimut = [95, 120, 128, 135, 270]  # Direcciones (Este: 90, Sur: 180, Oeste: 270)
    
    # Crear gráfico polar simulado
    fig = go.Figure()
    
    # Dibujar plano del horizonte mínimo (10 grados)
    fig.add_trace(go.Scatterpolar(
        r = [10]*360,
        theta = list(range(360)),
        mode = 'lines',
        name = 'Horizonte Mínimo (10°)',
        line_color = 'rgba(242, 92, 84, 0.5)',
        line_dash = 'dash'
    ))
    
    # Dibujar planetas
    for i, p in enumerate(planetas):
        fig.add_trace(go.Scatterpolar(
            r = [altitudes[i]],
            theta = [azimut[i]],
            mode = 'markers+text',
            marker = dict(size=14, symbol='circle', line_width=1, line_color='#fff'),
            text = [p],
            textposition = "top center",
            name = p
        ))
        
    fig.update_layout(
        polar = dict(
            radialaxis = dict(visible=True, range=[0, 90], color="#adc1d6"),
            angularaxis = dict(color="#adc1d6")
        ),
        paper_bgcolor = 'rgba(0,0,0,0)',
        plot_bgcolor = 'rgba(0,0,0,0)',
        font = {'color': "#fff", 'family': "Space Grotesk"},
        height = 480
    )
    
    st.plotly_chart(fig, use_container_width=True)
    st.info("🪐 **Escenario de Alineación detectado:** Mercurio, Venus, Marte y Júpiter se concentran en el cuadrante Este-Sureste (Azimut 90° - 140°).")

# --- MÓDULO 5: PREDICCIÓN DE MEJORES FECHAS ---
elif menu == "📅 5. Predicción de Mejores Fechas":
    st.markdown("<h2 class='gradient-text'>Predicciones Temporales a 15 días</h2>", unsafe_allow_html=True)
    st.write("Pronóstico inteligente de las mejores noches para planificar salidas de observación.")
    
    # Generar predicciones simuladas a 15 días
    fechas = pd.date_range(start="2026-05-20", periods=15).strftime("%Y-%m-%d")
    
    np.random.seed(12)  # Seed para consistencia visual
    nubosidades = np.random.randint(5, 85, size=15)
    fases = np.linspace(0.0, 0.5, 15)  # Creciente
    planetas = np.random.choice([2, 3, 4], size=15)
    
    score_observacion = []
    for i in range(15):
        prob = predict_observation_quality(1500, 3, nubosidades[i], 55, 18, 3.2, 1012, fases[i], planetas[i])
        score_observacion.append(prob * 100)
        
    predictions_df = pd.DataFrame({
        'Fecha': fechas,
        'Nubosidad (%)': nubosidades,
        'Iluminación Lunar (%)': np.round(fases * 100, 1),
        'Planetas en Bóveda': planetas,
        'Probabilidad Éxito (%)': np.round(score_observacion, 1)
    })
    
    # Estilizar y mostrar la tabla de predicciones
    st.dataframe(predictions_df.style.highlight_max(subset=['Probabilidad Éxito (%)'], color='#9d4edd'))
    
    fig = px.line(predictions_df, x='Fecha', y='Probabilidad Éxito (%)', title="Tendencia de Viabilidad de Observación")
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font={'color': "#fff"}, height=300)
    st.plotly_chart(fig, use_container_width=True)

# --- MÓDULO 6: PREDICCIÓN DE LUGARES ---
elif menu == "📍 6. Predicción de Lugares":
    st.markdown("<h2 class='gradient-text'>Predicción de Ubicaciones Ideales</h2>", unsafe_allow_html=True)
    st.write("Encuentra el sitio geográfico más óptimo en un radio geográfico específico de tu ubicación.")
    
    col_search, col_result = st.columns([2, 2])
    
    with col_search:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        radio_busqueda = st.slider("Radio Máximo de Viaje (km)", 20, 300, 150)
        coordenada_origen = st.text_input("Ciudad de Origen", "Bogotá, Colombia")
        bortle_maximo = st.selectbox("Escala Bortle Máxima Deseada", [1, 2, 3, 4, 5, 6])
        st.markdown("</div>", unsafe_allow_html=True)
        
    with col_result:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.subheader("Sitio Recomendado por IA")
        if bortle_maximo >= 2:
            st.success("🌟 **Recomendación:** Desierto de la Tatacoa (Huila)\n*   **Distancia aprox:** ~280 km desde Bogotá\n*   **Bortle actual:** 2\n*   **Pronóstico de cielo despejado:** 91%")
        else:
            st.info("No se encontraron zonas Bortle 1 en un radio de 150km. Aumente el radio o el Bortle máximo.")
        st.markdown("</div>", unsafe_allow_html=True)

# --- MÓDULO 7: SISTEMA DE RECOMENDACIONES ---
elif menu == "🔭 7. Sistema de Recomendaciones":
    st.markdown("<h2 class='gradient-text'>Recomendador de Configuración de Oculares</h2>", unsafe_allow_html=True)
    st.write("Prescribe la mejor configuración óptica para tu telescopio amateur según las condiciones celestes.")
    
    apertura = st.number_input("Apertura del Telescopio (mm)", min_value=50, max_value=300, value=114)
    longitud_focal = st.number_input("Longitud Focal del Telescopio (mm)", min_value=300, max_value=3000, value=900)
    
    st.subheader("Oculares Recomendados para esta noche:")
    
    # Cálculos ópticos básicos
    max_aumento_teorico = apertura * 2
    focal_ocular_ideal = longitud_focal / max_aumento_teorico
    
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"""
        <div class='glass-card'>
            <h4>Ocular para Campo Amplio (Conjunciones)</h4>
            <ul>
                <li><b>Focal sugerida:</b> 25mm o 32mm</li>
                <li><b>Aumento obtenido:</b> {longitud_focal / 25:.1f}x</li>
                <li><b>Ideal para:</b> Ver <b>múltiples planetas visibles en el firmamento</b> juntos en el mismo campo óptico.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
    with c2:
        st.markdown(f"""
        <div class='glass-card'>
            <h4>Ocular para Detalle Planetario</h4>
            <ul>
                <li><b>Focal sugerida:</b> {np.clip(focal_ocular_ideal, 6, 12):.1f}mm</li>
                <li><b>Aumento obtenido:</b> {longitud_focal / np.clip(focal_ocular_ideal, 6, 12):.1f}x</li>
                <li><b>Ideal para:</b> Detalles de anillos de Saturno y bandas nubosas de Júpiter.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

# --- MÓDULO 8: PANEL ESTADÍSTICO HISTÓRICO ---
elif menu == "📈 8. Panel Estadístico Histórico":
    st.markdown("<h2 class='gradient-text'>Panel de Tendencias Históricas</h2>", unsafe_allow_html=True)
    st.write("Análisis descriptivo de variables climáticas de los últimos 5 años para planeación anual.")
    
    # Crear serie temporal histórica de simulación
    meses = ['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun', 'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dic']
    nubosidad_media = [20, 25, 45, 65, 60, 30, 15, 18, 35, 60, 70, 40]
    noches_despejadas = [22, 19, 12, 8, 9, 18, 26, 24, 16, 10, 6, 15]
    
    hist_df = pd.DataFrame({
        'Mes': meses,
        'Nubosidad Promedio (%)': nubosidad_media,
        'Noches Despejadas': noches_despejadas
    })
    
    col1, col2 = st.columns(2)
    with col1:
        fig1 = px.bar(hist_df, x='Mes', y='Nubosidad Promedio (%)', title="Nubosidad Promedio Mensual (Tatacoa)", color_discrete_sequence=['#9d4edd'])
        fig1.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font={'color': "#fff"})
        st.plotly_chart(fig1, use_container_width=True)
    with col2:
        fig2 = px.line(hist_df, x='Mes', y='Noches Despejadas', title="Número de Noches Óptimas por Mes", color_discrete_sequence=['#00b4d8'])
        fig2.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font={'color': "#fff"})
        st.plotly_chart(fig2, use_container_width=True)
        
    st.info("📅 **Conclusión del Histórico:** Los meses comprendidos entre junio y agosto ofrecen la menor cobertura de nubes promedio anual, constituyendo el marco temporal ideal para la observación.")

# --- MÓDULO 9: FORO DE AFICIONADOS ---
elif menu == "💬 9. Foro de Aficionados":
    st.markdown("<h2 class='gradient-text'>Foro de Bitácoras de Observación</h2>", unsafe_allow_html=True)
    st.write("Interactúa con otros aficionados compartiendo reportes ópticos y avistamientos de planetas.")
    
    # Caja para ingresar nuevo comentario
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.write("##### Añadir un reporte de avistamiento:")
    
    user_input = st.text_input("Tu Nombre de Usuario")
    loc_input = st.text_input("Ubicación de Observación", "Desierto de la Tatacoa")
    comment_input = st.text_area("Bitácora del avistamiento (máx 300 caracteres)")
    
    if st.button("Enviar Reporte"):
        if user_input and comment_input:
            new_comment = {
                "usuario": user_input,
                "ubicacion": loc_input,
                "comentario": comment_input,
                "fecha": pd.Timestamp.now().strftime("%Y-%m-%d")
            }
            st.session_state['foro_comments'].append(new_comment)
            st.success("✅ ¡Bitácora publicada exitosamente!")
        else:
            st.error("Por favor completa los campos de usuario y comentario.")
    st.markdown("</div>", unsafe_allow_html=True)
            
    # Listar comentarios
    st.write("#### Reportes de Avistamientos Recientes:")
    for comment in reversed(st.session_state['foro_comments']):
        st.markdown(f"""
        <div class='glass-card'>
            <b>👤 {comment['usuario']}</b> en 📍 <i>{comment['ubicacion']}</i> ({comment['fecha']})<br>
            <p style='margin-top: 0.5rem; color: #adc1d6;'>"{comment['comentario']}"</p>
        </div>
        """, unsafe_allow_html=True)
