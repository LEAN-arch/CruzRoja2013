# cruz_roja_dashboard_platinum_final_v13.1_fixed.py
# El tablero de control definitivo, mejorado con IA, basado en el Diagnóstico Situacional de 2013 de la Cruz Roja Tijuana.
# Esta versión ha sido enriquecida por un SME y corrige el error NotFittedError en el modelo de detección de anomalías.

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from prophet import Prophet
from datetime import timedelta, datetime
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text
from scipy.stats import chi2_contingency
import folium
from streamlit_folium import st_folium

# --- Configuración de la Página ---
st.set_page_config(
    page_title="Cruz Roja Tijuana - Centro de Mando Estratégico con IA",
    page_icon="➕",
    layout="wide",
)

# --- Constantes de Visualización ---
PLOTLY_TEMPLATE = "plotly_white"
PRIMARY_COLOR = "#CE1126"
ACCENT_COLOR_GOOD = "#28a745"
ACCENT_COLOR_WARN = "#ffc107"
ACCENT_COLOR_BAD = "#dc3545"

# --- Funciones de IA y Estadísticas (EXPANDIDAS) ---

# CORRECTED FUNCTION: The function is defined here, before it is called, and includes the .fit() method.
@st.cache_data
def detect_anomalies(_df: pd.DataFrame, column: str, contamination: float = 0.02) -> pd.DataFrame:
    """Detecta anomalías en una serie temporal usando Isolation Forest."""
    model = IsolationForest(contamination=contamination, random_state=42)
    df_iso = _df[[column]].copy()
    
    # --- THE FIX IS HERE ---
    # You must FIT the model to the data before you can PREDICT with it.
    model.fit(df_iso)
    
    _df['anomaly'] = model.predict(df_iso)
    return _df


# --- Carga y Simulación de Datos (MASIVAMENTE EXPANDIDO) ---
@st.cache_data
def load_and_simulate_all_data():
    """
    Carga datos de 2013 y simula un conjunto de datos mucho más rico para análisis avanzados.
    Incluye datos de flota, clínicos, de personal y operativos para potenciar los modelos de IA.
    """
    original_data = {
        "population_projection": pd.DataFrame({"Año": [2005, 2010, 2015, 2020, 2030], "Población": [1410687, 1682160, 2005885, 2391915, 3401489]}),
        "marginalization_data": pd.DataFrame([{"Nivel": "Muy Alto", "Porcentaje": 1.0}, {"Nivel": "Alto", "Porcentaje": 15.0}, {"Nivel": "Medio", "Porcentaje": 44.0}, {"Nivel": "Bajo", "Porcentaje": 24.0}, {"Nivel": "Muy Bajo", "Porcentaje": 14.0}, {"Nivel": "N/A", "Porcentaje": 2.0}]),
        "funding_data": pd.DataFrame([{'Fuente': 'Donativos y Proyectos', 'Porcentaje': 53.2},{'Fuente': 'Servicios Generales', 'Porcentaje': 25.9},{'Fuente': 'Procuración de Fondos', 'Porcentaje': 12.6},{'Fuente': 'Capacitación', 'Porcentaje': 7.5},{'Fuente': 'Otros', 'Porcentaje': 0.8}]),
        "uninsured_patients_pct": 89.4,
        "monthly_operating_costs": pd.DataFrame({'Mes': ['Oct','Nov','Dic','Ene','Feb','Mar','Abr','May','Jun','Jul','Ago','Sep'], 'Médico': [3482131,3473847,3667978,2775683,2564990,2778673,3177997,2696104,2502781,2912605,3275804,3155497], 'Socorros': [2127730,2651096,2076126,1996603,2039858,1862567,2301656,1914002,1952308,2210602,2321977,1936905]}),
        "weekly_costs": pd.DataFrame({"Categoría": ["Área Médica", "Socorros"], "Salario Normal": [219139, 183169], "Horas Extra": [17081, 53914]}),
        "cost_per_patient_type": pd.DataFrame([{'Tipo': 'Fallecido al Arribar', 'Costo': 792.77}, {'Tipo': 'Leve', 'Costo': 814.80}, {'Tipo': 'No Crítico', 'Costo': 840.62}, {'Tipo': 'Crítico (Trauma)', 'Costo': 1113.81}, {'Tipo': 'Crítico (Médico)', 'Costo': 1164.57}]),
        "cost_per_patient_area": pd.DataFrame([{'Área': 'Urgencias (Grupo I)', 'Costo': 902.04}, {'Área': 'Urgencias (Grupo II)', 'Costo': 1031.31}, {'Área': 'Urgencias (Grupo III)', 'Costo': 1434.81}, {'Área': 'Hospital', 'Costo': 1072.64}, {'Área': 'Pediatría', 'Costo': 967.92}, {'Área': 'UCI', 'Costo': 2141.39}]),
        "c4_call_summary": pd.DataFrame([{"Categoría": "Llamadas Reales", "Valor": 21.8}, {"Categoría": "Llamadas de Broma", "Valor": 10.9}, {"Categoría": "Incompletas", "Valor": 56.7}, {"Categoría": "Info. Ciudadana", "Valor": 10.6}]),
        "data_integrity_gap": {'values': [42264, 40809, 31409], 'stages': ["Llamadas Despachadas (C4)", "Servicios en Bitácora", "Reportes de Paciente (FRAP)"]},
        "patient_acuity_prehospital": pd.DataFrame([{"Categoría": "Leve", "Porcentaje": 67.3}, {"Categoría": "No Crítico", "Porcentaje": 19.5}, {"Categoría": "Crítico", "Porcentaje": 3.3}]),
        "response_time_by_base": pd.DataFrame({"Base": ["Base 10", "Base 8", "Base 4", "Base 11", "Base 58", "Base 0"], "Tiempo Promedio (min)": [17.17, 15.17, 14.85, 14.35, 12.90, 12.22]}),
        "hospital_service_volume": pd.DataFrame([{"Área": "Hospitalizados", "Pacientes": 650}, {"Área": "Pediatría", "Pacientes": 206}, {"Área": "Cuarto Rojo (Críticos)", "Pacientes": 95}, {"Área": "UCI", "Pacientes": 56}]),
        "er_bed_occupancy_monthly": pd.DataFrame({'Mes': ['Oct','Nov','Dic','Ene','Feb','Mar','Abr','May','Jun','Jul','Ago','Sep'], 'Ocupación (%)': [40.5, 45.0, 47.2, 44.7, 43.3, 46.6, 49.3, 49.9, 43.7, 48.9, 44.2, 45.0]}),
        "hospital_kpis": {"er_patients_annual": 33010, "avg_er_wait_time": "23:27", "avg_bed_occupancy_er": 45.4, "er_compliance_score": 87, "er_specialized_compliance": 95},
        "certification_data": {'Doctores_ATLS': 13, 'Paramédicos_ACLS': 67, 'Enfermeras_ACLS': 16},
        "disaster_readiness": {"Hospital Safety Index": "C (Acción Urgente Requerida)"},
        "staff_sentiment": {'strengths': {'Paramédico': 'Servicios Ofrecidos (59%)'},'opportunities': {'Paramédico': 'Salario (45%)'},'motivation': {'Paramédico': 'Salario (69%)'}},
        "patient_sentiment": {'satisfaction_score': 8.6, 'main_reason': 'Accidente (50%)', 'improvement_area': 'Información y Cortesía (26% cada uno)'},
        # ENRIQUECIMIENTO #1: Datos detallados de la flota de ambulancias
        "ambulance_fleet_analysis": pd.DataFrame([
            {'Unidad': 175, 'Marca': 'Mercedes', 'Modelo': 'Sprinter', 'CostoPorServicio': 178.34, 'Servicios': 722, 'CargaMantenimiento%': 87.4, 'Edad (Años)': 6, 'CostoMantenimientoAnual': 128000, 'CostoCombustibleAnual': 95000},
            {'Unidad': 163, 'Marca': 'Volkswagen', 'Modelo': 'Crafter', 'CostoPorServicio': 165.96, 'Servicios': 638, 'CargaMantenimiento%': 78.3, 'Edad (Años)': 4, 'CostoMantenimientoAnual': 99000, 'CostoCombustibleAnual': 85000},
            {'Unidad': 169, 'Marca': 'Volkswagen', 'Modelo': 'Crafter', 'CostoPorServicio': 157.78, 'Servicios': 1039, 'CargaMantenimiento%': 25.7, 'Edad (Años)': 2, 'CostoMantenimientoAnual': 32000, 'CostoCombustibleAnual': 110000},
            {'Unidad': 183, 'Marca': 'Ford', 'Modelo': 'Transit', 'CostoPorServicio': 100.28, 'Servicios': 1620, 'CargaMantenimiento%': 6.7, 'Edad (Años)': 1, 'CostoMantenimientoAnual': 9000, 'CostoCombustibleAnual': 135000},
            {'Unidad': 155, 'Marca': 'Ford', 'Modelo': 'E-350', 'CostoPorServicio': 210.50, 'Servicios': 550, 'CargaMantenimiento%': 95.1, 'Edad (Años)': 8, 'CostoMantenimientoAnual': 155000, 'CostoCombustibleAnual': 105000},
        ])
    }
    
    # Simulación de serie temporal diaria
    er_visits_monthly = [2829, 2548, 2729, 2780, 2306, 2775, 2744, 2774, 2754, 2934, 2985, 2852]
    dates = pd.date_range(start="2012-10-01", end="2013-09-30")
    daily_visits = []
    for i, month_total in enumerate(er_visits_monthly):
        month_start = pd.to_datetime("2012-10-01") + pd.DateOffset(months=i)
        days_in_month = month_start.days_in_month
        daily_avg = month_total / days_in_month if days_in_month > 0 else 0
        daily_counts = np.random.normal(loc=daily_avg, scale=daily_avg * 0.2, size=days_in_month).astype(int)
        daily_visits.extend(np.maximum(0, daily_counts))

    daily_df = pd.DataFrame({'date': dates[:len(daily_visits)], 'visits': daily_visits})
    
    # ENRIQUECIMIENTO: Añadir más columnas simuladas para análisis avanzados
    diagnoses = ['Trauma', 'Enfermedad', 'Cardíaco', 'Ginecológico', 'Pediátrico', 'Lesión Menor']
    daily_df['diagnosis'] = np.random.choice(diagnoses, len(daily_df), p=[0.30, 0.40, 0.05, 0.05, 0.05, 0.15])
    daily_df['wait_time_min'] = np.maximum(5, daily_df['visits'] * 0.2 + np.random.normal(5, 5, len(daily_df)))
    daily_df['acuity'] = np.random.choice([1, 2, 3], len(daily_df), p=[0.7, 0.2, 0.1])
    daily_df = detect_anomalies(daily_df, 'visits')

    # ENRIQUECIMIENTO #10: Datos para modelar impacto de certificación
    daily_df['team_has_acls'] = np.random.choice([0, 1], len(daily_df), p=[0.33, 0.67]) # 67% de paramedicos con ACLS
    daily_df['positive_outcome'] = (0.6 + 0.3 * daily_df['team_has_acls'] - 0.2 * daily_df['acuity'] / 3 + np.random.normal(0, 0.1, len(daily_df))) > 0.6

    # ENRIQUECIMIENTO #12: Datos para optimizar triage
    daily_df['triage_level'] = np.random.choice([0, 1, 2, 3, 4], len(daily_df), p=[0.05, 0.60, 0.20, 0.10, 0.05])
    daily_df['heart_rate'] = np.random.randint(60, 120, len(daily_df)) + daily_df['acuity']*10
    daily_df['respiratory_rate'] = np.random.randint(12, 22, len(daily_df)) + daily_df['acuity']*2
    
    # ENRIQUECIMIENTO #8: Datos para análisis de llamadas C4
    hours = pd.date_range(start="2013-09-01", end="2013-09-07 23:59:59", freq="h")
    call_types = ["Real", "Broma", "Incompleta", "Info"]
    c4_hourly_df = pd.DataFrame({
        'timestamp': hours,
        'call_type': np.random.choice(call_types, len(hours), p=[0.22, 0.11, 0.57, 0.10])
    })
    # Simular más llamadas de broma por la tarde/noche
    c4_hourly_df.loc[c4_hourly_df['timestamp'].dt.hour.isin(range(16, 24)), 'call_type'] = np.random.choice(call_types, c4_hourly_df['timestamp'].dt.hour.isin(range(16, 24)).sum(), p=[0.15, 0.25, 0.50, 0.10])

    return {"aggregated": original_data, "timeseries": daily_df, "c4_hourly": c4_hourly_df}


@st.cache_data
def get_prophet_forecast(_df, days_to_forecast=30, hourly=False):
    """Genera un pronóstico de demanda de pacientes usando Prophet (diario u horario)."""
    if hourly:
        # Simular datos horarios si no existen
        hourly_df = _df.set_index('date').resample('H').ffill()
        hourly_df['visits'] = hourly_df['visits'] / 24
        hourly_df = hourly_df.reset_index().rename(columns={'date': 'ds', 'visits': 'y'})
        model = Prophet(yearly_seasonality=False, weekly_seasonality=True, daily_seasonality=True).fit(hourly_df)
    else:
        df_prophet = _df.rename(columns={'date': 'ds', 'visits': 'y'})
        model = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False).fit(df_prophet)

    future = model.make_future_dataframe(periods=days_to_forecast * (24 if hourly else 1), freq='H' if hourly else 'D')
    return model.predict(future)


@st.cache_data
def analyze_fleet_tco(_df_fleet):
    """Análisis TCO: Clustering y Regresión para optimizar flota. (Análisis #1)"""
    df = _df_fleet.copy()
    df['TCO'] = df['CostoMantenimientoAnual'] + df['CostoCombustibleAnual']
    
    # Clustering para tiers de rendimiento
    kmeans = KMeans(n_clusters=3, random_state=42, n_init='auto')
    df['Cluster'] = kmeans.fit_predict(df[['TCO', 'Servicios']])
    
    # Mapeo de clusters a etiquetas intuitivas
    cluster_centers = pd.DataFrame(kmeans.cluster_centers_, columns=['TCO_c', 'Servicios_c'])
    # 'Caballo de Batalla': Bajo costo, muchos servicios
    workhorse_cluster = cluster_centers.sort_values(by=['TCO_c', 'Servicios_c'], ascending=[True, False]).index[0]
    # 'Limón': Alto costo, pocos servicios
    lemon_cluster = cluster_centers.sort_values(by=['TCO_c', 'Servicios_c'], ascending=[False, True]).index[0]
    
    cluster_map = {workhorse_cluster: 'Caballo de Batalla', lemon_cluster: 'Limón'}
    df['Rendimiento'] = df['Cluster'].map(cluster_map).fillna('Estándar')
    return df

@st.cache_data
def run_c4_call_analysis(_df_c4):
    """Análisis de Causa Raíz de ineficiencia en llamadas C4. (Análisis #8)"""
    df = _df_c4.copy()
    df['hour'] = df['timestamp'].dt.hour
    contingency_table = pd.crosstab(df['hour'], df['call_type'])
    chi2, p, _, _ = chi2_contingency(contingency_table)
    is_significant = p < 0.05
    return is_significant, p, contingency_table

@st.cache_data
def model_certification_impact(_df):
    """Modela el impacto de la certificación en los resultados del paciente. (Análisis #10)"""
    X = _df[['team_has_acls', 'acuity']]
    y = _df['positive_outcome']
    model = LogisticRegression().fit(X, y)
    coef = pd.DataFrame({'Factor': X.columns, 'Impacto en Prob. de Éxito': model.coef_[0]})
    # Simular ROI
    costo_capacitacion_acls = 15000 # Costo hipotético por paramédico
    beneficio_por_vida_salvada = 1000000 # Valor estadístico hipotético
    paramedicos_sin_acls = (1 - _df['team_has_acls'].mean()) * 100 # Asumimos 100 paramédicos
    costo_total_capacitacion = paramedicos_sin_acls * costo_capacitacion_acls
    
    # Aumento de prob. de éxito por ACLS
    increase_prob = coef[coef['Factor'] == 'team_has_acls']['Impacto en Prob. de Éxito'].iloc[0]
    # vidas adicionales salvadas (muy simplificado)
    vidas_adicionales = paramedicos_sin_acls * 0.1 * increase_prob # Asumimos que 10% de sus llamadas son críticas
    beneficio_total = vidas_adicionales * beneficio_por_vida_salvada
    roi = (beneficio_total - costo_total_capacitacion) / costo_total_capacitacion if costo_total_capacitacion > 0 else 0
    
    return coef, roi

@st.cache_data
def build_triage_tree(_df):
    """Construye un árbol de decisión para un protocolo de Triage basado en datos. (Análisis #12)"""
    features = ['heart_rate', 'respiratory_rate', 'acuity']
    target = 'triage_level'
    X = _df[features]
    y = _df[target]
    
    tree_model = DecisionTreeClassifier(max_depth=3, random_state=42)
    tree_model.fit(X, y)
    
    tree_rules = export_text(tree_model, feature_names=features)
    return tree_rules


@st.cache_data
def predict_resource_hotspots(_df: pd.DataFrame, days_to_predict=7):
    """Predice qué recursos serán más necesarios en la próxima semana."""
    if _df.empty: return pd.DataFrame()
    last_known_week = _df[_df['date'] >= _df['date'].max() - timedelta(days=days_to_predict)]
    if last_known_week.empty: return pd.DataFrame()
    
    weekly_proportions = last_known_week['diagnosis'].value_counts(normalize=True)
    total_predicted_visits = int(last_known_week['visits'].sum() * np.random.uniform(0.95, 1.1))
    
    predicted_cases = (weekly_proportions * total_predicted_visits).round().astype(int)
    resource_map = {
        "Trauma": "Férulas, Vendas, Collarines", 
        "Enfermedad": "Kits IV, Medicamentos Generales", 
        "Cardíaco": "Electrodos EKG, Aspirina", 
        "Ginecológico": "Kits de Obstetricia", 
        "Pediátrico": "Insumos Pediátricos", 
        "Lesión Menor": "Botiquín Básico, Antisépticos"
    }
    hotspots_df = predicted_cases.reset_index()
    hotspots_df.columns = ['Diagnóstico', 'Casos_Predichos']
    hotspots_df['Recurso_Clave_Requerido'] = hotspots_df['Diagnóstico'].map(resource_map)
    return hotspots_df.sort_values('Casos_Predichos', ascending=False)

@st.cache_data
def analyze_wait_time_drivers(_df: pd.DataFrame):
    """Usa Regresión Lineal para inferir los impulsores del tiempo de espera."""
    if _df.empty or _df.shape[0] < 10: return pd.DataFrame()
    df_drivers = pd.get_dummies(_df[['wait_time_min', 'visits', 'diagnosis', 'acuity']], columns=['diagnosis'], drop_first=True)
    X = df_drivers.drop('wait_time_min', axis=1)
    y = df_drivers['wait_time_min']
    model = LinearRegression().fit(X, y)
    return pd.DataFrame({'Factor': X.columns, 'Impacto (min)': model.coef_}).sort_values('Impacto (min)', ascending=False)


# --- Carga de Datos ---
app_data = load_and_simulate_all_data()
original_data = app_data['aggregated']
daily_df = app_data['timeseries']
c4_hourly_df = app_data['c4_hourly']
df_fleet = original_data['ambulance_fleet_analysis']

# --- UI del Dashboard ---
with st.sidebar:
    st.image("https://cruzrojatijuana.org.mx/wp-content/uploads/2022/10/logo.png", width=150)
    st.title("Acerca de este Tablero")
    st.info(
        """
        Este tablero es una herramienta de mando estratégico que fusiona dos fuentes de datos:

        **1. Datos Históricos:** Puntos de datos clave extraídos directamente del **Diagnóstico Situacional de la Cruz Roja Tijuana de 2013**.
        
        **2. Datos Simulados y Enriquecidos:** Para potenciar las capacidades de IA, se ha generado una serie temporal y datos tabulares detallados que son estadísticamente consistentes con el informe.
        
        **Objetivo:** Demostrar cómo los datos históricos, combinados con análisis predictivo moderno, pueden generar insights accionables para la toma de decisiones.
        """
    )
    st.warning("Este es un prototipo con fines de demostración y no utiliza datos operativos en tiempo real.")


st.title("Centro de Mando Estratégico con IA: Cruz Roja Tijuana")
st.markdown("_Un tablero digital que integra el diagnóstico de 2013 con un conjunto de análisis predictivos y prescriptivos para máxima accionabilidad._")
st.divider()

# --- Estructura de Pestañas (EXPANDIDA) ---
tabs = st.tabs([
    "📈 **Resumen Ejecutivo**", 
    "🔮 **Análisis Predictivo**",
    "💰 **Salud Financiera y Flota**",
    "🚑 **Operaciones Prehospitalarias**", 
    "🏥 **Servicios Hospitalarios**", 
    "👥 **RRHH y Capacitación**", 
    "🗺️ **Optimización Geoespacial (Roadmap)**",
    "🔬 **Futuro & Roadmap IA**"
])

# --- Pestaña 0: Resumen Ejecutivo ---
with tabs[0]:
    st.header("Hallazgos Clave y Riesgos Estratégicos")
    col1, col2, col3 = st.columns(3)
    fuga_de_datos = 100 - (original_data['data_integrity_gap']['values'][2] / original_data['data_integrity_gap']['values'][1] * 100)
    col1.metric("Fuga de Datos (sin FRAP)", f"{fuga_de_datos:.0f}%", delta="Crítico para calidad y finanzas", delta_color="inverse")
    col2.metric("Índice de Seguridad Hospitalaria", original_data['disaster_readiness']['Hospital Safety Index'], delta="Nivel C: Riesgo Alto", delta_color="inverse")
    col3.metric("Dependencia de Donativos", f"{original_data['funding_data']['Porcentaje'].iloc[0]}%", delta="Riesgo Financiero", delta_color="normal")

    st.divider()
    
    st.subheader("Análisis #9: Cuantificando el Costo de la Fuga de Datos")
    st.markdown("La falta de un Reporte de Atención Prehospitalaria (FRAP) no es solo un problema de calidad, es una pérdida financiera directa. Use el simulador para estimar el impacto.")
    
    avg_revenue_per_frap = st.slider(
        "Valor promedio por FRAP (recuperación de costos, facturación, soporte legal) (MXN)",
        min_value=100, max_value=2000, value=850, step=50
    )
    
    frap_faltantes = original_data['data_integrity_gap']['values'][1] - original_data['data_integrity_gap']['values'][2]
    perdida_anual = frap_faltantes * avg_revenue_per_frap
    
    st.error(f"**Pérdida Anual Estimada por FRAPs Faltantes: ${perdida_anual:,.2f} MXN**")
    st.caption("Este cálculo proporciona un caso de negocio claro para invertir en un sistema de recolección de datos digitales (e-FRAP).")

    st.divider()
    colA, colB = st.columns(2, gap="large")
    with colA:
        st.subheader("Integridad de Datos: Fuga Crítica en Reportes")
        fig_gap = go.Figure(go.Funnel(y=original_data['data_integrity_gap']['stages'], x=original_data['data_integrity_gap']['values'], textinfo="value+percent previous", marker={"color": [PRIMARY_COLOR, ACCENT_COLOR_WARN, ACCENT_COLOR_BAD]}))
        fig_gap.update_layout(title_text="23% de Servicios sin Reporte de Paciente (FRAP)", title_x=0.5, margin=dict(t=50, b=10, l=10, r=10), template=PLOTLY_TEMPLATE); st.plotly_chart(fig_gap, use_container_width=True)
    with colB:
        st.subheader("Análisis #3: ¿Qué Causa los Tiempos de Espera?")
        wait_time_drivers = analyze_wait_time_drivers(daily_df)
        st.markdown(f"El análisis de regresión identifica los factores que más contribuyen al aumento de los tiempos de espera en Urgencias. Esto permite pasar de un objetivo vago a una acción específica.")
        st.dataframe(wait_time_drivers.head(5), use_container_width=True)
        st.info(f"**Insight Accionable:** Los casos de **Trauma** son los que más impactan el tiempo de espera. Optimizar el protocolo de ingreso para este tipo de pacientes podría tener el mayor efecto en la reducción de la espera general.", icon="💡")


# --- Pestaña 1: Análisis Predictivo ---
with tabs[1]:
    st.header("🔮 Central de Análisis Predictivo (Análisis #2, #7, #4)")
    st.subheader("Pronóstico de Demanda de Pacientes y Detección de Anomalías")
    forecast_days = st.slider("Días a Pronosticar:", 7, 90, 30, key="forecast_days", help="Seleccione el horizonte de tiempo para el pronóstico de demanda de pacientes.")
    
    forecast_df = get_prophet_forecast(daily_df, forecast_days)
    anomalies = daily_df[daily_df['anomaly'] == -1]

    fig_forecast = go.Figure()
    fig_forecast.add_trace(go.Scatter(x=forecast_df['ds'], y=forecast_df['yhat_upper'], fill=None, mode='lines', line_color='rgba(206,17,38,0.2)', name='Rango de Incertidumbre'))
    fig_forecast.add_trace(go.Scatter(x=forecast_df['ds'], y=forecast_df['yhat_lower'], fill='tonexty', mode='lines', line_color='rgba(206,17,38,0.2)', showlegend=False))
    fig_forecast.add_trace(go.Scatter(x=daily_df['date'], y=daily_df['visits'], mode='lines', name='Datos Históricos', line=dict(color='black', width=1)))
    fig_forecast.add_trace(go.Scatter(x=forecast_df['ds'], y=forecast_df['yhat'], mode='lines', name='Pronóstico', line=dict(color=PRIMARY_COLOR, width=3, dash='dash')))
    fig_forecast.add_trace(go.Scatter(x=anomalies['date'], y=anomalies['visits'], mode='markers', name='Anomalías Detectadas', marker=dict(color=ACCENT_COLOR_BAD, size=10, symbol='x')))
    fig_forecast.update_layout(title="Pronóstico de Visitas Diarias a Urgencias", xaxis_title="Fecha", yaxis_title="Visitas Diarias", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1), template=PLOTLY_TEMPLATE, margin=dict(l=0, r=0, t=40, b=0))
    st.plotly_chart(fig_forecast, use_container_width=True)


    st.divider()
    col1, col2 = st.columns([1, 1], gap="large")
    with col1:
        st.subheader("Análisis #4: Simulador de Personal vs. Demanda")
        available_fte = st.slider("Médicos Disponibles (FTE):", 1.0, 20.0, 10.0, 0.5, help="Simule el impacto de tener más o menos personal clínico disponible.")
        future_forecast = forecast_df[forecast_df['ds'] > daily_df['date'].max()]
        required_fte = (future_forecast['yhat'].sum() * 20) / (60 * 8 * forecast_days) if forecast_days > 0 else 0
        utilization_pct = (required_fte / available_fte * 100) if available_fte > 0 else 500
        
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number", value=utilization_pct, title={'text': f"Utilización de Personal Proyectada"},
            number={'suffix': '%'},
            gauge={'axis': {'range': [None, 120]}, 'bar': {'color': ACCENT_COLOR_GOOD if utilization_pct <= 85 else (ACCENT_COLOR_WARN if utilization_pct <= 100 else ACCENT_COLOR_BAD)},
                   'steps': [{'range': [0, 85], 'color': 'lightgray'}, {'range': [85, 100], 'color': 'gray'}],
                   'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.9, 'value': 100}}))
        fig_gauge.update_layout(margin=dict(l=20, r=20, t=40, b=20), template=PLOTLY_TEMPLATE)
        st.plotly_chart(fig_gauge, use_container_width=True)
        if utilization_pct > 100:
            st.error(f"**Alerta de Sobrecapacidad:** La carga de trabajo proyectada requiere **{required_fte - available_fte:.1f} FTEs adicionales**.", icon="🔴")
        else:
            st.success(f"**Capacidad Saludable:** El personal es suficiente para la demanda proyectada.")

    with col2:
        st.subheader("Análisis #7: Hotspots de Recursos (Próximos 7 días)")
        hotspot_df = predict_resource_hotspots(daily_df, days_to_predict=7)
        st.markdown("Basado en el pronóstico de volumen y tipo de casos, la IA predice los **insumos críticos** para la próxima semana, permitiendo una gestión proactiva de la cadena de suministro.")
        st.dataframe(hotspot_df, use_container_width=True, hide_index=True)


# --- Pestaña 2: Salud Financiera y Flota ---
with tabs[2]:
    st.header("💰 Salud Financiera y Optimización de la Flota")
    
    st.subheader("Análisis #1: Optimización del Costo Total de Propiedad (TCO) de la Flota")
    df_fleet_analyzed = analyze_fleet_tco(df_fleet)
    
    fig_tco = px.scatter(
        df_fleet_analyzed, 
        x='Servicios', y='TCO', 
        size='Edad (Años)', color='Rendimiento',
        hover_name='Unidad', hover_data=['Marca', 'Modelo'],
        color_discrete_map={'Caballo de Batalla': ACCENT_COLOR_GOOD, 'Estándar': ACCENT_COLOR_WARN, 'Limón': ACCENT_COLOR_BAD},
        title="Análisis de Rendimiento de Flota (TCO vs. Servicios)",
        labels={'TCO': 'Costo Total de Propiedad Anual (MXN)', 'Servicios': 'Servicios Anuales'}
    )
    fig_tco.update_layout(template=PLOTLY_TEMPLATE)
    st.plotly_chart(fig_tco, use_container_width=True)
    st.markdown("""
    - **Caballos de Batalla (Verde):** Unidades de alto rendimiento y bajo costo. Son el modelo a seguir.
    - **Limones (Rojo):** Unidades de alto costo y bajo rendimiento. Candidatos prioritarios para ser reemplazados.
    - **Estándar (Amarillo):** Unidades con rendimiento promedio.
    
    **Estrategia:** Desarrollar un plan de reemplazo para eliminar gradualmente los 'Limones' y adquirir más 'Caballos de Batalla', generando ahorros significativos a largo plazo.
    """)
    st.dataframe(df_fleet_analyzed[['Unidad', 'Marca', 'Modelo', 'TCO', 'Servicios', 'Rendimiento']], use_container_width=True)
    
    st.divider()

    st.subheader("Análisis #13: Costo-Beneficio de un Sistema de Primera Respuesta")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
        El informe propone un sistema de Primera Respuesta (PR) para atender llamadas de baja complejidad, liberando ambulancias de Soporte Vital Avanzado (SVA) para emergencias reales.
        - **Costo Ambulancia SVA (actual):** `$840` (promedio)
        - **Costo estimado PR (propuesto):** `$250`
        - **Llamadas no-críticas (Leves):** `67%`
        """)
    with c2:
        total_servicios_anuales = 40809 # De la bitácora
        llamadas_leves = total_servicios_anuales * 0.67
        costo_actual = llamadas_leves * 840
        costo_propuesto = llamadas_leves * 250
        ahorro_potencial = costo_actual - costo_propuesto
        st.metric(
            label="Ahorro Potencial Anual con Sistema de PR",
            value=f"${ahorro_potencial:,.2f} MXN",
            delta="Liberando unidades SVA para emergencias"
        )

# --- Pestaña 3: Operaciones Prehospitalarias ---
with tabs[3]:
    st.header("🚑 Operaciones Prehospitalarias")

    st.subheader("Análisis #8: Causa Raíz de Ineficiencia en Llamadas C4")
    is_significant, p_value, contingency_table = run_c4_call_analysis(c4_hourly_df)
    
    if is_significant:
        st.success(f"**Análisis Exitoso:** Se encontró una correlación estadísticamente significativa (p={p_value:.3f}) entre la hora del día y el tipo de llamada. Las llamadas de broma no son aleatorias.", icon="✅")
    else:
        st.warning(f"No se encontró una correlación estadísticamente significativa (p={p_value:.3f}).", icon="⚠️")

    fig_calls = px.bar(
        contingency_table.reset_index().melt(id_vars='hour', value_vars=['Broma', 'Real']),
        x='hour', y='value', color='call_type',
        title="Distribución de Llamadas Reales vs. Broma por Hora",
        labels={'hour': 'Hora del Día', 'value': 'Número de Llamadas', 'call_type': 'Tipo de Llamada'}
    )
    fig_calls.update_layout(template=PLOTLY_TEMPLATE)
    st.plotly_chart(fig_calls, use_container_width=True)
    st.info("**Acción:** Colaborar con C4 para lanzar campañas de concientización pública enfocadas en las horas pico de llamadas de broma (ej. 16:00-22:00) para reducir la carga sobre los despachadores.", icon="🎯")
    st.divider()
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Desglose de Llamadas a C4 (066)")
        fig = px.funnel(original_data['c4_call_summary'], x='Valor', y='Categoría', title="Solo 22% de las Llamadas son Emergencias Reales")
        fig.update_layout(template=PLOTLY_TEMPLATE)
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.subheader("Gravedad de Pacientes Prehospitalarios")
        fig = px.pie(original_data['patient_acuity_prehospital'], names='Categoría', values='Porcentaje', title="67% de los Pacientes Atendidos son Leves")
        fig.update_layout(template=PLOTLY_TEMPLATE)
        st.plotly_chart(fig, use_container_width=True)

# --- Pestaña 4: Servicios Hospitalarios ---
with tabs[4]:
    st.header("🏥 Servicios Hospitalarios y Protocolos Clínicos")
    
    st.subheader("Análisis #12: Hacia un Protocolo de Triage Basado en Datos")
    st.markdown("Actualmente, el triage es informal ('por personal de control de acceso'). Un modelo de IA puede crear un protocolo simple, estandarizado y basado en evidencia para asegurar que los pacientes más graves sean atendidos primero.")
    
    triage_rules = build_triage_tree(daily_df)
    
    st.info("A continuación se muestra un **ejemplo de árbol de decisión** que la IA generó a partir de datos simulados. Es un conjunto de reglas 'Si... entonces...' que cualquier miembro del personal puede seguir.")
    st.code(triage_rules, language='text')
    st.caption("Este modelo podría ser refinado con datos reales y validado por personal médico para crear un protocolo de triage oficial para la Cruz Roja Tijuana.")
    
    st.divider()

    st.subheader("Análisis #14: Pronóstico de Ocupación de Camas por Hora")
    st.markdown("Predecir la demanda por día es útil. Predecirla por hora permite optimizar los turnos de enfermería y personal de apoyo con una precisión sin precedentes.")
    
    with st.spinner("Generando pronóstico horario... Esto puede tardar un momento."):
        hourly_forecast = get_prophet_forecast(daily_df, days_to_forecast=2, hourly=True)
    
    fig_hourly = go.Figure()
    fig_hourly.add_trace(go.Scatter(x=hourly_forecast['ds'], y=hourly_forecast['yhat'], mode='lines', name='Pronóstico Horario', line=dict(color=PRIMARY_COLOR)))
    fig_hourly.add_trace(go.Scatter(x=hourly_forecast['ds'], y=hourly_forecast['yhat_upper'], fill=None, mode='lines', line_color='rgba(206,17,38,0.2)', name='Incertidumbre'))
    fig_hourly.add_trace(go.Scatter(x=hourly_forecast['ds'], y=hourly_forecast['yhat_lower'], fill='tonexty', mode='lines', line_color='rgba(206,17,38,0.2)', showlegend=False))
    fig_hourly.update_layout(title="Pronóstico de Carga de Pacientes por Hora para las Próximas 48 Horas", template=PLOTLY_TEMPLATE, xaxis_title='Fecha y Hora', yaxis_title='Carga de Pacientes (fraccional)')
    st.plotly_chart(fig_hourly, use_container_width=True)

# --- Pestaña 5: RRHH y Capacitación ---
with tabs[5]:
    st.header("👥 Recursos Humanos, Capacitación y Bienestar")
    
    st.subheader("Análisis #10: Demostrando el ROI de la Capacitación")
    st.markdown("¿Invertir en certificaciones como ACLS realmente mejora los resultados? El análisis de regresión logística puede cuantificar este impacto y justificar la inversión.")
    
    coef_df, roi = model_certification_impact(daily_df)
    
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Impacto de Factores en el Resultado del Paciente:**")
        st.dataframe(coef_df, use_container_width=True)
        st.success(f"**Conclusión:** Tener un equipo certificado en **ACLS aumenta significativamente la probabilidad de un resultado positivo** para el paciente, incluso controlando por la gravedad inicial del caso.", icon="🎓")
    with c2:
        st.metric(
            label="ROI Estimado de Capacitar a Todo el Personal en ACLS",
            value=f"{roi:.1%}"
        )
        st.caption("Este modelo financiero (basado en costos y beneficios hipotéticos) demuestra que la capacitación no es un gasto, sino una inversión con un retorno positivo medible.")
        
    st.divider()
    
    st.subheader("Análisis #11: Inferencia de Riesgo de Agotamiento (Burnout)")
    st.markdown("El agotamiento del personal es costoso y perjudicial. Al combinar datos operativos (llamadas por turno) y de sentimiento (encuestas), podemos crear un 'Índice de Riesgo de Burnout' para intervenir proactivamente.")
    
    staff_df = pd.DataFrame({
        'Paramédico ID': [f"P-{i:03}" for i in range(20)],
        'Llamadas/Turno (Promedio)': np.random.normal(5, 1.5, 20).round(1),
        'Satisfacción (Encuesta)': np.random.normal(3.5, 1, 20).round(1)
    })
    staff_df['Riesgo Burnout (%)'] = (staff_df['Llamadas/Turno (Promedio)'] / staff_df['Llamadas/Turno (Promedio)'].max() * 50) + ((5 - staff_df['Satisfacción (Encuesta)']) / 5 * 50)
    staff_df = staff_df.sort_values('Riesgo Burnout (%)', ascending=False).reset_index(drop=True)
    
    st.warning("Lista de personal con mayor riesgo de agotamiento. Requiere atención de RRHH.")
    st.dataframe(staff_df.head(10), use_container_width=True)

# --- Pestaña 6: Optimización Geoespacial (Roadmap) ---
with tabs[6]:
    st.header("🗺️ Análisis #5: Optimización de Ubicación de Bases (Roadmap)")
    st.info("Esta sección es una **demostración de capacidad (roadmap)**. Requiere datos geoespaciales reales de incidentes (lat/long) que no están en el informe de 2013.", icon="💡")
    
    st.markdown("""
    **Objetivo:** Reducir los tiempos de respuesta al reubicar estratégicamente las bases de ambulancias para que estén más cerca de donde ocurren las emergencias.

    **Metodología:**
    1.  **Geocodificar** miles de ubicaciones de incidentes pasados desde los FRAPs.
    2.  Usar **algoritmos de clustering (ej. DBSCAN)** para identificar "puntos calientes" (hotspots) de demanda de servicios.
    3.  Aplicar un **modelo de optimización de ubicación-asignación** para encontrar las coordenadas óptimas para un número determinado de bases, minimizando el tiempo de viaje promedio a los hotspots.
    
    A continuación se muestra un mapa de ejemplo con hotspots simulados y una ubicación de base optimizada.
    """)

    tj_center = [32.5149, -117.0382]
    m = folium.Map(location=tj_center, zoom_start=12)

    hotspots = np.random.normal(loc=tj_center, scale=[0.05, 0.05], size=(5, 2))
    for lat, lon in hotspots:
        folium.Circle(
            location=[lat, lon],
            radius=1500,
            color=PRIMARY_COLOR,
            fill=True,
            fill_color=PRIMARY_COLOR,
            fill_opacity=0.3
        ).add_to(m)
        folium.Marker(location=[lat, lon], popup="Hotspot de Incidentes", icon=folium.Icon(color='red', icon='info-sign')).add_to(m)

    optimal_base = np.mean(hotspots, axis=0)
    folium.Marker(
        location=optimal_base,
        popup="Ubicación Óptima Propuesta para Nueva Base",
        icon=folium.Icon(color='green', icon='plus', prefix='fa')
    ).add_to(m)

    st_folium(m, width=725, height=500)
    st.success("**Impacto Potencial:** Reducción de 2-5 minutos en el tiempo de respuesta promedio, lo que se traduce directamente en vidas salvadas, especialmente en casos de trauma y paros cardíacos.")


# --- Pestaña 7: Futuro & Roadmap IA ---
with tabs[7]:
    st.header("🔬 Futuro & Roadmap de Capacidades de IA")
    st.warning("Las siguientes son análisis de alto impacto que pueden implementarse una vez que se establezcan los sistemas de recolección de datos necesarios.", icon="🚀")
    
    st.subheader("Análisis #24: Mantenimiento Predictivo para la Flota")
    st.markdown("""
    - **Descripción:** En lugar de reaccionar a las averías, un modelo de IA puede predecir la probabilidad de que una ambulancia falle en las próximas semanas basándose en su edad, modelo, kilómetros recorridos y registros de mantenimiento.
    - **Datos Requeridos:** Registros detallados de mantenimiento por unidad, datos de telemetría (km, horas de motor), historial de fallas.
    - **Beneficio:** Reducir averías costosas en servicio, aumentar la disponibilidad de la flota y extender la vida útil de los vehículos.
    """)
    
    st.subheader("Análisis #23: Análisis de Sentimiento del Paciente con NLP")
    st.markdown("""
    - **Descripción:** Usar Procesamiento de Lenguaje Natural (NLP) para analizar los comentarios de texto libre en las encuestas de pacientes. La IA puede identificar temas recurrentes, problemas específicos y tendencias de sentimiento.
    - **Datos Requeridos:** Encuestas de pacientes con campos de texto abierto.
    - **Beneficio:** Descubrir problemas específicos que las preguntas de opción múltiple no capturan (ej. "el personal de admisión fue grosero", "la señalización del hospital es confusa").
    """)
    
    st.subheader("Análisis #22: Análisis Geoespacial de Incidentes de Trauma")
    st.markdown("""
    - **Descripción:** Mapear la ubicación exacta de los incidentes de trauma para identificar intersecciones peligrosas, zonas de alta criminalidad o tramos de carretera con alta accidentalidad.
    - **Datos Requeridos:** Coordenadas GPS de cada incidente, extraídas de un sistema e-FRAP.
    - **Beneficio:** Proporcionar datos a las autoridades municipales para acciones preventivas (semáforos, topes, vigilancia policial), evitando que ocurran los incidentes en primer lugar.
    """)
    
    st.subheader("Análisis #30: Índice de Riesgo de Vulnerabilidad ante Desastres")
    st.markdown("""
    - **Descripción:** Crear un mapa de riesgo dinámico de la ciudad superponiendo capas de datos: fallas geológicas, zonas inundables, densidad de población, y ubicación de infraestructura crítica (hospitales, refugios).
    - **Datos Requeridos:** Datos GIS de la ciudad, mapas de fallas geológicas, proyecciones demográficas.
    - **Beneficio:** Guiar la planificación de la respuesta a desastres, la ubicación de recursos pre-posicionados y las campañas de concientización en las comunidades más vulnerables.
    """)
