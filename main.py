from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
from pathlib import Path

import pandas as pd
import numpy as np
import joblib
import requests
import re


# ============================================================
# RUTAS SEGURAS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "modelos"

print("Buscando modelos en:", MODEL_DIR)


# ============================================================
# CARGA DE MODELOS
# ============================================================

modelo_v1 = joblib.load(MODEL_DIR / "modelo_xgboost_inundacion_mariato.pkl")
config_v1 = joblib.load(MODEL_DIR / "configuracion_modelo_mariato.pkl")

modelo_rf_v2 = joblib.load(MODEL_DIR / "modelo_random_forest_mariato_v2.pkl")
modelo_lgbm_v2 = joblib.load(MODEL_DIR / "modelo_lightgbm_mariato_v2.pkl")
config_v2 = joblib.load(MODEL_DIR / "configuracion_modelo_mariato_v2.pkl")
config_v21 = joblib.load(MODEL_DIR / "configuracion_modelo_mariato_v21.pkl")

features_v1 = config_v21["features_v1"]
features_v2 = config_v21["features_v2"]

umbral_v1 = config_v21["umbral_v1"]
umbral_v2 = config_v21["umbral_v2"]

peso_rf = config_v21["peso_rf_v2"]
peso_lgbm = config_v21["peso_lgbm_v2"]

percentiles_v2 = config_v21["percentiles_v2"]

print("Modelos PLUVIAP V2.1 cargados correctamente.")


# ============================================================
# FASTAPI
# ============================================================

app = FastAPI(
    title="PLUVIAP API Predictiva",
    description="API para predicción de riesgo de inundaciones en Mariato usando modelo V2.1",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# MODELOS DE ENTRADA
# ============================================================

class PredictionRequest(BaseModel):
    features: Dict[str, Any]


class SimplePredictionRequest(BaseModel):
    lluvia_1d: Optional[float] = 12.4
    lluvia_3d: Optional[float] = 34.8
    lluvia_7d: Optional[float] = 68.2
    lluvia_15d: Optional[float] = 124.6
    lluvia_30d: Optional[float] = 210.5
    api_proxy_lluvia: Optional[float] = 0.31
    fase_enso: Optional[str] = "Neutral"
    anomalia_nino34: Optional[float] = 0.10
    marea_alta: Optional[int] = 0
    sistema_tropical_activo: Optional[int] = 0
    mes: Optional[int] = None

# ============================================================
# DATOS METEOROLÓGICOS ACTUALES / PRONÓSTICO
# ============================================================

LAT_MARIATO = 7.65
LON_MARIATO = -81.00
TIMEZONE_MARIATO = "America/Panama"


def sumar_precipitacion_por_rango(times, values, inicio, fin):
    """
    Suma precipitación horaria entre dos fechas.
    times: lista de datetime.
    values: lista de precipitación en mm.
    inicio / fin: datetime local.
    """
    total = 0.0

    for t, v in zip(times, values):
        if inicio <= t < fin:
            try:
                total += float(v or 0)
            except Exception:
                total += 0.0

    return total


def obtener_lluvia_open_meteo():
    """
    Consulta lluvia horaria para Mariato usando Open-Meteo.

    Para alerta temprana usamos una estrategia preventiva:
    - lluvia_1d: mayor valor entre lluvia observada últimas 24h y pronóstico próximas 24h.
    - lluvia_3d: mayor valor entre lluvia observada últimas 72h y pronóstico próximas 72h.
    - lluvia_7d: mayor valor entre lluvia reciente 7 días y pronóstico 7 días.

    Esto permite que PLUVIAP reaccione tanto a lluvia reciente como a lluvia esperada.
    """

    url = "https://api.open-meteo.com/v1/forecast"

    params = {
        "latitude": LAT_MARIATO,
        "longitude": LON_MARIATO,
        "hourly": "precipitation",
        "past_days": 7,
        "forecast_days": 7,
        "timezone": TIMEZONE_MARIATO
    }

    response = requests.get(url, params=params, timeout=20)
    response.raise_for_status()

    data = response.json()

    hourly = data.get("hourly", {})
    time_raw = hourly.get("time", [])
    precipitation_raw = hourly.get("precipitation", [])

    if not time_raw or not precipitation_raw:
        raise ValueError("Open-Meteo no devolvió datos horarios de precipitación.")

    times = [datetime.fromisoformat(t) for t in time_raw]
    values = [float(v or 0) for v in precipitation_raw]

    ahora = datetime.now()

    lluvia_obs_24h = sumar_precipitacion_por_rango(
        times,
        values,
        ahora - timedelta(hours=24),
        ahora
    )

    lluvia_pred_24h = sumar_precipitacion_por_rango(
        times,
        values,
        ahora,
        ahora + timedelta(hours=24)
    )

    lluvia_obs_72h = sumar_precipitacion_por_rango(
        times,
        values,
        ahora - timedelta(hours=72),
        ahora
    )

    lluvia_pred_72h = sumar_precipitacion_por_rango(
        times,
        values,
        ahora,
        ahora + timedelta(hours=72)
    )

    lluvia_obs_7d = sumar_precipitacion_por_rango(
        times,
        values,
        ahora - timedelta(days=7),
        ahora
    )

    lluvia_pred_7d = sumar_precipitacion_por_rango(
        times,
        values,
        ahora,
        ahora + timedelta(days=7)
    )

    lluvia_1d = max(lluvia_obs_24h, lluvia_pred_24h)
    lluvia_3d = max(lluvia_obs_72h, lluvia_pred_72h)
    lluvia_7d = max(lluvia_obs_7d, lluvia_pred_7d)

    return {
        "lluvia_1d": round(float(lluvia_1d), 2),
        "lluvia_3d": round(float(lluvia_3d), 2),
        "lluvia_7d": round(float(lluvia_7d), 2),
        "detalle_lluvia": {
            "lluvia_obs_24h": round(float(lluvia_obs_24h), 2),
            "lluvia_pred_24h": round(float(lluvia_pred_24h), 2),
            "lluvia_obs_72h": round(float(lluvia_obs_72h), 2),
            "lluvia_pred_72h": round(float(lluvia_pred_72h), 2),
            "lluvia_obs_7d": round(float(lluvia_obs_7d), 2),
            "lluvia_pred_7d": round(float(lluvia_pred_7d), 2)
        }
    }


def calcular_api_proxy_lluvia(lluvia_1d, lluvia_3d, lluvia_7d):
    """
    Calcula una saturación aproximada usando acumulados de lluvia.

    Es una variable proxy:
    - No mide humedad real del suelo.
    - Sirve como aproximación operativa para el modelo.
    """

    api_bruto = (
        0.50 * lluvia_1d +
        0.30 * lluvia_3d +
        0.20 * lluvia_7d
    )

    api_normalizado = min(api_bruto / 300.0, 1.0)

    return round(float(api_normalizado), 2)


def obtener_datos_actuales_mariato():
    """
    Construye las variables actuales que necesita el modelo V2.1
    usando fuentes gratuitas disponibles.
    """

    lluvia = obtener_lluvia_open_meteo()
    enso = obtener_enso_noaa()
    marea = obtener_marea_open_meteo()
    ciclon = obtener_sistema_tropical_nhc()

    lluvia_1d = lluvia["lluvia_1d"]
    lluvia_3d = lluvia["lluvia_3d"]
    lluvia_7d = lluvia["lluvia_7d"]

    api_proxy = calcular_api_proxy_lluvia(
        lluvia_1d=lluvia_1d,
        lluvia_3d=lluvia_3d,
        lluvia_7d=lluvia_7d
    )

    datos = {
        "lluvia_1d": lluvia_1d,
        "lluvia_3d": lluvia_3d,
        "lluvia_7d": lluvia_7d,

        # Aún son aproximaciones hasta conectar histórico diario 15/30 días.
        "lluvia_15d": lluvia_7d,
        "lluvia_30d": lluvia_7d,

        "api_proxy_lluvia": api_proxy,

        "fase_enso": enso["fase_enso"],
        "anomalia_nino34": enso["anomalia_nino34"],

        "marea_alta": marea["marea_alta"],
        "sistema_tropical_activo": ciclon["sistema_tropical_activo"],

        "mes": datetime.now().month
    }

    datos["_fuentes"] = {
        "lluvia": "Open-Meteo",
        "enso": enso.get("fuente_enso"),
        "marea": marea.get("fuente_marea"),
        "ciclones": ciclon.get("fuente_ciclones"),
        "marea_estado": marea.get("marea_estado"),
        "nivel_mar_proxy": marea.get("nivel_mar_proxy"),
        "nombre_sistema_tropical": ciclon.get("nombre_sistema_tropical"),
        "detalle_ciclones": ciclon.get("detalle_ciclones")
    }

    return datos

# ============================================================
# ENSO / NIÑO 3.4 - NOAA CPC
# ============================================================

def obtener_enso_noaa():
    """
    Obtiene la anomalía Niño 3.4 desde NOAA CPC.
    Si falla, devuelve valor neutral de respaldo.
    """

    url = "https://www.cpc.ncep.noaa.gov/data/indices/wksst9120.for"

    try:
        response = requests.get(url, timeout=20)
        response.raise_for_status()

        lineas = response.text.strip().splitlines()

        # Buscar últimas líneas con datos numéricos
        lineas_datos = []
        for linea in lineas:
            numeros = re.findall(r"[-+]?\d+\.\d+", linea)
            if len(numeros) >= 8:
                lineas_datos.append(linea)

        if not lineas_datos:
            raise ValueError("No se encontraron líneas válidas en NOAA CPC.")

        ultima = lineas_datos[-1]
        numeros = [float(x) for x in re.findall(r"[-+]?\d+\.\d+", ultima)]

        # Formato típico:
        # Nino1+2 SST/SSTA, Nino3 SST/SSTA, Nino3.4 SST/SSTA, Nino4 SST/SSTA
        anomalia_nino34 = numeros[5]

        if anomalia_nino34 >= 0.5:
            fase = "El Niño"
        elif anomalia_nino34 <= -0.5:
            fase = "La Niña"
        else:
            fase = "Neutral"

        return {
            "fase_enso": fase,
            "anomalia_nino34": round(float(anomalia_nino34), 2),
            "fuente_enso": "NOAA CPC"
        }

    except Exception as e:
        print("Error obteniendo ENSO NOAA:", e)

        return {
            "fase_enso": "Neutral",
            "anomalia_nino34": 0.10,
            "fuente_enso": "respaldo_local"
        }


# ============================================================
# MAREA PROXY - OPEN-METEO MARINE
# ============================================================

def obtener_marea_open_meteo():
    """
    Usa Open-Meteo Marine como proxy gratuito de marea.

    Importante:
    - sea_level_height_msl considera variaciones del nivel del mar incluyendo mareas.
    - No debe usarse como dato oficial de navegación.
    - Para PLUVIAP se usa solo como señal aproximada de marea alta.
    """

    url = "https://marine-api.open-meteo.com/v1/marine"

    params = {
        "latitude": LAT_MARIATO,
        "longitude": LON_MARIATO,
        "hourly": "sea_level_height_msl",
        "past_days": 2,
        "forecast_days": 2,
        "timezone": TIMEZONE_MARIATO
    }

    try:
        response = requests.get(url, params=params, timeout=20)
        response.raise_for_status()

        data = response.json()
        hourly = data.get("hourly", {})

        tiempos_raw = hourly.get("time", [])
        niveles_raw = hourly.get("sea_level_height_msl", [])

        if not tiempos_raw or not niveles_raw:
            raise ValueError("Open-Meteo Marine no devolvió nivel del mar.")

        tiempos = [datetime.fromisoformat(t) for t in tiempos_raw]
        niveles = [float(v or 0) for v in niveles_raw]

        ahora = datetime.now()

        # Buscar valor más cercano a la hora actual
        indice_actual = min(
            range(len(tiempos)),
            key=lambda i: abs((tiempos[i] - ahora).total_seconds())
        )

        nivel_actual = niveles[indice_actual]

        # Umbral proxy: si el nivel actual está sobre el percentil 75 de la ventana,
        # se considera marea alta operacional.
        p75 = float(np.percentile(niveles, 75))
        p90 = float(np.percentile(niveles, 90))

        if nivel_actual >= p90:
            estado = "Alta"
            marea_alta = 1
        elif nivel_actual >= p75:
            estado = "Elevada"
            marea_alta = 1
        else:
            estado = "Normal"
            marea_alta = 0

        return {
            "marea_alta": int(marea_alta),
            "marea_estado": estado,
            "nivel_mar_proxy": round(float(nivel_actual), 3),
            "umbral_marea_p75": round(float(p75), 3),
            "umbral_marea_p90": round(float(p90), 3),
            "fuente_marea": "Open-Meteo Marine"
        }

    except Exception as e:
        print("Error obteniendo marea Open-Meteo Marine:", e)

        return {
            "marea_alta": 0,
            "marea_estado": "Normal",
            "nivel_mar_proxy": None,
            "umbral_marea_p75": None,
            "umbral_marea_p90": None,
            "fuente_marea": "respaldo_local"
        }


# ============================================================
# SISTEMAS TROPICALES - NOAA NHC
# ============================================================

def convertir_coordenada_tropical(valor):
    """
    Convierte coordenadas tipo '13.5N' o '72.4W' a float.
    """

    if valor is None:
        return None

    if isinstance(valor, (int, float)):
        return float(valor)

    texto = str(valor).strip().upper()

    match = re.search(r"([-+]?\d+\.?\d*)", texto)
    if not match:
        return None

    numero = float(match.group(1))

    if "S" in texto or "W" in texto:
        numero = -abs(numero)

    return numero


def buscar_valor_recursivo(obj, posibles_claves):
    """
    Busca una clave dentro de estructuras JSON anidadas.
    """

    if isinstance(obj, dict):
        for clave, valor in obj.items():
            if clave.lower() in posibles_claves:
                return valor

        for valor in obj.values():
            encontrado = buscar_valor_recursivo(valor, posibles_claves)
            if encontrado is not None:
                return encontrado

    elif isinstance(obj, list):
        for item in obj:
            encontrado = buscar_valor_recursivo(item, posibles_claves)
            if encontrado is not None:
                return encontrado

    return None


def obtener_sistema_tropical_nhc():
    """
    Consulta ciclones activos del National Hurricane Center.

    Estrategia:
    - Si hay tormentas activas cerca del Caribe, Centroamérica o Pacífico oriental,
      se marca sistema_tropical_activo = 1.
    - Si no hay tormentas cercanas, queda 0.
    """

    url = "https://www.nhc.noaa.gov/CurrentStorms.json"

    try:
        response = requests.get(url, timeout=20)
        response.raise_for_status()

        data = response.json()

        # Intentar obtener lista de tormentas activas
        tormentas = []

        if isinstance(data, dict):
            for clave in ["activeStorms", "storms", "CurrentStorms"]:
                if clave in data and isinstance(data[clave], list):
                    tormentas = data[clave]
                    break

            if not tormentas:
                # Si la estructura cambia, buscar listas dentro del JSON
                for valor in data.values():
                    if isinstance(valor, list):
                        tormentas = valor
                        break

        if not tormentas:
            return {
                "sistema_tropical_activo": 0,
                "nombre_sistema_tropical": None,
                "fuente_ciclones": "NOAA NHC",
                "detalle_ciclones": "sin_tormentas_activas"
            }

        sistemas_relevantes = []

        for tormenta in tormentas:
            lat_raw = buscar_valor_recursivo(
                tormenta,
                {"lat", "latitude", "centerlat", "center_lat"}
            )

            lon_raw = buscar_valor_recursivo(
                tormenta,
                {"lon", "lng", "longitude", "centerlon", "center_lon"}
            )

            lat = convertir_coordenada_tropical(lat_raw)
            lon = convertir_coordenada_tropical(lon_raw)

            nombre = buscar_valor_recursivo(
                tormenta,
                {"name", "stormname", "storm_name", "systemname"}
            )

            # Zona amplia de interés para Panamá, Caribe y Pacífico oriental cercano
            if lat is not None and lon is not None:
                cerca_panama = (
                    0 <= lat <= 25 and
                    -110 <= lon <= -55
                )

                if cerca_panama:
                    sistemas_relevantes.append({
                        "nombre": nombre,
                        "lat": lat,
                        "lon": lon
                    })

        if sistemas_relevantes:
            return {
                "sistema_tropical_activo": 1,
                "nombre_sistema_tropical": sistemas_relevantes[0]["nombre"],
                "fuente_ciclones": "NOAA NHC",
                "detalle_ciclones": sistemas_relevantes
            }

        return {
            "sistema_tropical_activo": 0,
            "nombre_sistema_tropical": None,
            "fuente_ciclones": "NOAA NHC",
            "detalle_ciclones": "sin_sistemas_cercanos"
        }

    except Exception as e:
        print("Error obteniendo sistemas tropicales NHC:", e)

        return {
            "sistema_tropical_activo": 0,
            "nombre_sistema_tropical": None,
            "fuente_ciclones": "respaldo_local",
            "detalle_ciclones": "error_consulta"
        }

# ============================================================
# FUNCIONES AUXILIARES
# ============================================================

def obtener_valor(fila, posibles_columnas, default=0):
    for col in posibles_columnas:
        if col in fila.index:
            valor = fila[col]
            if pd.isna(valor):
                return default
            return valor
    return default


def clasificar_alerta_v1_centinela(prob_v1, umbral_v1):
    if prob_v1 >= 0.80:
        return "amarilla_tecnica"
    elif prob_v1 >= max(umbral_v1, 0.35):
        return "verde_tecnica"
    elif prob_v1 >= umbral_v1:
        return "monitoreo_silencioso"
    else:
        return "normal"


def juicio_operacional_v2(fila, prob_modelo, percentiles, umbral_modelo):
    razones = []
    score_fisico = 0
    prob_ajustada = float(prob_modelo)

    lluvia_1d = obtener_valor(
        fila,
        ["lluvia_1d", "lluvia_dia", "lluvia_oficial_dia"],
        0
    )

    lluvia_3d = obtener_valor(
        fila,
        ["lluvia_3d"],
        0
    )

    lluvia_7d = obtener_valor(
        fila,
        ["lluvia_7d"],
        0
    )

    api_proxy = obtener_valor(
        fila,
        ["api_proxy_lluvia"],
        0
    )

    ciclon = obtener_valor(
        fila,
        ["ciclon_activo", "ciclon_documental_activo", "sistema_tropical_activo"],
        0
    )

    marea_alta = obtener_valor(
        fila,
        ["pleamar_binaria", "marea_alta", "marea_alta_binaria"],
        0
    )

    # Lluvia 1 día
    if "lluvia_1d" in percentiles:
        if lluvia_1d >= percentiles["lluvia_1d"]["p99"]:
            score_fisico += 4
            prob_ajustada = max(prob_ajustada, 0.88)
            razones.append("lluvia_1d_extrema_p99")

        elif lluvia_1d >= percentiles["lluvia_1d"]["p98"]:
            score_fisico += 3
            prob_ajustada = max(prob_ajustada, 0.78)
            razones.append("lluvia_1d_extrema_p98")

        elif lluvia_1d >= percentiles["lluvia_1d"]["p95"]:
            score_fisico += 2
            prob_ajustada = max(prob_ajustada, 0.55)
            razones.append("lluvia_1d_muy_alta_p95")

    # Lluvia 3 días
    if "lluvia_3d" in percentiles:
        if lluvia_3d >= percentiles["lluvia_3d"]["p98"]:
            score_fisico += 3
            prob_ajustada = max(prob_ajustada, 0.75)
            razones.append("lluvia_3d_extrema_p98")

        elif lluvia_3d >= percentiles["lluvia_3d"]["p95"]:
            score_fisico += 2
            prob_ajustada = max(prob_ajustada, 0.52)
            razones.append("lluvia_3d_alta_p95")

    # Lluvia 7 días / saturación
    if "lluvia_7d" in percentiles:
        if lluvia_7d >= percentiles["lluvia_7d"]["p98"]:
            score_fisico += 3
            prob_ajustada = max(prob_ajustada, 0.72)
            razones.append("lluvia_7d_extrema_saturacion")

        elif lluvia_7d >= percentiles["lluvia_7d"]["p95"]:
            score_fisico += 2
            prob_ajustada = max(prob_ajustada, 0.50)
            razones.append("lluvia_7d_alta_saturacion")

    # API proxy
    if "api_proxy_lluvia" in percentiles:
        if api_proxy >= percentiles["api_proxy_lluvia"]["p98"]:
            score_fisico += 2
            prob_ajustada = max(prob_ajustada, 0.65)
            razones.append("api_proxy_extremo")

        elif api_proxy >= percentiles["api_proxy_lluvia"]["p95"]:
            score_fisico += 1
            prob_ajustada = max(prob_ajustada, 0.48)
            razones.append("api_proxy_alto")

    # Sistema tropical
    if ciclon == 1:
        score_fisico += 2
        prob_ajustada = max(prob_ajustada, 0.58)
        razones.append("sistema_tropical_activo")

    # Marea alta / efecto tapón
    if marea_alta == 1:
        razones.append("marea_alta_pleamar")

        if score_fisico >= 3 or prob_modelo >= 0.35:
            score_fisico += 2
            prob_ajustada = max(prob_ajustada, 0.70)
            razones.append("efecto_tapon_lluvia_marea")
        else:
            score_fisico += 1
            prob_ajustada = max(prob_ajustada, 0.40)

    # Clasificación final V2
    if prob_ajustada >= 0.82 or score_fisico >= 7:
        alerta = "roja"
    elif prob_ajustada >= 0.58 or score_fisico >= 5:
        alerta = "amarilla"
    elif prob_ajustada >= umbral_modelo or score_fisico >= 3:
        alerta = "verde"
    else:
        alerta = "normal"

    return prob_ajustada, alerta, razones, score_fisico


def fusion_operativa_v21(alerta_v2, prob_v2, score_fisico, prob_v1, centinela_v1):
    if alerta_v2 == "roja":
        return "roja", ["v2_alerta_roja_confirmada"]

    if alerta_v2 == "amarilla":
        return "amarilla", ["v2_alerta_amarilla_confirmada"]

    if alerta_v2 == "verde":
        return "verde", ["v2_alerta_verde_confirmada"]

    if alerta_v2 == "normal":

        if centinela_v1 == "amarilla_tecnica":
            if score_fisico >= 1 or prob_v2 >= 0.10:
                return "amarilla", [
                    "centinela_v1_alto",
                    "confirmacion_minima_v2_o_fisica"
                ]
            else:
                return "verde", [
                    "centinela_v1_alto",
                    "monitoreo_preventivo_sin_confirmacion_fisica"
                ]

        if centinela_v1 == "verde_tecnica":
            return "verde", [
                "centinela_v1_detecta_riesgo",
                "prevencion_anti_falso_negativo"
            ]

        if centinela_v1 == "monitoreo_silencioso":
            return "verde", [
                "centinela_v1_umbral_bajo",
                "monitoreo_tecnico"
            ]

    return "normal", ["sin_senales_relevantes"]


def estado_titulo(alerta):
    if alerta == "normal":
        return "NORMAL"
    if alerta == "verde":
        return "MONITOREO"
    if alerta == "amarilla":
        return "PREPARACIÓN"
    if alerta == "roja":
        return "ALERTA CRÍTICA"
    return "SIN CLASIFICAR"


def riesgo_texto(alerta):
    if alerta == "normal":
        return "Bajo"
    if alerta == "verde":
        return "Monitoreo"
    if alerta == "amarilla":
        return "Moderado"
    if alerta == "roja":
        return "Alto"
    return "Desconocido"


def generar_mensaje(alerta):
    if alerta == "normal":
        return "No hay riesgo de inundación en su área."
    if alerta == "verde":
        return "Monitoreo preventivo activo. Manténgase informado."
    if alerta == "amarilla":
        return "Preparación recomendada. Evite cruzar ríos o quebradas."
    if alerta == "roja":
        return "Alerta crítica. Siga instrucciones de las autoridades."
    return "Estado no disponible."


def tendencia_por_probabilidad(prob):
    if prob < 0.30:
        return "Estable"
    if prob < 0.55:
        return "En aumento"
    if prob < 0.80:
        return "Elevada"
    return "Crítica"


def crear_dataframe_features(features_requeridas, datos_usuario):
    fila = {}

    for feature in features_requeridas:
        fila[feature] = datos_usuario.get(feature, np.nan)

    return pd.DataFrame([fila])


# ============================================================
# FUNCIÓN PRINCIPAL DE PREDICCIÓN V2.1
# ============================================================

def predecir_v21(datos_usuario: Dict[str, Any]):
    X_v1 = crear_dataframe_features(features_v1, datos_usuario)
    X_v2 = crear_dataframe_features(features_v2, datos_usuario)

    prob_v1 = modelo_v1.predict_proba(X_v1)[:, 1][0]

    prob_rf = modelo_rf_v2.predict_proba(X_v2)[:, 1][0]
    prob_lgbm = modelo_lgbm_v2.predict_proba(X_v2)[:, 1][0]

    prob_v2 = (peso_rf * prob_rf) + (peso_lgbm * prob_lgbm)

    fila_v2 = X_v2.iloc[0]

    prob_operativa_v2, alerta_v2, razones_v2, score_fisico = juicio_operacional_v2(
        fila=fila_v2,
        prob_modelo=prob_v2,
        percentiles=percentiles_v2,
        umbral_modelo=umbral_v2
    )

    centinela_v1 = clasificar_alerta_v1_centinela(prob_v1, umbral_v1)

    alerta_final, razones_finales = fusion_operativa_v21(
        alerta_v2=alerta_v2,
        prob_v2=prob_v2,
        score_fisico=score_fisico,
        prob_v1=prob_v1,
        centinela_v1=centinela_v1
    )

    lluvia_1d = float(datos_usuario.get("lluvia_1d", 0) or 0)
    lluvia_3d = float(datos_usuario.get("lluvia_3d", 0) or 0)
    lluvia_7d = float(datos_usuario.get("lluvia_7d", 0) or 0)
    lluvia_15d = float(datos_usuario.get("lluvia_15d", 0) or 0)
    lluvia_30d = float(datos_usuario.get("lluvia_30d", 0) or 0)
    api_proxy = float(datos_usuario.get("api_proxy_lluvia", 0) or 0)

    fase_enso = str(datos_usuario.get("fase_enso", "Neutral"))
    anomalia_nino34 = float(datos_usuario.get("anomalia_nino34", 0) or 0)

    marea_alta = int(datos_usuario.get("marea_alta", 0) or 0)
    sistema_tropical = int(datos_usuario.get("sistema_tropical_activo", 0) or 0)

    return {
        "version": "V2.1",

        "alertaFinal": alerta_final,
        "alertaV2": alerta_v2,
        "estadoTitulo": estado_titulo(alerta_final),
        "mensaje": generar_mensaje(alerta_final),

        "probabilidadV1Centinela": round(float(prob_v1), 4),
        "centinelaV1": centinela_v1,

        "probabilidadRfV2": round(float(prob_rf), 4),
        "probabilidadLgbmV2": round(float(prob_lgbm), 4),
        "probabilidadEnsembleV2": round(float(prob_v2), 4),
        "probabilidadOperativa": round(float(prob_operativa_v2), 4),

        "scoreFisico": int(score_fisico),

        "lluvia1d": lluvia_1d,
        "lluvia3d": lluvia_3d,
        "lluvia7d": lluvia_7d,
        "lluvia15d": lluvia_15d,
        "lluvia30d": lluvia_30d,
        "apiProxy": api_proxy,

        "faseEnso": fase_enso,
        "anomaliaNino34": anomalia_nino34,

        "mareaEstado": "Alta" if marea_alta == 1 else "Normal",
        "efectoTapon": True if marea_alta == 1 and score_fisico >= 3 else False,
        "sistemaTropicalActivo": True if sistema_tropical == 1 else False,

        "tendencia": tendencia_por_probabilidad(prob_operativa_v2),
        "riesgoTexto": riesgo_texto(alerta_final),
        "ubicacion": "Mariato, Veraguas, Panamá",
        "zonasRiesgo": 2,

        "razonesV2": razones_v2,
        "razonesFinales": razones_finales,

        "fechaActualizacion": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

def probar_fuente_segura(nombre, funcion):
    try:
        resultado = funcion()
        return {
            "ok": True,
            "fuente": nombre,
            "resultado": resultado
        }
    except Exception as e:
        return {
            "ok": False,
            "fuente": nombre,
            "errorTipo": type(e).__name__,
            "errorMensaje": str(e)
        }

# ============================================================
# ENDPOINTS
# ============================================================

@app.get("/")
def root():
    return {
        "message": "PLUVIAP API funcionando",
        "version": "V2.1",
        "status": "ok"
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "modelos": "cargados",
        "version": "V2.1",
        "ruta_modelos": str(MODEL_DIR)
    }


@app.post("/predict")
def predict(request: PredictionRequest):
    return predecir_v21(request.features)


@app.post("/predict-simple")
def predict_simple(request: SimplePredictionRequest):
    datos = {
        "lluvia_1d": request.lluvia_1d,
        "lluvia_3d": request.lluvia_3d,
        "lluvia_7d": request.lluvia_7d,
        "lluvia_15d": request.lluvia_15d,
        "lluvia_30d": request.lluvia_30d,
        "api_proxy_lluvia": request.api_proxy_lluvia,
        "fase_enso": request.fase_enso,
        "anomalia_nino34": request.anomalia_nino34,
        "marea_alta": request.marea_alta,
        "sistema_tropical_activo": request.sistema_tropical_activo,
        "mes": request.mes if request.mes is not None else datetime.now().month
    }

    return predecir_v21(datos)


@app.get("/current-prediction")
def current_prediction():
    try:
        datos = obtener_datos_actuales_mariato()
        respuesta = predecir_v21(datos)

        fuentes = datos.get("_fuentes", {})

        respuesta["fuenteDatos"] = "Open-Meteo + NOAA CPC + Open-Meteo Marine + NOAA NHC"
        respuesta["tipoDatos"] = "datos_hidrometeorologicos_actuales"

        respuesta["fuenteLluvia"] = fuentes.get("lluvia", "Open-Meteo")
        respuesta["fuenteEnso"] = fuentes.get("enso")
        respuesta["fuenteMarea"] = fuentes.get("marea")
        respuesta["fuenteCiclones"] = fuentes.get("ciclones")

        respuesta["mareaEstado"] = fuentes.get(
            "marea_estado",
            respuesta.get("mareaEstado", "Normal")
        )
        respuesta["nivelMarProxy"] = fuentes.get("nivel_mar_proxy")
        respuesta["nombreSistemaTropical"] = fuentes.get("nombre_sistema_tropical")
        respuesta["detalleCiclones"] = fuentes.get("detalle_ciclones")

        respuesta["fechaActualizacion"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        return respuesta

    except Exception as e:
        print("Error obteniendo datos actuales:", e)

        datos_respaldo = {
            "lluvia_1d": 12.4,
            "lluvia_3d": 34.8,
            "lluvia_7d": 68.2,
            "lluvia_15d": 124.6,
            "lluvia_30d": 210.5,
            "api_proxy_lluvia": 0.31,
            "fase_enso": "Neutral",
            "anomalia_nino34": 0.10,
            "marea_alta": 0,
            "sistema_tropical_activo": 0,
            "mes": datetime.now().month
        }

        respuesta = predecir_v21(datos_respaldo)
        respuesta["fuenteDatos"] = "respaldo_local"
        respuesta["tipoDatos"] = "fallback"
        respuesta["mensaje"] = "No se pudieron consultar datos actuales. Mostrando predicción de respaldo."
        respuesta["fechaActualizacion"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        respuesta["errorDatosActuales"] = f"{type(e).__name__}: {str(e)}"

        return respuesta


@app.get("/weather-debug")
def weather_debug():
    resultado = probar_fuente_segura(
        "Open-Meteo lluvia",
        obtener_lluvia_open_meteo
    )

    if not resultado["ok"]:
        return {
            "ubicacion": "Mariato, Veraguas, Panamá",
            "estado": "error",
            "fuente": "Open-Meteo",
            "errorTipo": resultado["errorTipo"],
            "errorMensaje": resultado["errorMensaje"],
            "fechaConsulta": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

    lluvia = resultado["resultado"]

    lluvia_1d = lluvia["lluvia_1d"]
    lluvia_3d = lluvia["lluvia_3d"]
    lluvia_7d = lluvia["lluvia_7d"]

    api_proxy = calcular_api_proxy_lluvia(
        lluvia_1d=lluvia_1d,
        lluvia_3d=lluvia_3d,
        lluvia_7d=lluvia_7d
    )

    return {
        "ubicacion": "Mariato, Veraguas, Panamá",
        "estado": "ok",
        "latitud": LAT_MARIATO,
        "longitud": LON_MARIATO,
        "fuente": "Open-Meteo",
        "lluvia_1d": lluvia_1d,
        "lluvia_3d": lluvia_3d,
        "lluvia_7d": lluvia_7d,
        "api_proxy_lluvia": api_proxy,
        "detalle": lluvia["detalle_lluvia"],
        "fechaConsulta": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }


@app.get("/sources-debug")
def sources_debug():
    lluvia = probar_fuente_segura(
        "Open-Meteo lluvia",
        obtener_lluvia_open_meteo
    )

    enso = probar_fuente_segura(
        "NOAA CPC ENSO",
        obtener_enso_noaa
    )

    marea = probar_fuente_segura(
        "Open-Meteo Marine",
        obtener_marea_open_meteo
    )

    ciclon = probar_fuente_segura(
        "NOAA NHC ciclones",
        obtener_sistema_tropical_nhc
    )

    return {
        "ubicacion": "Mariato, Veraguas, Panamá",
        "fechaConsulta": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "lluvia_open_meteo": lluvia,
        "enso_noaa": enso,
        "marea_open_meteo_marine": marea,
        "ciclones_noaa_nhc": ciclon
    }


@app.get("/routes-debug")
def routes_debug():
    return {
        "routes": [
            route.path
            for route in app.routes
        ]
    }