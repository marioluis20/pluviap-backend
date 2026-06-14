from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from datetime import datetime
from typing import Dict, Any, Optional
from pathlib import Path

import pandas as pd
import numpy as np
import joblib


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
    datos = {
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

    return predecir_v21(datos)