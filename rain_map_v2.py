from fastapi import APIRouter
from datetime import datetime, timedelta, timezone
import os
import requests


router = APIRouter()


# ============================================================
# CONFIGURACIÓN GENERAL DEL MAPA
# ============================================================

TIMEZONE_MARIATO = "America/Panama"

OPEN_METEO_API_KEY = os.getenv("OPEN_METEO_API_KEY")

OPEN_METEO_FORECAST_URL = (
    "https://customer-api.open-meteo.com/v1/forecast"
    if OPEN_METEO_API_KEY
    else "https://api.open-meteo.com/v1/forecast"
)

MET_NORWAY_URL = "https://api.met.no/weatherapi/locationforecast/2.0/compact"
MET_NORWAY_USER_AGENT = "PLUVIAP/1.0 contacto:marioluisluismario@gmail.com"


# ============================================================
# ZONAS MONITOREADAS PARA EL MAPA
# ============================================================

RAIN_ZONES_MARIATO = [
    {"nombre": "Mariato", "lat": 7.6500, "lon": -81.0000},
    {"nombre": "Loma de Quebro", "lat": 7.5600, "lon": -80.9700},
    {"nombre": "Río Quebro", "lat": 7.5400, "lon": -80.9800},
    {"nombre": "Río Pavo", "lat": 7.4700, "lon": -80.9900},
    {"nombre": "Arenas", "lat": 7.6000, "lon": -80.9000},
    {"nombre": "El Cacao", "lat": 7.4300, "lon": -80.8800},
    {"nombre": "Palo Seco / costa", "lat": 7.6200, "lon": -81.0700},
    {"nombre": "Tebario", "lat": 7.7200, "lon": -80.9500},
]


RAIN_MAP_CACHE = {}
RAIN_MAP_CACHE_TTL_MINUTOS = 60


# ============================================================
# FUNCIONES AUXILIARES
# ============================================================

def normalizar_horizonte(horizon_hours: int = 0) -> int:
    try:
        horizon_hours = int(horizon_hours)
    except Exception:
        horizon_hours = 0

    # Para la app actual usamos presente.
    # Dejamos estos valores por si luego quieres volver a usar horizonte.
    if horizon_hours not in [0, 3, 6, 9, 12]:
        horizon_hours = 0

    return horizon_hours


def clasificar_lluvia_mm(mm: float) -> str:
    try:
        mm = float(mm or 0.0)
    except Exception:
        mm = 0.0

    if mm < 0.1:
        return "sin_lluvia"
    if mm < 1:
        return "mínima"
    if mm < 5:
        return "baja"
    if mm < 15:
        return "moderada"
    if mm < 25:
        return "fuerte"

    return "intensa"


def parsear_fecha_met_norway(fecha_texto: str):
    return datetime.fromisoformat(fecha_texto.replace("Z", "+00:00"))


def crear_respuesta_mapa(
    horizon_hours: int,
    fuente_datos: str,
    mensaje: str,
    zonas: list,
    cache: bool = False,
    cache_estado: str = "sin_cache",
    cache_edad_minutos: float = 0,
    errores_fuentes=None
):
    ahora = datetime.now()

    respuesta = {
        "fechaActualizacion": ahora.strftime("%Y-%m-%d %H:%M:%S"),
        "proximaActualizacion": (
            ahora + timedelta(minutes=RAIN_MAP_CACHE_TTL_MINUTOS)
        ).strftime("%Y-%m-%d %H:%M:%S"),
        "unidad": "mm",
        "horizonteHoras": horizon_hours,
        "fuenteDatos": fuente_datos,
        "mensaje": mensaje,
        "zonas": zonas,
        "_cache": cache,
        "_cache_estado": cache_estado,
        "_cache_edad_minutos": cache_edad_minutos
    }

    if errores_fuentes:
        respuesta["erroresFuentes"] = errores_fuentes

    return respuesta


# ============================================================
# OPEN-METEO POR COORDENADAS
# ============================================================

def obtener_rain_map_open_meteo_v2(horizon_hours: int = 0):
    """
    Consulta Open-Meteo para todas las zonas en una sola llamada.

    El valor precipitacionMm representa el acumulado de lluvia
    de una ventana de 3 horas por zona.
    """

    horizon_hours = normalizar_horizonte(horizon_hours)

    latitudes = ",".join(str(z["lat"]) for z in RAIN_ZONES_MARIATO)
    longitudes = ",".join(str(z["lon"]) for z in RAIN_ZONES_MARIATO)

    params = {
        "latitude": latitudes,
        "longitude": longitudes,
        "hourly": "precipitation",
        "forecast_days": 2,
        "timezone": TIMEZONE_MARIATO
    }

    if OPEN_METEO_API_KEY:
        params["apikey"] = OPEN_METEO_API_KEY

    response = requests.get(
        OPEN_METEO_FORECAST_URL,
        params=params,
        timeout=25
    )

    response.raise_for_status()

    data = response.json()

    respuestas = data if isinstance(data, list) else [data]

    ahora = datetime.now()
    inicio_ventana = ahora + timedelta(hours=horizon_hours)
    fin_ventana = inicio_ventana + timedelta(hours=3)

    zonas_resultado = []

    for zona, respuesta_zona in zip(RAIN_ZONES_MARIATO, respuestas):
        hourly = respuesta_zona.get("hourly", {})
        tiempos_raw = hourly.get("time", [])
        precipitaciones_raw = hourly.get("precipitation", [])

        acumulado = 0.0

        for tiempo_texto, valor in zip(tiempos_raw, precipitaciones_raw):
            try:
                tiempo = datetime.fromisoformat(tiempo_texto)

                if inicio_ventana <= tiempo < fin_ventana:
                    acumulado += float(valor or 0.0)

            except Exception:
                continue

        acumulado = round(float(acumulado), 2)

        zonas_resultado.append({
            "nombre": zona["nombre"],
            "lat": zona["lat"],
            "lon": zona["lon"],
            "precipitacionMm": acumulado,
            "nivel": clasificar_lluvia_mm(acumulado)
        })

    return crear_respuesta_mapa(
        horizon_hours=horizon_hours,
        fuente_datos="Open-Meteo",
        mensaje="Precipitación estimada por coordenadas monitoreadas.",
        zonas=zonas_resultado
    )


# ============================================================
# MET NORWAY COMO RESPALDO
# ============================================================

def obtener_rain_map_met_norway_v2(horizon_hours: int = 0):
    horizon_hours = normalizar_horizonte(horizon_hours)

    ahora_utc = datetime.now(timezone.utc)
    inicio_ventana = ahora_utc + timedelta(hours=horizon_hours)
    fin_ventana = inicio_ventana + timedelta(hours=3)

    headers = {
        "User-Agent": MET_NORWAY_USER_AGENT,
        "Accept": "application/json"
    }

    zonas_resultado = []

    for zona in RAIN_ZONES_MARIATO:
        params = {
            "lat": zona["lat"],
            "lon": zona["lon"]
        }

        response = requests.get(
            MET_NORWAY_URL,
            params=params,
            headers=headers,
            timeout=25
        )

        response.raise_for_status()

        data = response.json()

        timeseries = (
            data
            .get("properties", {})
            .get("timeseries", [])
        )

        acumulado = 0.0
        encontro_1h = False
        respaldo_6h = None

        for item in timeseries:
            try:
                tiempo = parsear_fecha_met_norway(item["time"])
                data_item = item.get("data", {})

                if inicio_ventana <= tiempo < fin_ventana:
                    next_1h = data_item.get("next_1_hours")

                    if next_1h:
                        detalles = next_1h.get("details", {})
                        acumulado += float(
                            detalles.get("precipitation_amount", 0.0) or 0.0
                        )
                        encontro_1h = True

                next_6h = data_item.get("next_6_hours")

                if next_6h and respaldo_6h is None:
                    tiempo_fin_6h = tiempo + timedelta(hours=6)

                    if tiempo <= inicio_ventana < tiempo_fin_6h:
                        detalles_6h = next_6h.get("details", {})
                        lluvia_6h = float(
                            detalles_6h.get("precipitation_amount", 0.0) or 0.0
                        )

                        # Ventana de 3h dentro de acumulado 6h.
                        respaldo_6h = lluvia_6h * 0.5

            except Exception:
                continue

        if not encontro_1h and acumulado == 0.0 and respaldo_6h is not None:
            acumulado = respaldo_6h

        acumulado = round(float(acumulado), 2)

        zonas_resultado.append({
            "nombre": zona["nombre"],
            "lat": zona["lat"],
            "lon": zona["lon"],
            "precipitacionMm": acumulado,
            "nivel": clasificar_lluvia_mm(acumulado)
        })

    return crear_respuesta_mapa(
        horizon_hours=horizon_hours,
        fuente_datos="MET Norway",
        mensaje="Precipitación estimada por zonas monitoreadas usando MET Norway.",
        zonas=zonas_resultado
    )


# ============================================================
# FALLBACK
# ============================================================

def crear_rain_map_fallback_v2(horizon_hours: int = 0):
    horizon_hours = normalizar_horizonte(horizon_hours)

    zonas = [
        {
            "nombre": z["nombre"],
            "lat": z["lat"],
            "lon": z["lon"],
            "precipitacionMm": 0.0,
            "nivel": "sin_datos"
        }
        for z in RAIN_ZONES_MARIATO
    ]

    return crear_respuesta_mapa(
        horizon_hours=horizon_hours,
        fuente_datos="fallback_temporal",
        mensaje="No se pudo consultar Open-Meteo ni MET Norway. Vista temporal sin lluvia real.",
        zonas=zonas
    )


# ============================================================
# OPERATIVO V2 CON CACHÉ
# ============================================================

def obtener_rain_map_operativo_v2(horizon_hours: int = 0):
    horizon_hours = normalizar_horizonte(horizon_hours)

    ahora = datetime.now()
    cache_key = f"rain_map_v2_h{horizon_hours}"

    if cache_key in RAIN_MAP_CACHE:
        cache = RAIN_MAP_CACHE[cache_key]
        minutos = (ahora - cache["fecha"]).total_seconds() / 60

        if minutos < RAIN_MAP_CACHE_TTL_MINUTOS:
            data_cache = cache["data"].copy()
            data_cache["_cache"] = True
            data_cache["_cache_estado"] = "cache_vigente_rain_map_v2"
            data_cache["_cache_edad_minutos"] = round(minutos, 2)
            return data_cache

    error_open = None
    error_met = None

    try:
        data = obtener_rain_map_open_meteo_v2(horizon_hours)

        data["_cache"] = False
        data["_cache_estado"] = "consulta_nueva_open_meteo_v2"
        data["_cache_edad_minutos"] = 0

        RAIN_MAP_CACHE[cache_key] = {
            "fecha": ahora,
            "data": data
        }

        return data

    except Exception as e:
        error_open = f"{type(e).__name__}: {str(e)}"

    try:
        data = obtener_rain_map_met_norway_v2(horizon_hours)

        data["_cache"] = False
        data["_cache_estado"] = "consulta_nueva_met_norway_v2"
        data["_cache_edad_minutos"] = 0
        data["erroresFuentes"] = [
            f"open_meteo_rain_map_v2: {error_open}"
        ]

        RAIN_MAP_CACHE[cache_key] = {
            "fecha": ahora,
            "data": data
        }

        return data

    except Exception as e:
        error_met = f"{type(e).__name__}: {str(e)}"

    fallback = crear_rain_map_fallback_v2(horizon_hours)

    fallback["erroresFuentes"] = [
        f"open_meteo_rain_map_v2: {error_open}",
        f"met_norway_rain_map_v2: {error_met}"
    ]

    return fallback


# ============================================================
# ENDPOINTS DE PRUEBA
# ============================================================

@router.get("/rain-map-v2")
def rain_map_v2(horizon_hours: int = 0):
    return obtener_rain_map_operativo_v2(horizon_hours)


@router.get("/rain-map-v2-debug")
def rain_map_v2_debug(horizon_hours: int = 0):
    return {
        "open_meteo_api_key_configurada": bool(OPEN_METEO_API_KEY),
        "open_meteo_forecast_url": OPEN_METEO_FORECAST_URL,
        "timezone": TIMEZONE_MARIATO,
        "zonas_configuradas": RAIN_ZONES_MARIATO,
        "resultado": obtener_rain_map_operativo_v2(horizon_hours)
    }