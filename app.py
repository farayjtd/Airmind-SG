import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import joblib
import json
import requests
import plotly.graph_objects as go
import folium
from streamlit_folium import st_folium
from datetime import datetime, timedelta
import pytz
import sqlite3
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# TIMEZONE & JAM
# ============================================================
SGT = pytz.timezone("Asia/Singapore")

def now_sgt():
    """Waktu sekarang SGT — untuk display"""
    return datetime.now(SGT)

def current_hour_sgt():
    """Jam sekarang SGT dibulatkan ke bawah — untuk prediksi"""
    n = datetime.now(SGT)
    return n.replace(minute=0, second=0, microsecond=0)

def get_pred_times(now_hour):
    """
    t+1h  : jam berikutnya (misal 06.00 → 07.00)
    t+6h  : kelipatan 6 dari 00.00 (00, 06, 12, 18)
    t+12h : kelipatan 12 dari 00.00 (00, 12)
    t+24h : 00.00 malam berikutnya
    """
    times = {}

    # t+1h — selalu jam berikutnya
    times['t+1h'] = now_hour + timedelta(hours=1)

    # t+6h — kelipatan 6 dari 00.00 yang belum lewat
    midnight = now_hour.replace(hour=0, minute=0, second=0, microsecond=0)
    for slot in [0, 6, 12, 18, 24]:
        candidate = midnight + timedelta(hours=slot)
        if candidate > now_hour:
            times['t+6h'] = candidate
            break

    # t+12h — kelipatan 12 dari 00.00 yang belum lewat
    for slot in [0, 12, 24]:
        candidate = midnight + timedelta(hours=slot)
        if candidate > now_hour:
            times['t+12h'] = candidate
            break

    # t+24h — 00.00 malam berikutnya
    times['t+24h'] = (midnight + timedelta(days=1))

    return times

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="AirMind SG",
    page_icon="💨",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============================================================
# LANGUAGE
# ============================================================
if "lang" not in st.session_state:
    st.session_state.lang = "EN"

TEXTS = {
    "EN": {
        "title"          : "AirMind SG",
        "subtitle"       : "Real-time air quality monitoring & prediction for Singapore",
        "refresh"        : "Refresh",
        "live"           : "Live",
        "updated"        : "Updated",
        "map_title"      : "Air Quality Map",
        "map_now"        : "Current",
        "map_pred"       : "Forecast",
        "region_title"   : "By Region",
        "avg_label"      : "Singapore Average",
        "weather_title"  : "Weather",
        "temp"           : "Temperature",
        "wind"           : "Wind Speed",
        "rain"           : "Rainfall",
        "humidity"       : "Humidity",
        "pred_title"     : "Air Quality Forecast",
        "pred_caption"   : "AI predictions for Singapore. Forecast times are aligned to system clock.",
        "trend_title"    : "Forecast Trend",
        "now_label"      : "Now",
        "shap_title"     : "Why did the AI predict this?",
        "shap_caption"   : "Key factors influencing the air quality forecast.",
        "shap_factors"   : "Top Influencing Factors",
        "shap_influence" : "Influence",
        "model_used"     : "Model",
        "hotspot_title"  : "Forest Fire Hotspots",
        "hotspot_caption": "Smoke from Sumatra & Borneo can reach Singapore within 1–3 days.",
        "sum_hotspot"    : "Sumatra Hotspots",
        "kal_hotspot"    : "Borneo Hotspots",
        "sum_frp"        : "Sumatra Fire Energy",
        "kal_frp"        : "Borneo Fire Energy",
        "alert_high"     : "High haze risk! {} fire hotspots detected.",
        "alert_med"      : "Moderate fire activity ({} hotspots). Monitor air quality.",
        "alert_low"      : "Low fire activity ({} hotspots). Low transboundary haze risk.",
        "comp_title"     : "Prediction vs Actual (1-Hour Horizon)",
        "comp_caption"   : "Each hour, the prediction made 1 hour ago is compared against the actual PM2.5 reading.",
        "comp_mean"      : "SG Mean",
        "no_data"        : "No data yet. Data accumulates hourly as predictions are fulfilled.",
        "total_rec"      : "Records",
        "since"          : "Since",
        "raw_data"       : "Raw Data",
        "guide_title"    : "PM2.5 Index Guide",
        "footer"         : "Data: NEA Singapore · Open-Meteo · NASA FIRMS | Model: RF + BiLSTM",
        "horizon_label"  : {
            "t+1h" : "Next Hour",
            "t+6h" : "Next 6-Hour Mark",
            "t+12h": "Next 12-Hour Mark",
            "t+24h": "Midnight Tonight",
        },
        "categories": {
            "Good"      : ("Good",        "0–12 µg/m³",   "Safe for all activities"),
            "Moderate"  : ("Moderate",    "13–35 µg/m³",  "Sensitive groups: limit strenuous outdoor activity"),
            "Unhealthy" : ("Unhealthy",   "36–55 µg/m³",  "Everyone: wear a mask outdoors"),
            "Very Bad"  : ("Very Bad",    "56–150 µg/m³", "Avoid outdoor activities"),
            "Hazardous" : ("Hazardous",   ">150 µg/m³",   "Stay indoors"),
        },
        "advice": {
            "Good"      : "Air is clean. Safe for all outdoor activities.",
            "Moderate"  : "Acceptable. Sensitive groups should limit strenuous outdoor activity.",
            "Unhealthy" : "Unhealthy for all. Reduce time outdoors and wear a mask.",
            "Very Bad"  : "Very unhealthy. Avoid outdoor activities.",
            "Hazardous" : "Hazardous. Stay indoors.",
        },
    },
    "ID": {
        "title"          : "AirMind SG",
        "subtitle"       : "Pemantauan & prediksi kualitas udara Singapura secara real-time",
        "refresh"        : "Perbarui",
        "live"           : "Langsung",
        "updated"        : "Diperbarui",
        "map_title"      : "Peta Kualitas Udara",
        "map_now"        : "Saat Ini",
        "map_pred"       : "Prediksi",
        "region_title"   : "Per Wilayah",
        "avg_label"      : "Rata-rata Singapura",
        "weather_title"  : "Cuaca",
        "temp"           : "Suhu",
        "wind"           : "Kec. Angin",
        "rain"           : "Hujan",
        "humidity"       : "Kelembaban",
        "pred_title"     : "Prediksi Kualitas Udara",
        "pred_caption"   : "Prediksi AI untuk Singapura. Waktu prediksi mengikuti jam sistem.",
        "trend_title"    : "Tren Prediksi",
        "now_label"      : "Sekarang",
        "shap_title"     : "Mengapa AI memprediksi seperti ini?",
        "shap_caption"   : "Faktor-faktor yang paling mempengaruhi prediksi kualitas udara.",
        "shap_factors"   : "Faktor Paling Berpengaruh",
        "shap_influence" : "Pengaruh",
        "model_used"     : "Model",
        "hotspot_title"  : "Kebakaran Hutan",
        "hotspot_caption": "Asap dari Sumatera & Kalimantan dapat mencapai Singapura dalam 1–3 hari.",
        "sum_hotspot"    : "Titik Api Sumatera",
        "kal_hotspot"    : "Titik Api Kalimantan",
        "sum_frp"        : "Energi Api Sumatera",
        "kal_frp"        : "Energi Api Kalimantan",
        "alert_high"     : "Risiko kabut asap tinggi! {} titik api terdeteksi.",
        "alert_med"      : "Aktivitas kebakaran sedang ({} titik api). Pantau kualitas udara.",
        "alert_low"      : "Aktivitas kebakaran rendah ({} titik api). Risiko kabut asap rendah.",
        "comp_title"     : "Prediksi vs Aktual (Horizon 1 Jam)",
        "comp_caption"   : "Setiap jam, prediksi 1 jam sebelumnya dibandingkan dengan data PM2.5 aktual.",
        "comp_mean"      : "Rata-rata SG",
        "no_data"        : "Belum ada data. Data terakumulasi setiap jam.",
        "total_rec"      : "Rekaman",
        "since"          : "Sejak",
        "raw_data"       : "Data Mentah",
        "guide_title"    : "Panduan Indeks PM2.5",
        "footer"         : "Data: NEA Singapore · Open-Meteo · NASA FIRMS | Model: RF + BiLSTM",
        "horizon_label"  : {
            "t+1h" : "Jam Berikutnya",
            "t+6h" : "Kelipatan 6 Jam",
            "t+12h": "Kelipatan 12 Jam",
            "t+24h": "Tengah Malam",
        },
        "categories": {
            "Good"      : ("Baik",         "0–12 µg/m³",   "Aman untuk semua aktivitas"),
            "Moderate"  : ("Sedang",       "13–35 µg/m³",  "Kelompok sensitif: kurangi aktivitas berat"),
            "Unhealthy" : ("Tidak Sehat",  "36–55 µg/m³",  "Semua orang: gunakan masker di luar"),
            "Very Bad"  : ("Sangat Buruk", "56–150 µg/m³", "Hindari aktivitas luar ruangan"),
            "Hazardous" : ("Berbahaya",    ">150 µg/m³",   "Tetap di dalam ruangan"),
        },
        "advice": {
            "Good"      : "Udara bersih. Aman untuk semua aktivitas luar ruangan.",
            "Moderate"  : "Cukup baik. Kelompok sensitif sebaiknya kurangi aktivitas berat di luar.",
            "Unhealthy" : "Tidak sehat. Kurangi waktu di luar dan gunakan masker.",
            "Very Bad"  : "Sangat tidak sehat. Hindari aktivitas luar ruangan.",
            "Hazardous" : "Berbahaya. Tetap di dalam ruangan.",
        },
    }
}

def t(key):
    return TEXTS[st.session_state.lang].get(key, key)

def thl(horizon):
    return TEXTS[st.session_state.lang]["horizon_label"][horizon]

def tc(key):
    return TEXTS[st.session_state.lang]["categories"][key]

def ta(val):
    lang = st.session_state.lang
    if val is None: return ""
    if val <= 12:    return TEXTS[lang]["advice"]["Good"]
    elif val <= 35:  return TEXTS[lang]["advice"]["Moderate"]
    elif val <= 55:  return TEXTS[lang]["advice"]["Unhealthy"]
    elif val <= 150: return TEXTS[lang]["advice"]["Very Bad"]
    else:            return TEXTS[lang]["advice"]["Hazardous"]

# ============================================================
# KONSTANTA
# ============================================================
FIRMS_API_KEY = "aeeb0e07b5ea9b583008b4dbd86eff81"
WINDOW        = 24
TARGET_COLS   = ['target_t1h','target_t6h','target_t12h','target_t24h']
HORIZONS      = ['t+1h','t+6h','t+12h','t+24h']

REGIONS = {
    "central": {"lat":1.35735,  "lon":103.82000, "label_en":"Central", "label_id":"Tengah"},
    "east"   : {"lat":1.35735,  "lon":103.94000, "label_en":"East",    "label_id":"Timur"},
    "north"  : {"lat":1.41803,  "lon":103.82000, "label_en":"North",   "label_id":"Utara"},
    "south"  : {"lat":1.29587,  "lon":103.82000, "label_en":"South",   "label_id":"Selatan"},
    "west"   : {"lat":1.35735,  "lon":103.70000, "label_en":"West",    "label_id":"Barat"},
}

def region_label(region):
    k = "label_en" if st.session_state.lang=="EN" else "label_id"
    return REGIONS[region][k]

FEATURE_NAMES = {
    "EN": {
        "pm25_east":"PM2.5 — East SG","pm25_central":"PM2.5 — Central SG",
        "pm25_north":"PM2.5 — North SG","pm25_south":"PM2.5 — South SG",
        "pm25_west":"PM2.5 — West SG","pm25_mean":"PM2.5 — SG Average",
        "pm25_lag_1h":"PM2.5 — 1hr Ago","pm25_lag_6h":"PM2.5 — 6hr Ago",
        "pm25_lag_12h":"PM2.5 — 12hr Ago","pm25_lag_24h":"PM2.5 — Yesterday",
        "pm25_lag_48h":"PM2.5 — 2 Days Ago","pm25_lag_168h":"PM2.5 — 1 Week Ago",
        "pm25_rolling_mean_6h":"6hr Rolling Avg","pm25_rolling_mean_12h":"12hr Rolling Avg",
        "pm25_rolling_mean_24h":"24hr Rolling Avg","pm25_rolling_mean_48h":"48hr Rolling Avg",
        "pm25_rolling_std_6h":"6hr PM2.5 Variability","pm25_rolling_std_24h":"24hr PM2.5 Variability",
        "windspeed_10m":"Wind Speed","winddir_cos":"Wind Direction (E-W)",
        "winddir_sin":"Wind Direction (N-S)","temperature_2m":"Air Temperature",
        "relative_humidity_2m":"Humidity","precipitation":"Rainfall",
        "pressure_msl":"Atmospheric Pressure","cloudcover":"Cloud Cover",
        "hotspot_count_sumatera":"Sumatra Fire Hotspots",
        "hotspot_count_kalimantan":"Borneo Fire Hotspots",
        "hotspot_sum_lag1d":"Sumatra Hotspots — Yesterday",
        "hotspot_sum_lag2d":"Sumatra Hotspots — 2 Days Ago",
        "hotspot_sum_lag3d":"Sumatra Hotspots — 3 Days Ago",
        "hotspot_kal_lag1d":"Borneo Hotspots — Yesterday",
        "hotspot_kal_lag2d":"Borneo Hotspots — 2 Days Ago",
        "hotspot_kal_lag3d":"Borneo Hotspots — 3 Days Ago",
        "month_sin":"Seasonal Pattern","month_cos":"Seasonal Pattern",
        "hour":"Time of Day","is_dry_season":"Dry Season",
    },
    "ID": {
        "pm25_east":"PM2.5 — Timur","pm25_central":"PM2.5 — Tengah",
        "pm25_north":"PM2.5 — Utara","pm25_south":"PM2.5 — Selatan",
        "pm25_west":"PM2.5 — Barat","pm25_mean":"PM2.5 — Rata-rata SG",
        "pm25_lag_1h":"PM2.5 — 1 Jam Lalu","pm25_lag_6h":"PM2.5 — 6 Jam Lalu",
        "pm25_lag_12h":"PM2.5 — 12 Jam Lalu","pm25_lag_24h":"PM2.5 — Kemarin",
        "pm25_lag_48h":"PM2.5 — 2 Hari Lalu","pm25_lag_168h":"PM2.5 — Seminggu Lalu",
        "pm25_rolling_mean_6h":"Rata-rata 6 Jam","pm25_rolling_mean_12h":"Rata-rata 12 Jam",
        "pm25_rolling_mean_24h":"Rata-rata 24 Jam","pm25_rolling_mean_48h":"Rata-rata 48 Jam",
        "pm25_rolling_std_6h":"Variabilitas 6 Jam","pm25_rolling_std_24h":"Variabilitas 24 Jam",
        "windspeed_10m":"Kec. Angin","winddir_cos":"Arah Angin (T-B)",
        "winddir_sin":"Arah Angin (U-S)","temperature_2m":"Suhu Udara",
        "relative_humidity_2m":"Kelembaban","precipitation":"Curah Hujan",
        "pressure_msl":"Tekanan Atmosfer","cloudcover":"Tutupan Awan",
        "hotspot_count_sumatera":"Titik Api Sumatera",
        "hotspot_count_kalimantan":"Titik Api Kalimantan",
        "hotspot_sum_lag1d":"Hotspot Sumatera Kemarin",
        "hotspot_sum_lag2d":"Hotspot Sumatera 2 Hari Lalu",
        "hotspot_sum_lag3d":"Hotspot Sumatera 3 Hari Lalu",
        "hotspot_kal_lag1d":"Hotspot Kalimantan Kemarin",
        "hotspot_kal_lag2d":"Hotspot Kalimantan 2 Hari Lalu",
        "hotspot_kal_lag3d":"Hotspot Kalimantan 3 Hari Lalu",
        "month_sin":"Pola Musiman","month_cos":"Pola Musiman",
        "hour":"Jam dalam Sehari","is_dry_season":"Musim Kemarau",
    }
}

def get_feat_name(feat):
    return FEATURE_NAMES[st.session_state.lang].get(feat, feat)

def pm25_cat(val):
    lang = st.session_state.lang
    if val is None or (isinstance(val,float) and np.isnan(val)):
        return "N/A","#94a3b8"
    if val<=12:    return ("Good"      if lang=="EN" else "Baik"),        "#22c55e"
    elif val<=35:  return ("Moderate"  if lang=="EN" else "Sedang"),      "#f59e0b"
    elif val<=55:  return ("Unhealthy" if lang=="EN" else "Tidak Sehat"), "#ef4444"
    elif val<=150: return ("Very Bad"  if lang=="EN" else "Sangat Buruk"),"#8b5cf6"
    else:          return ("Hazardous" if lang=="EN" else "Berbahaya"),   "#7f1d1d"

# ============================================================
# CSS
# ============================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html,body,[class*="css"]{font-family:'Inter',sans-serif;}
.block-container{padding-top:1.5rem;padding-bottom:2rem;max-width:1400px;}
.app-title{font-size:1.8rem;font-weight:800;color:#0f172a;letter-spacing:-0.5px;margin:0;}
.app-sub{font-size:0.85rem;color:#94a3b8;margin-top:2px;}
.live-wrap{display:flex;align-items:center;gap:8px;margin-bottom:1.2rem;}
.live-dot{width:8px;height:8px;background:#22c55e;border-radius:50%;animation:pulse 2s infinite;}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:0.3}}
.live-txt{font-size:0.78rem;color:#94a3b8;font-weight:500;}
.card{background:white;border-radius:16px;padding:1.2rem 1.4rem;
      box-shadow:0 1px 3px rgba(0,0,0,0.06),0 4px 16px rgba(0,0,0,0.04);margin-bottom:0.8rem;}
.card-label{font-size:0.7rem;font-weight:600;color:#94a3b8;
            text-transform:uppercase;letter-spacing:0.8px;margin-bottom:6px;}
.card-value{font-size:2.4rem;font-weight:800;color:#0f172a;line-height:1;margin-bottom:4px;}
.card-unit{font-size:0.75rem;color:#cbd5e1;}
.card-badge{display:inline-block;padding:4px 12px;border-radius:6px;
            font-size:0.78rem;font-weight:700;margin-top:8px;}
.region-item{display:flex;justify-content:space-between;align-items:center;
             padding:9px 14px;background:#f8fafc;border-radius:10px;
             margin-bottom:5px;border-left:3px solid #e2e8f0;}
.region-name{font-size:0.85rem;font-weight:600;color:#334155;}
.region-val{font-size:0.9rem;font-weight:800;}
.region-cat{font-size:0.72rem;font-weight:600;padding:2px 8px;border-radius:4px;}
.advice-box{background:#f0fdf4;border:1px solid #bbf7d0;border-radius:10px;
            padding:10px 14px;font-size:0.83rem;color:#166534;line-height:1.5;margin:8px 0;}
.advice-warn{background:#fffbeb;border-color:#fde68a;color:#92400e;}
.advice-danger{background:#fff1f2;border-color:#fecdd3;color:#9f1239;}
.pred-card{background:white;border-radius:14px;padding:1.3rem 1rem;text-align:center;
           box-shadow:0 1px 3px rgba(0,0,0,0.06),0 4px 12px rgba(0,0,0,0.04);
           border-top:3px solid #e2e8f0;height:100%;}
.pred-horizon{font-size:0.7rem;font-weight:700;color:#94a3b8;
              text-transform:uppercase;letter-spacing:0.6px;}
.pred-time{font-size:0.82rem;font-weight:700;color:#334155;margin-bottom:6px;}
.pred-date{font-size:0.7rem;color:#cbd5e1;margin-bottom:8px;}
.pred-value{font-size:2.2rem;font-weight:800;line-height:1;}
.pred-unit{font-size:0.7rem;color:#cbd5e1;margin-bottom:8px;}
.pred-badge{display:inline-block;padding:3px 10px;border-radius:6px;font-size:0.75rem;font-weight:700;}
.pred-model{font-size:0.65rem;color:#cbd5e1;margin-top:8px;}
.sec-title{font-size:1rem;font-weight:700;color:#0f172a;margin:1.2rem 0 0.3rem;}
.sec-cap{font-size:0.8rem;color:#94a3b8;margin-bottom:1rem;}
.shap-item{margin-bottom:12px;}
.shap-name{font-size:0.82rem;font-weight:600;color:#334155;margin-bottom:4px;}
.shap-bar-bg{background:#f1f5f9;border-radius:4px;height:8px;overflow:hidden;}
.shap-pct{font-size:0.7rem;color:#94a3b8;margin-top:2px;}
.hs-card{background:white;border-radius:14px;padding:1.2rem;
         text-align:center;box-shadow:0 1px 3px rgba(0,0,0,0.06);}
.hs-num{font-size:1.8rem;font-weight:800;color:#0f172a;}
.hs-label{font-size:0.75rem;color:#94a3b8;font-weight:500;margin-top:2px;}
.mbox{background:#f8fafc;border-radius:10px;padding:10px 14px;
      text-align:center;border:1px solid #e2e8f0;}
.mbox-num{font-size:1.4rem;font-weight:800;color:#0f172a;}
.mbox-lbl{font-size:0.72rem;color:#94a3b8;font-weight:500;margin-top:2px;}
.leg-card{border-radius:10px;padding:10px 12px;text-align:center;}
.leg-cat{font-size:0.82rem;font-weight:700;margin-bottom:2px;}
.leg-range{font-size:0.75rem;font-weight:600;color:#475569;margin-bottom:4px;}
.leg-desc{font-size:0.7rem;color:#64748b;line-height:1.4;}
hr{border:none;border-top:1px solid #f1f5f9;margin:1.5rem 0;}
</style>
""", unsafe_allow_html=True)

# ============================================================
# DATABASE
# ============================================================
DB_PATH = "airmind_cache.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c    = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS pred_1h (
            id             INTEGER PRIMARY KEY AUTOINCREMENT,
            pred_for       TEXT NOT NULL UNIQUE,
            predicted_mean REAL,
            actual_mean    REAL,
            actual_central REAL,
            actual_east    REAL,
            actual_north   REAL,
            actual_south   REAL,
            actual_west    REAL,
            created_at     TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()

def save_pred_1h(pred_val, target_time):
    conn     = sqlite3.connect(DB_PATH)
    c        = conn.cursor()
    time_str = target_time.strftime("%Y-%m-%d %H:00")
    try:
        c.execute("""
            INSERT OR IGNORE INTO pred_1h (pred_for, predicted_mean)
            VALUES (?, ?)
        """, (time_str, pred_val))
        conn.commit()
    except: pass
    conn.close()

def update_actual_1h(pm25_now, now_hour):
    conn     = sqlite3.connect(DB_PATH)
    c        = conn.cursor()
    now_str  = now_hour.strftime("%Y-%m-%d %H:00")
    mean_val = np.nanmean([v for v in pm25_now.values() if v is not None])
    c.execute("""
        UPDATE pred_1h
        SET actual_mean=?, actual_central=?, actual_east=?,
            actual_north=?, actual_south=?, actual_west=?
        WHERE pred_for=? AND actual_mean IS NULL
    """, (mean_val,
          pm25_now.get('central'), pm25_now.get('east'),
          pm25_now.get('north'),   pm25_now.get('south'),
          pm25_now.get('west'),    now_str))
    conn.commit()
    conn.close()

def get_comparison_data(limit=72):
    conn = sqlite3.connect(DB_PATH)
    df   = pd.read_sql_query("""
        SELECT pred_for, predicted_mean,
               actual_mean, actual_central, actual_east,
               actual_north, actual_south, actual_west
        FROM pred_1h
        WHERE actual_mean IS NOT NULL
        ORDER BY pred_for DESC LIMIT ?
    """, conn, params=(limit,))
    conn.close()
    return df

def get_db_stats():
    conn = sqlite3.connect(DB_PATH)
    c    = conn.cursor()
    c.execute("SELECT COUNT(*) FROM pred_1h WHERE actual_mean IS NOT NULL")
    total = c.fetchone()[0]
    c.execute("SELECT MIN(pred_for) FROM pred_1h WHERE actual_mean IS NOT NULL")
    since = c.fetchone()[0]
    conn.close()
    return total, since

init_db()

# ============================================================
# LOAD MODEL
# ============================================================
@st.cache_resource
def load_models():
    try:
        scaler_X     = joblib.load("model/scaler_X.pkl")
        scalers_y    = joblib.load("model/scalers_y.pkl")
        feature_cols = joblib.load("model/feature_cols.pkl")
        with open("model/shap_summary.json") as f:
            shap_summary = json.load(f)

        class BiLSTMModel(nn.Module):
            def __init__(self, input_size, hidden_size=128,
                         num_layers=2, dropout=0.2):
                super().__init__()
                self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                                    batch_first=True, dropout=dropout,
                                    bidirectional=True)
                self.norm = nn.LayerNorm(hidden_size*2)
                self.fc   = nn.Sequential(
                    nn.Linear(hidden_size*2,64), nn.ReLU(),
                    nn.Dropout(dropout), nn.Linear(64,1)
                )
            def forward(self, x):
                out,_ = self.lstm(x)
                return self.fc(self.norm(out[:,-1,:]))

        bilstm = {}
        for tgt in ['target_t12h','target_t24h']:
            m = BiLSTMModel(input_size=len(feature_cols))
            m.load_state_dict(torch.load(
                f"model/bilstm_{tgt}_best.pt", map_location='cpu'))
            m.eval()
            bilstm[tgt] = m

        rf = {}
        for tgt in ['target_t1h','target_t6h']:
            rf[tgt] = joblib.load(f"model/rf_{tgt}.pkl")

        return scaler_X, scalers_y, feature_cols, shap_summary, bilstm, rf
    except Exception as e:
        st.error(f"Model load error: {e}")
        return None,None,None,None,None,None

(scaler_X, scalers_y, feature_cols,
 shap_summary, bilstm_models, rf_models) = load_models()

# ============================================================
# FETCH DATA
# ============================================================
@st.cache_data(ttl=3600)
def fetch_pm25():
    try:
        r  = requests.get("https://api.data.gov.sg/v1/environment/pm25", timeout=10)
        rd = r.json()["items"][0]["readings"]["pm25_one_hourly"]
        return {reg: rd.get(reg) for reg in REGIONS}
    except:
        return {reg: 15.0 for reg in REGIONS}

@st.cache_data(ttl=3600)
def fetch_weather():
    try:
        params = {
            "latitude":1.3521,"longitude":103.8198,
            "current":"temperature_2m,relative_humidity_2m,precipitation,"
                      "windspeed_10m,winddirection_10m,pressure_msl,cloudcover",
            "timezone":"Asia/Singapore"
        }
        r = requests.get("https://api.open-meteo.com/v1/forecast",
                         params=params, timeout=10)
        return r.json()["current"]
    except:
        return {"temperature_2m":28,"relative_humidity_2m":80,
                "precipitation":0,"windspeed_10m":10,
                "winddirection_10m":180,"pressure_msl":1013,"cloudcover":50}

@st.cache_data(ttl=86400)
def fetch_hotspot():
    try:
        result = {}
        for area, bbox in [("sumatera","95,-6,109,6"),
                            ("kalimantan","108,-5,119,5")]:
            url = (f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/"
                   f"{FIRMS_API_KEY}/VIIRS_SNPP_NRT/{bbox}/1")
            r   = requests.get(url, timeout=15)
            if r.status_code == 200:
                import io, csv
                rows  = list(csv.DictReader(io.StringIO(r.text)))
                valid = [row for row in rows
                         if row.get('confidence','') in ['n','h']]
                frp   = sum(float(row.get('frp',0)) for row in valid
                            if row.get('frp','0').replace('.','').replace('-','').isdigit())
                result[f"count_{area}"]   = len(valid)
                result[f"frp_sum_{area}"] = frp
            else:
                result[f"count_{area}"]   = 0
                result[f"frp_sum_{area}"] = 0.0
        return result
    except:
        return {"count_sumatera":0,"frp_sum_sumatera":0,
                "count_kalimantan":0,"frp_sum_kalimantan":0}

# ============================================================
# PREDIKSI
# ============================================================
def build_fv(pm25_mean, pm25_now, weather, hotspot, now_hour):
    row = []
    for feat in feature_cols:
        if feat == 'pm25_mean':
            row.append(pm25_mean)
        elif feat.startswith('pm25_') and not any(
            x in feat for x in ['lag','rolling','mean']
        ):
            reg = feat.replace('pm25_','')
            row.append(pm25_now.get(reg, pm25_mean) or pm25_mean)
        elif feat == 'temperature_2m':
            row.append(weather.get('temperature_2m',28))
        elif feat == 'relative_humidity_2m':
            row.append(weather.get('relative_humidity_2m',80))
        elif feat == 'precipitation':
            row.append(weather.get('precipitation',0))
        elif feat == 'windspeed_10m':
            row.append(weather.get('windspeed_10m',10))
        elif feat == 'pressure_msl':
            row.append(weather.get('pressure_msl',1013))
        elif feat == 'cloudcover':
            row.append(weather.get('cloudcover',50))
        elif feat == 'winddir_sin':
            row.append(np.sin(np.deg2rad(weather.get('winddirection_10m',180))))
        elif feat == 'winddir_cos':
            row.append(np.cos(np.deg2rad(weather.get('winddirection_10m',180))))
        elif feat == 'hour':
            row.append(now_hour.hour)
        elif feat == 'month':
            row.append(now_hour.month)
        elif feat == 'month_sin':
            row.append(np.sin(2*np.pi*now_hour.month/12))
        elif feat == 'month_cos':
            row.append(np.cos(2*np.pi*now_hour.month/12))
        elif feat == 'is_dry_season':
            row.append(1 if now_hour.month in [6,7,8,9] else 0)
        elif feat == 'hotspot_count_sumatera':
            row.append(hotspot.get('count_sumatera',0))
        elif feat == 'hotspot_frp_mean_sumatera':
            row.append(hotspot.get('frp_sum_sumatera',0))
        elif feat == 'hotspot_count_kalimantan':
            row.append(hotspot.get('count_kalimantan',0))
        elif feat == 'hotspot_frp_mean_kalimantan':
            row.append(hotspot.get('frp_sum_kalimantan',0))
        elif 'lag' in feat or 'rolling' in feat:
            row.append(pm25_mean)
        elif 'hotspot' in feat:
            row.append(0)
        else:
            row.append(0)
    return np.array(row, dtype=np.float32)

def make_predictions(pm25_now, weather, hotspot, now_hour):
    if scaler_X is None: return None, None
    try:
        pm25_mean = np.nanmean([v for v in pm25_now.values() if v])
        fv        = build_fv(pm25_mean, pm25_now, weather, hotspot, now_hour)
        X_sc      = scaler_X.transform(np.tile(fv,(WINDOW,1)))

        preds = {}
        for tgt, h in [('target_t1h','t+1h'),('target_t6h','t+6h')]:
            ps = rf_models[tgt].predict(X_sc[-1].reshape(1,-1))[0]
            preds[h] = max(0, float(scalers_y[tgt].inverse_transform([[ps]])[0][0]))

        X_t = torch.FloatTensor(X_sc).unsqueeze(0)
        for tgt, h in [('target_t12h','t+12h'),('target_t24h','t+24h')]:
            with torch.no_grad():
                ps = bilstm_models[tgt](X_t).numpy().flatten()[0]
            preds[h] = max(0, float(scalers_y[tgt].inverse_transform([[ps]])[0][0]))

        return preds, pm25_mean
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None, None

# ============================================================
# PETA
# ============================================================
def create_map(pm25_now, pred_mean=None, show_pred=False):
    m = folium.Map(location=[1.3521,103.8198], zoom_start=11,
                   tiles="CartoDB positron")
    for region, info in REGIONS.items():
        val  = (pred_mean if show_pred and pred_mean else None) or pm25_now.get(region) or 0
        mode = t('map_pred') if show_pred else t('map_now')
        cat, color = pm25_cat(val)
        label = region_label(region)

        folium.CircleMarker(
            location=[info["lat"],info["lon"]],
            radius=36, color=color, fill=True,
            fill_color=color, fill_opacity=0.7,
            popup=folium.Popup(
                f"<b>{label}</b><br>{mode}<br>"
                f"PM2.5: <b>{val:.1f} µg/m³</b><br>"
                f"<b style='color:{color}'>{cat}</b>", max_width=160),
            tooltip=f"{label}: {val:.1f} µg/m³ — {cat}"
        ).add_to(m)
        folium.Marker(
            location=[info["lat"],info["lon"]],
            icon=folium.DivIcon(
                html=f'<div style="font-size:11px;font-weight:800;color:white;'
                     f'text-align:center;text-shadow:1px 1px 3px rgba(0,0,0,0.9)">'
                     f'{label}<br>{val:.0f}</div>',
                icon_size=(70,30), icon_anchor=(35,15))
        ).add_to(m)
    return m

# ============================================================
# CHART HELPERS
# ============================================================
BASE_LAYOUT = dict(
    plot_bgcolor='white', paper_bgcolor='white',
    font=dict(family="Inter, sans-serif", color="#334155", size=11),
    margin=dict(l=0, r=60, t=10, b=0)
)

def add_thresholds(fig):
    for y, lbl, col in [
        (12,"Good/Baik","#22c55e"),
        (35,"Moderate/Sedang","#f59e0b"),
        (55,"Unhealthy","#ef4444")
    ]:
        fig.add_hline(y=y, line_dash="dot", line_color=col, opacity=0.35,
                      annotation_text=lbl, annotation_position="right",
                      annotation_font_size=9)

# ============================================================
# MAIN APP
# ============================================================

# Header
c1, c2, c3 = st.columns([4, 0.55, 0.55])
with c1:
    st.markdown(
        f'<div class="app-title">{t("title")}</div>'
        f'<div class="app-sub">{t("subtitle")}</div>',
        unsafe_allow_html=True
    )
with c2:
    st.markdown("<br>", unsafe_allow_html=True)
    lp = st.radio("", ["EN","ID"], horizontal=True,
                   index=0 if st.session_state.lang=="EN" else 1,
                   label_visibility="collapsed")
    if lp != st.session_state.lang:
        st.session_state.lang = lp
        st.rerun()
with c3:
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button(t("refresh"), use_container_width=True):
        st.cache_data.clear()
        st.rerun()

# Fetch data
with st.spinner(""):
    pm25_now = fetch_pm25()
    weather  = fetch_weather()
    hotspot  = fetch_hotspot()

now      = now_sgt()
now_hour = current_hour_sgt()
ts_str   = now.strftime("%d %b %Y, %H:%M SGT")

st.markdown(
    f'<div class="live-wrap">'
    f'<div class="live-dot"></div>'
    f'<span class="live-txt">{t("live")} &nbsp;·&nbsp; {t("updated")}: {ts_str}</span>'
    f'</div>',
    unsafe_allow_html=True
)

# Prediksi + waktu target
predictions, pm25_mean_val = make_predictions(pm25_now, weather, hotspot, now_hour)
pred_times = get_pred_times(now_hour)
pm25_vals  = [v for v in pm25_now.values() if v is not None]
pm25_mean  = np.mean(pm25_vals) if pm25_vals else 0

# Simpan t+1h ke DB
if predictions:
    save_pred_1h(predictions['t+1h'], pred_times['t+1h'])
update_actual_1h(pm25_now, now_hour)

# ============================================================
# S1 — MAP + STATUS
# ============================================================
st.markdown(f'<div class="sec-title">{t("map_title")}</div>',
            unsafe_allow_html=True)

map_opts = [t("map_now")] + [
    f'{t("map_pred")} — {thl(h)} ({pred_times[h].strftime("%H:%M")} SGT)'
    for h in HORIZONS
]
map_mode  = st.radio("", map_opts, horizontal=True,
                     label_visibility="collapsed")
show_pred = map_mode != t("map_now")
sel_h     = HORIZONS[map_opts.index(map_mode)-1] if show_pred else None
pred_map_val = predictions.get(sel_h) if predictions and sel_h else None

col_map, col_info = st.columns([1.5, 1])
with col_map:
    sg_map = create_map(pm25_now, pred_map_val, show_pred=show_pred)
    st_folium(sg_map, width=560, height=400, returned_objects=[])

with col_info:
    cat_avg, color_avg = pm25_cat(pm25_mean)
    adv_cls = "advice-box"
    if pm25_mean > 55:   adv_cls = "advice-box advice-danger"
    elif pm25_mean > 35: adv_cls = "advice-box advice-warn"

    st.markdown(f"""
    <div class="card" style="border-left:4px solid {color_avg}">
        <div class="card-label">{t('avg_label')}</div>
        <div class="card-value" style="color:{color_avg}">{pm25_mean:.1f}</div>
        <div class="card-unit">µg/m³ PM2.5</div>
        <span class="card-badge" style="background:{color_avg}18;color:{color_avg}">
            {cat_avg}
        </span>
    </div>
    <div class="{adv_cls}">{ta(pm25_mean)}</div>
    """, unsafe_allow_html=True)

    st.markdown(f'<div class="sec-title" style="font-size:0.88rem">'
                f'{t("region_title")}</div>', unsafe_allow_html=True)
    for region in REGIONS:
        val = pm25_now.get(region) or 0
        cat, color = pm25_cat(val)
        st.markdown(f"""
        <div class="region-item" style="border-left-color:{color}">
            <span class="region-name">{region_label(region)}</span>
            <span class="region-val" style="color:{color}">{val:.0f} µg/m³</span>
            <span class="region-cat" style="background:{color}18;color:{color}">{cat}</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown(f'<div class="sec-title" style="font-size:0.88rem;margin-top:1rem">'
                f'{t("weather_title")}</div>', unsafe_allow_html=True)
    wc1, wc2 = st.columns(2)
    wc1.metric(t("temp"),     f"{weather.get('temperature_2m','--')}°C")
    wc2.metric(t("wind"),     f"{weather.get('windspeed_10m','--')} km/h")
    wc3, wc4 = st.columns(2)
    wc3.metric(t("humidity"), f"{weather.get('relative_humidity_2m','--')}%")
    wc4.metric(t("rain"),     f"{weather.get('precipitation','--')} mm")

# ============================================================
# S2 — FORECAST
# ============================================================
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown(f'<div class="sec-title">{t("pred_title")}</div>',
            unsafe_allow_html=True)
st.markdown(f'<div class="sec-cap">{t("pred_caption")}</div>',
            unsafe_allow_html=True)

if predictions:
    p_cols = st.columns(4)
    for col, horizon in zip(p_cols, HORIZONS):
        val        = predictions[horizon]
        cat, color = pm25_cat(val)
        pt         = pred_times[horizon]
        time_lbl   = pt.strftime('%H:%M')
        date_lbl   = pt.strftime('%d %b')
        mdl        = "Random Forest" if horizon in ['t+1h','t+6h'] else "BiLSTM"
        with col:
            st.markdown(f"""
            <div class="pred-card" style="border-top-color:{color}">
                <div class="pred-horizon">{thl(horizon)}</div>
                <div class="pred-time">{time_lbl} SGT</div>
                <div class="pred-date">{date_lbl}</div>
                <div class="pred-value" style="color:{color}">{val:.1f}</div>
                <div class="pred-unit">µg/m³</div>
                <span class="pred-badge" style="background:{color}18;color:{color}">
                    {cat}
                </span>
                <div class="pred-model">{t('model_used')}: {mdl}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f'<div class="sec-title" style="font-size:0.9rem">'
                f'{t("trend_title")}</div>', unsafe_allow_html=True)

    times_chart = [now_hour.strftime("%Y-%m-%d %H:%M")] + [
        pred_times[h].strftime("%Y-%m-%d %H:%M") for h in HORIZONS
    ]
    vals_chart = [pm25_mean] + [predictions[h] for h in HORIZONS]
    clr_pts    = [pm25_cat(v)[1] for v in vals_chart]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=times_chart, y=vals_chart, mode='lines+markers',
        line=dict(color='#3b82f6', width=2.5),
        marker=dict(size=9, color=clr_pts, line=dict(width=2,color='white')),
        fill='tozeroy', fillcolor='rgba(59,130,246,0.06)',
        hovertemplate='%{x|%H:%M %d %b}<br>PM2.5: %{y:.1f} µg/m³<extra></extra>'
    ))
    add_thresholds(fig)
    fig.add_shape(
        type="line", x0=times_chart[0], x1=times_chart[0],
        y0=0, y1=1, yref="paper",
        line=dict(color="#94a3b8", dash="dash", width=1.5)
    )
    fig.add_annotation(
        x=times_chart[0], y=1, yref="paper", text=t("now_label"),
        showarrow=False, font=dict(size=10, color="#94a3b8"), xanchor="left"
    )
    fig.update_layout(
        height=260, showlegend=False,
        xaxis=dict(tickformat="%H:%M\n%d %b", gridcolor="#f1f5f9"),
        yaxis=dict(title="PM2.5 (µg/m³)", rangemode='tozero', gridcolor="#f1f5f9"),
        **BASE_LAYOUT
    )
    st.plotly_chart(fig, use_container_width=True)

# ============================================================
# S3 — SHAP
# ============================================================
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown(f'<div class="sec-title">{t("shap_title")}</div>',
            unsafe_allow_html=True)
st.markdown(f'<div class="sec-cap">{t("shap_caption")}</div>',
            unsafe_allow_html=True)

if shap_summary:
    tabs = st.tabs([thl(h) for h in HORIZONS])
    for tab, horizon in zip(tabs, HORIZONS):
        with tab:
            summary  = shap_summary.get(horizon, {})
            features = summary.get("features", [])
            values   = summary.get("shap_values", [])
            if not features: continue
            max_val = max(values) if values else 1

            cc1, cc2 = st.columns([1.3, 1])
            with cc1:
                sidx    = np.argsort(values)
                fs      = [get_feat_name(features[i]) for i in sidx]
                vs      = [values[i] for i in sidx]
                norms   = [v/max_val for v in vs]
                bcolors = [f"rgba(59,130,246,{0.35+0.65*n})"
                           if n<=0.5 else f"rgba(239,68,68,{0.35+0.65*n})"
                           for n in norms]
                fig_s = go.Figure(go.Bar(
                    x=vs, y=fs, orientation='h',
                    marker=dict(color=bcolors, line_width=0),
                    hovertemplate='%{y}<br>%{x:.5f}<extra></extra>'
                ))
                fig_s.update_layout(
                    height=300, margin=dict(l=0,r=10,t=0,b=0),
                    xaxis=dict(title=t("shap_influence"), gridcolor="#f1f5f9"),
                    yaxis=dict(gridcolor="#f1f5f9"),
                    **{k:v for k,v in BASE_LAYOUT.items()
                       if k not in ['margin']}
                )
                st.plotly_chart(fig_s, use_container_width=True)

            with cc2:
                st.markdown(
                    f'<div style="font-size:0.82rem;font-weight:700;'
                    f'color:#334155;margin-bottom:12px">{t("shap_factors")}</div>',
                    unsafe_allow_html=True
                )
                for rank, (feat, val) in enumerate(zip(features[:5], values[:5])):
                    pct   = int(val/max_val*100)
                    label = get_feat_name(feat)
                    st.markdown(f"""
                    <div class="shap-item">
                        <div class="shap-name">
                            <span style="color:#cbd5e1;font-size:0.7rem;margin-right:6px">
                                {rank+1:02d}
                            </span>{label}
                        </div>
                        <div class="shap-bar-bg">
                            <div style="background:#3b82f6;width:{pct}%;
                                        height:100%;border-radius:4px"></div>
                        </div>
                        <div class="shap-pct">{pct}% {t('shap_influence').lower()}</div>
                    </div>
                    """, unsafe_allow_html=True)
                mdl = "Random Forest" if horizon in ['t+1h','t+6h'] else "BiLSTM"
                st.markdown(
                    f'<div style="margin-top:12px;padding:8px 12px;background:#f8fafc;'
                    f'border-radius:8px;font-size:0.78rem;color:#64748b">'
                    f'{t("model_used")}: <b>{mdl}</b></div>',
                    unsafe_allow_html=True
                )

# ============================================================
# S4 — HOTSPOT
# ============================================================
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown(f'<div class="sec-title">{t("hotspot_title")}</div>',
            unsafe_allow_html=True)
st.markdown(f'<div class="sec-cap">{t("hotspot_caption")}</div>',
            unsafe_allow_html=True)

total_hs = (hotspot.get('count_sumatera',0) + hotspot.get('count_kalimantan',0))
for col, (lbl,val,unit) in zip(st.columns(4),[
    (t("sum_hotspot"),  hotspot.get('count_sumatera',0),    "hotspots"),
    (t("kal_hotspot"),  hotspot.get('count_kalimantan',0),  "hotspots"),
    (t("sum_frp"),      hotspot.get('frp_sum_sumatera',0),  "MW"),
    (t("kal_frp"),      hotspot.get('frp_sum_kalimantan',0),"MW"),
]):
    disp = int(val) if unit=='hotspots' else f"{val:.0f}"
    with col:
        st.markdown(f"""
        <div class="hs-card">
            <div class="hs-num">{disp}</div>
            <div class="hs-label">{lbl} ({unit})</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
if total_hs>500:   st.error(t("alert_high").format(total_hs))
elif total_hs>100: st.warning(t("alert_med").format(total_hs))
else:              st.success(t("alert_low").format(total_hs))

# ============================================================
# S5 — PREDICTION VS ACTUAL
# ============================================================
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown(f'<div class="sec-title">{t("comp_title")}</div>',
            unsafe_allow_html=True)
st.markdown(f'<div class="sec-cap">{t("comp_caption")}</div>',
            unsafe_allow_html=True)

total_rec, since_db = get_db_stats()
db1, db2 = st.columns(2)
for col,(lbl,val) in zip([db1,db2],[
    (t("total_rec"), f"{total_rec:,}"),
    (t("since"), since_db[:16] if since_db else "-"),
]):
    with col:
        st.markdown(f"""
        <div class="mbox"><div class="mbox-num">{val}</div>
        <div class="mbox-lbl">{lbl}</div></div>
        """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
df_comp = get_comparison_data(limit=72)

if df_comp.empty:
    st.info(t("no_data"))
else:
    df_comp['pred_for'] = pd.to_datetime(df_comp['pred_for'])
    df_comp = df_comp.sort_values('pred_for')

    tab_map = {
        t("comp_mean")         : "actual_mean",
        region_label("central"): "actual_central",
        region_label("east")   : "actual_east",
        region_label("north")  : "actual_north",
        region_label("south")  : "actual_south",
        region_label("west")   : "actual_west",
    }
    comp_tabs = st.tabs(list(tab_map.keys()))

    for tab, (tab_lbl, actual_col) in zip(comp_tabs, tab_map.items()):
        with tab:
            df_tab = df_comp[['pred_for','predicted_mean',actual_col]].dropna()
            df_tab = df_tab.rename(columns={actual_col:'actual'})

            if df_tab.empty:
                st.info(t("no_data"))
                continue

            mae  = np.mean(np.abs(df_tab['predicted_mean'] - df_tab['actual']))
            rmse = np.sqrt(np.mean((df_tab['predicted_mean'] - df_tab['actual'])**2))
            bias = np.mean(df_tab['predicted_mean'] - df_tab['actual'])
            denom = np.sum((df_tab['actual'] - df_tab['actual'].mean())**2)
            r2   = 1 - np.sum((df_tab['actual'] - df_tab['predicted_mean'])**2)/(denom+1e-8)

            for col,(lbl,val) in zip(st.columns(4),[
                ("MAE",  f"{mae:.2f} µg/m³"),
                ("RMSE", f"{rmse:.2f} µg/m³"),
                ("R²",   f"{r2:.3f}"),
                ("Bias", f"{bias:+.2f} µg/m³"),
            ]):
                with col:
                    st.markdown(f"""
                    <div class="mbox">
                        <div class="mbox-num">{val}</div>
                        <div class="mbox-lbl">{lbl}</div>
                    </div>
                    """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            pred_lbl   = "Predicted Mean" if st.session_state.lang=="EN" else "Prediksi Mean"
            actual_lbl = f"Actual — {tab_lbl}" if st.session_state.lang=="EN" else f"Aktual — {tab_lbl}"

            fig_line = go.Figure()
            fig_line.add_trace(go.Scatter(
                x=df_tab['pred_for'], y=df_tab['actual'],
                mode='lines+markers', name=actual_lbl,
                line=dict(color='#0f172a', width=2), marker=dict(size=5),
                hovertemplate='%{x|%d %b %H:%M}<br>'+actual_lbl+': %{y:.1f}<extra></extra>'
            ))
            fig_line.add_trace(go.Scatter(
                x=df_tab['pred_for'], y=df_tab['predicted_mean'],
                mode='lines+markers', name=pred_lbl,
                line=dict(color='#3b82f6', width=2, dash='dash'),
                marker=dict(size=5),
                hovertemplate='%{x|%d %b %H:%M}<br>'+pred_lbl+': %{y:.1f}<extra></extra>'
            ))
            fig_line.add_trace(go.Scatter(
                x=pd.concat([df_tab['pred_for'], df_tab['pred_for'][::-1]]),
                y=pd.concat([df_tab['predicted_mean'], df_tab['actual'][::-1]]),
                fill='toself', fillcolor='rgba(59,130,246,0.07)',
                line=dict(color='rgba(0,0,0,0)'),
                showlegend=False, hoverinfo='skip'
            ))
            add_thresholds(fig_line)
            fig_line.update_layout(
                height=280,
                xaxis=dict(tickformat="%H:%M\n%d %b", gridcolor="#f1f5f9"),
                yaxis=dict(title="PM2.5 (µg/m³)", rangemode='tozero', gridcolor="#f1f5f9"),
                legend=dict(orientation="h", y=1.08, x=0),
                **BASE_LAYOUT
            )
            st.plotly_chart(fig_line, use_container_width=True)

            cs1, cs2 = st.columns(2)
            with cs1:
                lim = max(df_tab['actual'].max(), df_tab['predicted_mean'].max())*1.1
                fig_sc = go.Figure()
                fig_sc.add_trace(go.Scatter(
                    x=df_tab['actual'], y=df_tab['predicted_mean'],
                    mode='markers',
                    marker=dict(color='#3b82f6', size=7, opacity=0.7),
                    hovertemplate='Actual: %{x:.1f}<br>Pred: %{y:.1f}<extra></extra>'
                ))
                fig_sc.add_shape(type="line",x0=0,x1=lim,y0=0,y1=lim,
                                  line=dict(color="#ef4444",dash="dot",width=1.5))
                fig_sc.update_layout(
                    height=240,
                    title=dict(text="Scatter: Actual vs Predicted",font=dict(size=11)),
                    xaxis=dict(title="Actual PM2.5",gridcolor="#f1f5f9"),
                    yaxis=dict(title="Predicted",gridcolor="#f1f5f9"),
                    **{k:v for k,v in BASE_LAYOUT.items() if k!='margin'}
                )
                fig_sc.update_layout(margin=dict(l=0,r=10,t=30,b=0))
                st.plotly_chart(fig_sc, use_container_width=True)

            with cs2:
                df_tab = df_tab.copy()
                df_tab['error'] = df_tab['predicted_mean'] - df_tab['actual']
                fig_err = go.Figure(go.Bar(
                    x=df_tab['pred_for'], y=df_tab['error'],
                    marker_color=['#ef4444' if e>0 else '#3b82f6'
                                  for e in df_tab['error']],
                    hovertemplate='%{x|%d %b %H:%M}<br>Error: %{y:.1f}<extra></extra>'
                ))
                fig_err.add_hline(y=0, line_color="#0f172a", line_width=1)
                fig_err.update_layout(
                    height=240,
                    title=dict(text="Prediction Error (Pred − Actual)",font=dict(size=11)),
                    xaxis=dict(tickformat="%d %b",gridcolor="#f1f5f9"),
                    yaxis=dict(title="Error (µg/m³)",gridcolor="#f1f5f9"),
                    **{k:v for k,v in BASE_LAYOUT.items() if k!='margin'}
                )
                fig_err.update_layout(margin=dict(l=0,r=10,t=30,b=0))
                st.plotly_chart(fig_err, use_container_width=True)

            with st.expander(t("raw_data")):
                df_show = df_tab[['pred_for','predicted_mean','actual','error']].copy()
                df_show['pred_for'] = df_show['pred_for'].dt.strftime("%d %b %Y %H:%M SGT")
                df_show['error']    = df_show['error'].round(2)
                df_show.columns     = ["Time","Predicted Mean","Actual","Error"]
                st.dataframe(df_show, use_container_width=True, hide_index=True)

# ============================================================
# S6 — LEGEND
# ============================================================
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown(f'<div class="sec-title">{t("guide_title")}</div>',
            unsafe_allow_html=True)

for col,(key,color) in zip(st.columns(5),[
    ("Good","#22c55e"),("Moderate","#f59e0b"),("Unhealthy","#ef4444"),
    ("Very Bad","#8b5cf6"),("Hazardous","#7f1d1d")
]):
    cat_name,rng,desc = tc(key)
    with col:
        st.markdown(f"""
        <div class="leg-card" style="background:{color}0d;border-top:3px solid {color}">
            <div class="leg-cat" style="color:{color}">{cat_name}</div>
            <div class="leg-range">{rng}</div>
            <div class="leg-desc">{desc}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)
st.markdown(f"""
<div style='text-align:center;color:#94a3b8;font-size:0.75rem;padding:4px 0'>
    <b style="color:#64748b">AirMind SG</b> &nbsp;·&nbsp; {t('footer')}
</div>
""", unsafe_allow_html=True)