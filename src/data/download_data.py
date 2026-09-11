"""
Data ingestion script for CPCB air quality and Open-Meteo / ECMWF ERA5 meteorological records.
"""
import os
import json
import urllib.request
import pandas as pd
from pathlib import Path
from src.utils.logger import get_logger
from src.utils.config import CITY_COORDINATES, DataConfig

logger = get_logger("DataIngestion")

CPCB_URL = "https://raw.githubusercontent.com/GowriShaju/Air-Quality-Prediction-EDA/main/city_day.csv"

def download_cpcb_data(dest_path: str = "data/raw/city_day.csv") -> str:
    """Download CPCB city_day.csv if not already present."""
    dest = Path(dest_path)
    if dest.exists() and dest.stat().st_size > 1000000:
        logger.info(f"CPCB dataset already present at {dest_path} ({dest.stat().st_size / (1024*1024):.2f} MB)")
        return dest_path
    
    dest.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Downloading CPCB dataset from {CPCB_URL} to {dest_path}...")
    req = urllib.request.Request(CPCB_URL, headers={"User-Agent": "Mozilla/5.0 (AQI Research)"})
    with urllib.request.urlopen(req, timeout=30) as resp, open(dest_path, "wb") as f:
        f.write(resp.read())
    logger.info("CPCB download complete.")
    return dest_path

def fetch_weather_data(city: str = "Delhi",
                       start_date: str = "2015-01-01",
                       end_date: str = "2020-07-01",
                       dest_path: str = "data/raw/weather_daily.csv") -> str:
    """Fetch daily ERA5 reanalysis weather data via Open-Meteo Historical Archive API."""
    dest = Path(dest_path)
    if dest.exists() and dest.stat().st_size > 10000:
        logger.info(f"Weather dataset already present at {dest_path}")
        return dest_path
    
    coords = CITY_COORDINATES.get(city)
    if not coords:
        raise ValueError(f"City '{city}' coordinates not configured. Available: {list(CITY_COORDINATES.keys())}")
    
    lat, lon = coords["latitude"], coords["longitude"]
    url = (
        f"https://archive-api.open-meteo.com/v1/archive?"
        f"latitude={lat}&longitude={lon}&start_date={start_date}&end_date={end_date}&"
        f"daily=temperature_2m_mean,relative_humidity_2m_mean,wind_speed_10m_mean&timezone=Asia%2FKolkata"
    )
    logger.info(f"Fetching ERA5 reanalysis weather for {city} ({lat}, {lon}) from {start_date} to {end_date}...")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (AQI Research)"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    
    df_weather = pd.DataFrame(data["daily"])
    df_weather.rename(columns={
        "time": "Date",
        "temperature_2m_mean": "Temperature",
        "relative_humidity_2m_mean": "Humidity",
        "wind_speed_10m_mean": "Wind_Speed"
    }, inplace=True)
    
    dest.parent.mkdir(parents=True, exist_ok=True)
    df_weather.to_csv(dest_path, index=False)
    logger.info(f"Weather data saved to {dest_path} with {len(df_weather)} records.")
    return dest_path

if __name__ == "__main__":
    download_cpcb_data()
    fetch_weather_data()
