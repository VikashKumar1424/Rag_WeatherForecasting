
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent.parent

CHROMA_DIR = BASE_DIR / "data" / "chroma"

# ---------------------------------------------------------
# Application configuration
# ---------------------------------------------------------

DEFAULT_LOCATION = os.getenv(
    "DEFAULT_LOCATION",
    "Bengaluru",
)

GEMINI_MODEL = os.getenv(
    "GEMINI_MODEL",
    "gemini-3.5-flash",
)

# ---------------------------------------------------------
# Google Gemini API key
# ---------------------------------------------------------

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise RuntimeError(
        "GOOGLE_API_KEY is not configured. "
        "Please add GOOGLE_API_KEY to your .env file."
    )

# ---------------------------------------------------------
# Open-Meteo
# ---------------------------------------------------------

OPEN_METEO_GEOCODING_URL = (
    "https://geocoding-api.open-meteo.com/v1/search"
)

OPEN_METEO_FORECAST_URL = (
    "https://api.open-meteo.com/v1/forecast"
)

OPEN_METEO_DOCS_URL = (
    "https://open-meteo.com/en/docs"
)

