from __future__ import annotations

from typing import Any

import requests
from bs4 import BeautifulSoup

from .config import (
    OPEN_METEO_DOCS_URL,
    OPEN_METEO_FORECAST_URL,
    OPEN_METEO_GEOCODING_URL,
)

HEADERS = {
    "User-Agent": (
        "WeatherRAGMonitor/1.0 "
        "(+https://open-meteo.com/)"
    )
}


class OpenMeteoScraper:
    """HTTP client for Open-Meteo geocoding, forecast, and docs."""

    def __init__(self, timeout: int = 20) -> None:
        self.timeout = timeout

    def geocode(
        self,
        location: str,
        country: str | None = None,
    ) -> dict[str, Any]:
        """Resolve a location name to Open-Meteo coordinates."""

        params: dict[str, Any] = {
            "name": location,
            "count": 5,
            "language": "en",
            "format": "json",
        }

        if country:
            params["country"] = country

        response = requests.get(
            OPEN_METEO_GEOCODING_URL,
            params=params,
            headers=HEADERS,
            timeout=self.timeout,
        )

        response.raise_for_status()

        results = response.json().get("results", [])

        if not results:
            raise ValueError(
                f"Location not found: {location}"
            )

        # Prefer exact name match.
        selected = next(
            (
                item
                for item in results
                if str(item.get("name", "")).lower()
                == location.lower()
            ),
            results[0],
        )

        return selected

    def fetch_forecast(
        self,
        latitude: float,
        longitude: float,
        timezone: str = "auto",
        forecast_days: int = 7,
    ) -> dict[str, Any]:
        """
        Fetch hourly and daily weather forecast.

        Includes current weather, hourly weather and
        daily forecast data.
        """

        forecast_days = max(
            1,
            min(int(forecast_days), 16),
        )

        hourly_fields = [
            "temperature_2m",
            "relative_humidity_2m",
            "apparent_temperature",
            "precipitation_probability",
            "precipitation",
            "rain",
            "weather_code",
            "cloud_cover",
            "wind_speed_10m",
            "wind_direction_10m",
        ]

        daily_fields = [
            "weather_code",
            "temperature_2m_max",
            "temperature_2m_min",
            "apparent_temperature_max",
            "apparent_temperature_min",
            "precipitation_probability_max",
            "precipitation_sum",
            "rain_sum",
            "wind_speed_10m_max",
            "wind_gusts_10m_max",
        ]

        current_fields = [
            "temperature_2m",
            "relative_humidity_2m",
            "apparent_temperature",
            "precipitation",
            "rain",
            "weather_code",
            "cloud_cover",
            "wind_speed_10m",
            "wind_direction_10m",
        ]

        params = {
            "latitude": latitude,
            "longitude": longitude,
            "timezone": timezone,
            "forecast_days": forecast_days,
            "current": ",".join(current_fields),
            "hourly": ",".join(hourly_fields),
            "daily": ",".join(daily_fields),
        }

        response = requests.get(
            OPEN_METEO_FORECAST_URL,
            params=params,
            headers=HEADERS,
            timeout=self.timeout,
        )

        response.raise_for_status()

        return response.json()

    def scrape_docs_metadata(self) -> dict[str, str]:
        """Scrape Open-Meteo docs metadata."""

        response = requests.get(
            OPEN_METEO_DOCS_URL,
            headers=HEADERS,
            timeout=self.timeout,
        )

        response.raise_for_status()

        soup = BeautifulSoup(
            response.text,
            "html.parser",
        )

        title = (
            soup.title.get_text(" ", strip=True)
            if soup.title
            else "Open-Meteo Docs"
        )

        description = ""

        meta = soup.find(
            "meta",
            attrs={"name": "description"},
        )

        if meta:
            description = meta.get(
                "content",
                "",
            ).strip()

        return {
            "title": title,
            "description": description,
            "source_url": OPEN_METEO_DOCS_URL,
        }