
from weather_rag.rag import weather_documents


def test_weather_documents():
    forecast = {
        "hourly": {
            "time": ["2026-09-01T00:00"],
            "temperature_2m": [25],
            "relative_humidity_2m": [80],
            "apparent_temperature": [27],
            "precipitation": [0],
            "rain": [0],
            "weather_code": [0],
            "cloud_cover": [10],
            "wind_speed_10m": [5],
            "wind_direction_10m": [180],
        }
    }

    docs = weather_documents(forecast, "Bettiah", 26.8, 84.5)

    assert len(docs) == 1
    assert "Bettiah" in docs[0].page_content
    assert "Clear sky" in docs[0].page_content
