
from weather_rag.weather_codes import weather_description


def test_clear_sky():
    assert weather_description(0) == "Clear sky"


def test_unknown_code():
    assert weather_description(999) == "Weather code 999"
