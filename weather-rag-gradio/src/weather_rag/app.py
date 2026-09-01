
from __future__ import annotations

from typing import Any

import gradio as gr

from .config import DEFAULT_LOCATION
from .rag import (
    answer_question,
    build_vector_store,
    weather_documents,
    load_vector_store,
)
from .scraper import OpenMeteoScraper
from .weather_codes import weather_description


# =========================================================
# UI CONFIGURATION
# =========================================================

CSS = """
.gradio-container {
    max-width: 1180px !important;
    margin: auto !important;
    background:
        radial-gradient(
            circle at 10% 10%,
            rgba(56, 189, 248, .12),
            transparent 30%
        ),
        radial-gradient(
            circle at 90% 0%,
            rgba(37, 99, 235, .10),
            transparent 32%
        ),
        #f8fafc;
}

/* Hero */
.hero {
    padding: 24px 28px;
    border-radius: 20px;
    background: linear-gradient(
        135deg,
        #0f172a,
        #1e3a8a
    );
    color: white;
    margin-bottom: 14px;
    box-shadow: 0 12px 35px rgba(15, 23, 42, .14);
}

.hero h1 {
    margin: 0 0 6px 0;
    font-size: 30px;
}

.hero p {
    margin: 0;
    opacity: .85;
    font-size: 14px;
}

/* Weather location header */
.location-header {
    padding: 14px 18px;
    border-radius: 16px;
    background: white;
    border: 1px solid #e2e8f0;
    margin-bottom: 10px;
}

.location-name {
    font-size: 22px;
    font-weight: 700;
    color: #0f172a;
}

.location-source {
    color: #64748b;
    font-size: 12px;
    margin-top: 3px;
}

/* Weather cards */
.weather-card {
    background: white;
    border: 1px solid #e2e8f0;
    border-radius: 16px;
    padding: 14px 16px;
    min-height: 95px;
    box-shadow: 0 4px 14px rgba(15, 23, 42, .05);
}

.weather-card-title {
    font-size: 13px;
    color: #64748b;
    margin-bottom: 7px;
}

.weather-card-value {
    font-size: 25px;
    font-weight: 700;
    color: #0f172a;
}

.weather-card-detail {
    font-size: 12px;
    color: #64748b;
    margin-top: 3px;
}

/* Section */
.section-title {
    font-size: 17px;
    font-weight: 700;
    color: #0f172a;
    margin: 12px 0 8px 0;
}

/* Status */
.status {
    border-radius: 12px !important;
}

/* Chat */
.chat-container {
    margin-top: 10px;
}

/* Reduce unnecessary Gradio spacing */
.gradio-container .gap {
    gap: 10px !important;
}

footer {
    display: none !important;
}
"""


THEME = gr.themes.Soft(
    primary_hue="blue",
    secondary_hue="sky",
    neutral_hue="slate",
)


WELCOME = """
### 🌤️ India Weather RAG

Enter **any Indian city, district, or state** and load the latest
forecast.

You can then ask questions such as:

- Will it rain tomorrow?
- What is today's temperature?
- What will the temperature be over the next 3 days?
- Should I carry an umbrella?
- Summarize the wind conditions.

> For precise weather, enter a city or district. A state name is
> resolved by Open-Meteo to a representative location.
"""


# =========================================================
# HELPERS
# =========================================================

def _safe_value(
    data: dict[str, Any],
    key: str,
    index: int = 0,
    default: float = 0.0,
) -> float:
    """Safely extract an indexed numeric value."""

    values = data.get(key)

    if not isinstance(values, list):
        return default

    if index >= len(values):
        return default

    value = values[index]

    if value is None:
        return default

    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _current_hour_index(
    hourly: dict[str, Any],
) -> int:
    """
    Find the forecast record closest to the current local hour.

    Open-Meteo returns timestamps in the requested timezone.
    For a simple dashboard, the first available record is used
    when an exact match cannot be determined.
    """

    times = hourly.get("time", [])

    if not times:
        return 0

    # Open-Meteo normally returns hourly data beginning at
    # the current hour when forecast data is requested.
    return 0


def _weather_summary_html(
    location_name: str,
    state: str,
) -> str:

    if state:
        label = f"{location_name}, {state}, India"
    else:
        label = f"{location_name}, India"

    return f"""
    <div class="location-header">
        <div class="location-name">📍 {label}</div>
        <div class="location-source">
            Live forecast data • Open-Meteo
        </div>
    </div>
    """


# =========================================================
# WEATHER + RAG
# =========================================================

def resolve_and_index(
    location: str,
    days: int,
) -> tuple[
    str,
    str,
    str,
    str,
    str,
    str,
    str,
    str,
    str,
    str,
]:
    """
    Resolve location, fetch Open-Meteo weather data,
    update the RAG vector store and return UI values.

    Returns:

        status
        resolved location
        location header
        temperature
        feels like
        humidity
        rain
        precipitation
        wind
        min/max temperature
    """

    location = (location or "").strip()

    if not location:
        return (
            "❌ Please enter an Indian city, district, or state.",
            "",
            "",
            "—",
            "—",
            "—",
            "—",
            "—",
            "—",
            "—",
        )

    scraper = OpenMeteoScraper()

    # -----------------------------------------------------
    # Geocoding
    # -----------------------------------------------------

    try:
        geo = scraper.geocode(
            location,
            "India",
        )
    except Exception as exc:
        return (
            (
                f"❌ Could not find **{location}** in India.\n\n"
                f"`{exc}`"
            ),
            "",
            "",
            "—",
            "—",
            "—",
            "—",
            "—",
            "—",
            "—",
        )

    # -----------------------------------------------------
    # Validate country
    # -----------------------------------------------------

    if str(
        geo.get("country", "")
    ).lower() != "india":

        return (
            (
                f"❌ **{location}** did not resolve to "
                "an Indian location."
            ),
            "",
            "",
            "—",
            "—",
            "—",
            "—",
            "—",
            "—",
            "—",
        )

    # -----------------------------------------------------
    # Coordinates
    # -----------------------------------------------------

    try:
        latitude = float(
            geo["latitude"]
        )

        longitude = float(
            geo["longitude"]
        )

    except (
        KeyError,
        TypeError,
        ValueError,
    ) as exc:

        return (
            (
                "❌ Invalid geocoding response "
                "from Open-Meteo.\n\n"
                f"`{exc}`"
            ),
            "",
            "",
            "—",
            "—",
            "—",
            "—",
            "—",
            "—",
            "—",
        )

    resolved_name = geo.get(
        "name",
        location,
    )

    state = geo.get(
        "admin1",
        "",
    )

    timezone = geo.get(
        "timezone",
        "auto",
    )

    # -----------------------------------------------------
    # Fetch forecast
    # -----------------------------------------------------

    try:

        forecast = scraper.fetch_forecast(
            latitude,
            longitude,
            timezone,
            int(days),
        )

        # -------------------------------------------------
        # Build RAG documents
        # -------------------------------------------------

        documents = weather_documents(
            forecast,
            resolved_name,
            latitude,
            longitude,
        )

        # -------------------------------------------------
        # Build vector store
        # -------------------------------------------------

        build_vector_store(
            documents
        )

    except Exception as exc:

        return (
            (
                f"❌ Weather data could not be loaded "
                f"for **{resolved_name}**.\n\n"
                f"`{exc}`"
            ),
            "",
            "",
            "—",
            "—",
            "—",
            "—",
            "—",
            "—",
            "—",
        )

    # =====================================================
    # Extract current weather
    # =====================================================

    hourly = forecast.get(
        "hourly",
        {},
    )

    daily = forecast.get(
        "daily",
        {},
    )

    index = _current_hour_index(
        hourly
    )

    temperature = _safe_value(
        hourly,
        "temperature_2m",
        index,
    )

    apparent_temperature = _safe_value(
        hourly,
        "apparent_temperature",
        index,
    )

    humidity = _safe_value(
        hourly,
        "relative_humidity_2m",
        index,
    )

    rain = _safe_value(
        hourly,
        "rain",
        index,
    )

    precipitation = _safe_value(
        hourly,
        "precipitation",
        index,
    )

    wind_speed = _safe_value(
        hourly,
        "wind_speed_10m",
        index,
    )

    # -----------------------------------------------------
    # Daily min/max
    # -----------------------------------------------------

    daily_max = _safe_value(
        daily,
        "temperature_2m_max",
        0,
    )

    daily_min = _safe_value(
        daily,
        "temperature_2m_min",
        0,
    )

    # -----------------------------------------------------
    # Weather condition
    # -----------------------------------------------------

    weather_code = _safe_value(
        hourly,
        "weather_code",
        index,
    )

    condition = weather_description(
        weather_code
    )

    # -----------------------------------------------------
    # Location label
    # -----------------------------------------------------

    if state:
        label = (
            f"{resolved_name}, "
            f"{state}, India"
        )
    else:
        label = (
            f"{resolved_name}, India"
        )

    location_html = _weather_summary_html(
        resolved_name,
        state,
    )

    # -----------------------------------------------------
    # UI values
    # -----------------------------------------------------

    temperature_value = (
        f"{temperature:.1f} °C"
    )

    feels_like_value = (
        f"{apparent_temperature:.1f} °C"
    )

    humidity_value = (
        f"{humidity:.0f}%"
    )

    rain_value = (
        f"{rain:.1f} mm"
    )

    precipitation_value = (
        f"{precipitation:.1f} mm"
    )

    wind_value = (
        f"{wind_speed:.1f} km/h"
    )

    temperature_range = (
        f"Min {daily_min:.1f} °C • "
        f"Max {daily_max:.1f} °C"
    )

    status = (
        f"✅ **{label}** is ready.\n\n"
        f"Current condition: **{condition}** • "
        f"Indexed **{len(documents)} hourly records** "
        f"from Open-Meteo."
    )

    return (
        status,
        resolved_name,
        location_html,
        temperature_value,
        feels_like_value,
        humidity_value,
        rain_value,
        precipitation_value,
        wind_value,
        temperature_range,
    )


# =========================================================
# REFRESH
# =========================================================

def refresh_index(
    location: str,
    days: int,
):
    """
    Refresh Open-Meteo weather data and RAG index.
    """

    return resolve_and_index(
        location,
        days,
    )


# =========================================================
# CHAT
# =========================================================

def chat(
    message: str,
    history: list,
    location: str,
    days: int,
    current_location: str,
) -> str:
    """
    Gradio ChatInterface callback.

    The response is always normalized to a string by
    answer_question().
    """

    del history

    requested = (
        location or ""
    ).strip()

    resolved = (
        current_location or ""
    ).strip()

    if not requested:
        return (
            "❌ Please enter an Indian city, district, "
            "or state first."
        )

    try:

        # -------------------------------------------------
        # Automatically rebuild when location changes
        # -------------------------------------------------

        if (
            not resolved
            or requested.lower()
            != resolved.lower()
        ):

            result = resolve_and_index(
                requested,
                days,
            )

            status = result[0]
            resolved = result[1]

            if status.startswith("❌"):
                return status

        # -------------------------------------------------
        # Load location-specific Chroma store
        # -------------------------------------------------

        store = load_vector_store(
            resolved
        )

        # -------------------------------------------------
        # Ask RAG
        # -------------------------------------------------

        response = answer_question(
            message,
            store,
            resolved,
        )

        # -------------------------------------------------
        # Always return plain text
        # -------------------------------------------------

        if isinstance(response, str):
            return response

        return str(response)

    except Exception as exc:

        return (
            f"⚠️ Unable to answer for "
            f"**{resolved or requested}**.\n\n"
            "Refresh the location and try again.\n\n"
            f"`{exc}`"
        )


# =========================================================
# GRADIO APPLICATION
# =========================================================

with gr.Blocks(
    title="India Weather RAG Monitor",
) as demo:

    # =====================================================
    # HERO
    # =====================================================

    gr.HTML(
        """
        <div class="hero">
            <h1>🌦️ India Weather RAG Monitor</h1>
            <p>
                Dynamic Indian locations • Open-Meteo •
                Gemini • LangChain • Chroma
            </p>
        </div>
        """
    )

    # =====================================================
    # STATE
    # =====================================================

    current_location = gr.State("")

    # =====================================================
    # LOCATION + CONTROLS
    # =====================================================

    with gr.Row():

        with gr.Column(
            scale=1,
            min_width=260,
        ):

            gr.Markdown(
                "### 🇮🇳 Location"
            )

            location = gr.Textbox(
                label="Indian City / District / State",
                value=(
                    DEFAULT_LOCATION
                    if DEFAULT_LOCATION
                    else ""
                ),
                placeholder=(
                    "e.g. Delhi, Mumbai, "
                    "Bengaluru, Patna"
                ),
            )

            gr.Markdown(
                "Country is restricted to **India**."
            )

            days = gr.Slider(
                minimum=1,
                maximum=16,
                value=7,
                step=1,
                label="Forecast days",
            )

            refresh = gr.Button(
                "🔄 Load Weather",
                variant="primary",
            )

            status = gr.Markdown(
                "Enter a location and click "
                "**Load Weather**.",
                elem_classes="status",
            )

        # =================================================
        # CURRENT WEATHER
        # =================================================

        with gr.Column(
            scale=3,
        ):

            location_header = gr.HTML(
                _weather_summary_html(
                    DEFAULT_LOCATION
                    or "India",
                    "",
                )
            )

            gr.Markdown(
                "### 🌡️ Current Weather",
                elem_classes="section-title",
            )

            with gr.Row():

                temperature = gr.HTML(
                    """
                    <div class="weather-card">
                        <div class="weather-card-title">
                            🌡️ Temperature
                        </div>
                        <div class="weather-card-value">
                            —
                        </div>
                        <div class="weather-card-detail">
                            Current temperature
                        </div>
                    </div>
                    """
                )

                feels_like = gr.HTML(
                    """
                    <div class="weather-card">
                        <div class="weather-card-title">
                            🌤️ Feels Like
                        </div>
                        <div class="weather-card-value">
                            —
                        </div>
                        <div class="weather-card-detail">
                            Apparent temperature
                        </div>
                    </div>
                    """
                )

                humidity = gr.HTML(
                    """
                    <div class="weather-card">
                        <div class="weather-card-title">
                            💧 Humidity
                        </div>
                        <div class="weather-card-value">
                            —
                        </div>
                        <div class="weather-card-detail">
                            Relative humidity
                        </div>
                    </div>
                    """
                )

            gr.Markdown(
                "### 🌧️ Rain & Wind",
                elem_classes="section-title",
            )

            with gr.Row():

                rain = gr.HTML(
                    """
                    <div class="weather-card">
                        <div class="weather-card-title">
                            🌧️ Rain
                        </div>
                        <div class="weather-card-value">
                            —
                        </div>
                        <div class="weather-card-detail">
                            Current hourly rain
                        </div>
                    </div>
                    """
                )

                precipitation = gr.HTML(
                    """
                    <div class="weather-card">
                        <div class="weather-card-title">
                            ☔ Precipitation
                        </div>
                        <div class="weather-card-value">
                            —
                        </div>
                        <div class="weather-card-detail">
                            Current hourly precipitation
                        </div>
                    </div>
                    """
                )

                wind = gr.HTML(
                    """
                    <div class="weather-card">
                        <div class="weather-card-title">
                            💨 Wind
                        </div>
                        <div class="weather-card-value">
                            —
                        </div>
                        <div class="weather-card-detail">
                            Wind speed
                        </div>
                    </div>
                    """
                )

            temperature_range = gr.HTML(
                """
                <div class="weather-card">
                    <div class="weather-card-title">
                        📊 Today's Temperature Range
                    </div>
                    <div class="weather-card-value">
                        —
                    </div>
                </div>
                """
            )

    # =====================================================
    # RAG CHAT
    # =====================================================

    gr.Markdown(
        "### 💬 Ask Weather Questions",
        elem_classes="section-title",
    )

    gr.Markdown(
        WELCOME
    )

    gr.ChatInterface(
        fn=chat,
        additional_inputs=[
            location,
            days,
            current_location,
        ],
    )

    # =====================================================
    # REFRESH EVENT
    # =====================================================

    refresh.click(
        fn=refresh_index,
        inputs=[
            location,
            days,
        ],
        outputs=[
            status,
            current_location,
            location_header,
            temperature,
            feels_like,
            humidity,
            rain,
            precipitation,
            wind,
            temperature_range,
        ],
    )


# =========================================================
# APPLICATION ENTRY POINT
# =========================================================

def main() -> None:
    """
    Start the Gradio application.
    """

    demo.launch(
        theme=THEME,
        css=CSS,
    )


if __name__ == "__main__":
    main()
