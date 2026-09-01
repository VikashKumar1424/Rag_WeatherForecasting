
# 🌦️ Weather RAG Monitor

A production-style RAG weather monitoring application built with:

- Python 3.11+
- `uv`
- Open-Meteo
- Requests + BeautifulSoup
- LangChain
- Chroma
- Gemini 3.5 Flash
- Gemini Embedding 2
- Gradio
- Pytest
- Jupyter
- Cursor

## Architecture

```text
Open-Meteo
   │
   ├── Geocoding API ────────┐
   │                         │
   ├── Forecast API ─────────┤
   │                         ▼
   └── Documentation HTML → Weather Scraper
                              │
                              ▼
                    Normalized Weather Documents
                              │
                              ▼
                       Text Splitter
                              │
                              ▼
                    Gemini Embedding 2
                              │
                              ▼
                           Chroma
                              │
User ── Gradio Chat ──► Retriever ──► Gemini 3.5 Flash
                              │
                              └──► grounded weather answer
```

## Important implementation note

Open-Meteo exposes weather information through APIs. Therefore the application uses `Requests` to retrieve the actual forecast JSON and `BeautifulSoup` to scrape the Open-Meteo documentation page for source/context metadata. This is more reliable than trying to scrape dynamic forecast values from the HTML page itself.

The Forecast API supports hourly forecast data and up to 16 forecast days. See the official documentation:
https://open-meteo.com/en/docs

## 1. Install

```bash
uv sync
```

## 2. Configure Gemini

```bash
cp .env.example .env
```

Put your Google AI Studio API key in `.env`:

```env
GOOGLE_API_KEY=YOUR_KEY
```

## 3. Build the RAG index

```bash
uv run python -m weather_rag.ingest --location "Mumbai" --country "India"
```

You can also specify coordinates:

```bash
uv run python -m weather_rag.ingest --latitude 26.8025 --longitude 84.5030 --location "Bettiah" --country "India"
```

## 4. Start Gradio

```bash
uv run python -m weather_rag.app
```

Then open the local Gradio URL shown in the terminal.

## 5. Run tests

```bash
uv run pytest
```

## 6. Open the notebook

```bash
uv run jupyter notebook notebooks/weather_rag_demo.ipynb
```

## Dynamic Indian locations

The UI is **not tied to Bettiah**. Users can enter any Indian city, district, or state name. The application sends the location to Open-Meteo Geocoding with `country=India`, validates that the resolved result is in India, fetches the forecast coordinates, and creates a location-specific Chroma collection.

Examples:

```text
Mumbai
Bengaluru
Patna
Kolkata
Jaipur
Hyderabad
Chennai
Kerala
Maharashtra
West Bengal
```

For a state name, weather is necessarily represented by the geocoded location returned by Open-Meteo. For accurate local weather, prefer a city or district.

The chat also automatically detects when the location textbox has changed and refreshes the correct location index before answering.

## Example questions

- What is the weather in Bettiah today?
- Will it rain tomorrow?
- What will the temperature be over the next 3 days?
- What are the expected wind conditions?
- Give me a concise weather summary for today.
- Should I carry an umbrella tomorrow?
- Compare today's and tomorrow's temperatures.

## RAG design

Each forecast record is converted into a LangChain `Document` containing:

- location
- timestamp
- temperature
- apparent temperature
- humidity
- precipitation
- rain
- weather code
- cloud cover
- wind speed
- wind direction
- source

The records are chunked, embedded with Gemini Embedding 2, and stored in Chroma.

At query time:

1. User asks a weather question.
2. The question is embedded.
3. Chroma retrieves the most relevant weather records.
4. Retrieved records are passed to Gemini 3.5 Flash.
5. Gemini answers only from retrieved weather context and clearly states when data is insufficient.

## Cursor

Recommended Cursor workflow:

```text
1. Open the project folder.
2. Run: uv sync
3. Copy .env.example -> .env
4. Add GOOGLE_API_KEY
5. Run ingestion.
6. Run Gradio.
7. Use tests before committing.
```
