# 🌦️ RAG Weather Forecasting

A production-ready Retrieval-Augmented Generation (RAG) weather monitoring application that combines real-time weather data with AI-powered natural language understanding.

## 📋 Overview

This project implements a sophisticated weather forecasting system using:

- **Data Source**: Open-Meteo Weather API (free, no authentication required)
- **RAG Framework**: LangChain with Chroma vector database
- **AI Models**: Google Gemini 3.5 Flash (generation) and Gemini Embedding 2 (embeddings)
- **Frontend**: Gradio interactive web interface
- **Backend**: Python 3.11+ with `uv` package manager
- **Testing**: Pytest suite included

## 🎯 Key Features

✨ **Natural Language Queries** - Ask weather questions in plain English  
📍 **Location-Aware** - Supports any Indian city, district, or state  
🔍 **RAG-Powered** - Retrieves relevant weather context before answering  
🚀 **Production-Ready** - Structured codebase with tests and dependency management  
💻 **Interactive UI** - Chat interface built with Gradio  
📊 **Multi-day Forecasts** - Up to 16 days of hourly forecast data  

## 📂 Project Structure

```
.
├── README.md (this file)
└── weather-rag-gradio/           # Main application package
    ├── README.md                 # Detailed setup & usage guide
    ├── pyproject.toml            # Project metadata & dependencies
    ├── uv.lock                   # Locked dependency versions
    ├── src/weather_rag/          # Application source code
    ├── data/                     # Weather data & vector stores
    ├── notebooks/                # Jupyter notebooks & demos
    └── tests/                    # Pytest test suite
```

## 🏗️ System Architecture

### High-Level Data Flow

```
┌─────────────────────────────────────────────────────────────┐
│                    RAG WEATHER SYSTEM                       │
└─────────────────────────────────────────────────────────────┘

                    ┌─────────────────┐
                    │  Open-Meteo API │
                    └────────┬────────┘
                             │
                ┌────────────┼────────────┐
                │            │            │
         ┌──────▼──┐  ┌──────▼──┐  ┌─────▼──┐
         │ Geocode │  │Forecast │  │ Docs   │
         │   API   │  │   API   │  │Scraper │
         └──────┬──┘  └──────┬──┘  └─────┬──┘
                │            │            │
                └────────────┼────────────┘
                             │
                    ┌────────▼────────┐
                    │  Data Processing│
                    │   & Cleaning    │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │  Text Splitter  │
                    │  (LangChain)    │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │ Gemini Embed 2  │
                    │  (Embeddings)   │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │  Chroma Vector  │
                    │   Database      │
                    └────────┬────────┘
                             │
                ┌────────────┼────────────┐
                │            │            │
         ┌──────▼──┐  ┌──────▼──┐  ┌─────▼──┐
         │  Gradio │  │ API     │  │Jupyter │
         │   UI    │  │Endpoint │  │  Demo  │
         └──────┬──┘  └──────┬──┘  └─────┬──┘
                │            │            │
                └────────────┼────────────┘
                             │
                    ┌────────▼────────┐
                    │  Retriever      │
                    │  (Semantic)     │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │ Gemini 3.5 Flash│
                    │   (Generator)   │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │ Grounded Answer │
                    │ (RAG Response)  │
                    └─────────────────┘
```

### Detailed Component Interactions

#### 1. **Data Ingestion Pipeline**

```
Ingestion Phase
───────────────────────────────────────────────────

Location Input (User)
      │
      ▼
┌─────────────────────┐
│  Geocoding Service  │ ← Open-Meteo Geocoding API
│  (Validate & Locate)│
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│ Forecast Retrieval  │ ← Open-Meteo Forecast API
│ (16-day hourly data)│   • Temperature
└────────┬────────────┘   • Humidity
         │                • Precipitation
         ▼                • Wind Speed
┌─────────────────────┐   • Cloud Cover
│ Scrape Documentation│ ← Open-Meteo Docs HTML
│ (Weather metadata)  │
└────────┬────────────┘
         │
         ▼
┌─────────────────────────────────┐
│ Normalize & Create Documents    │
│ (LangChain Document objects)    │
│ • location, timestamp           │
│ • temp, humidity, precipitation │
│ • source metadata               │
└────────┬─────────────────────────┘
         │
         ▼
┌─────────────────────────────────┐
│ Chunk Documents                 │
│ (RecursiveCharacterTextSplitter)│
└────────┬─────────────────────────┘
         │
         ▼
┌─────────────────────────────────┐
│ Generate Embeddings             │
│ (Gemini Embedding 2 Model)      │
└────────┬─────────────────────────┘
         │
         ▼
┌─────────────────────────────────┐
│ Store in Vector Database        │
│ (Chroma DB - Persistent)        │
└─────────────────────────────────┘
```

#### 2. **Query & Response Pipeline**

```
Query Phase
───────────────────────────────────────────────────

User Question
(Natural Language)
      │
      ▼
┌──────────────────────┐
│ Gradio Chat Input    │
│ + Location Context   │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────────────┐
│ Refresh RAG Index (Optional) │
│ (If location changed)        │
└──────────┬───────────────────┘
           │
           ▼
┌──────────────────────────────┐
│ Embed Query                  │
│ (Gemini Embedding 2)         │
└──────────┬───────────────────┘
           │
           ▼
┌──────────────────────────────┐
│ Vector Similarity Search     │
│ (Retrieve Top-K Docs)        │
│ (Chroma)                     │
└──────────┬───────────────────┘
           │
           ▼
┌──────────────────────────────┐
│ Build Context                │
│ (Retrieved Weather Data)     │
└──────────┬───────────────────┘
           │
           ▼
┌──────────────────────────────┐
│ Prompt Construction          │
│ • System prompt              │
│ • Context (weather data)     │
│ • User question              │
└──────────┬───────────────────┘
           │
           ▼
┌──────────────────────────────┐
│ Gemini 3.5 Flash LLM         │
│ (Generate Response)          │
│ • Answers from context only  │
│ • Cites data when uncertain  │
└──────────┬───────────────────┘
           │
           ▼
┌──────────────────────────────┐
│ Grounded Answer              │
│ (Based on RAG context)       │
└──────────┬───────────────────┘
           │
           ▼
┌──────────────────────────────┐
│ Display in Gradio UI         │
│ (Streaming output)           │
└──────────────────────────────┘
```

#### 3. **System Components & Technologies**

```
┌─────────────────────────────────────────────────────────────┐
│                      FRONTEND LAYER                         │
├─────────────────────────────────────────────────────────────┤
│  • Gradio Web UI (Chat Interface)                          │
│  • Jupyter Notebooks (Demo & Exploration)                  │
└─────────────────────────────────────────────────────────────┘
                            ▲
                            │
┌─────────────────────────────────────────────────────────────┐
│                   APPLICATION LAYER                         │
├─────────────────────────────────────────────────────────────┤
│  • Query Handler                                           │
│  • RAG Orchestrator                                        │
│  • Location Validator                                      │
│  • Chat Memory Management                                  │
└─────────────────────────────────────────────────────────────┘
                            ▲
                            │
┌─────────────────────────────────────────────────────────────┐
│                    AI/ML LAYER                              │
├─────────────────────────────────────────────────────────────┤
│  • LangChain Framework                                     │
│  • Gemini 3.5 Flash (Generation)                          │
│  • Gemini Embedding 2 (Semantic Embedding)                │
│  • Chroma Vector DB (Persistent Storage)                  │
└─────────────────────────────────────────────────────────────┘
                            ▲
                            │
┌─────────────────────────────────────────────────────────────┐
│                    DATA ACCESS LAYER                        │
├─────────────────────────────────────────────────────────────┤
│  • Open-Meteo Geocoding API                                │
│  • Open-Meteo Forecast API                                 │
│  • HTML Scraper (Documentation)                            │
│  • Local File System (Data Cache)                          │
└─────────────────────────────────────────────────────────────┘
```

## 🔄 RAG (Retrieval-Augmented Generation) Workflow

```
1. RETRIEVAL PHASE
   ┌─────────────────────────────────┐
   │ User asks: "What's the weather  │
   │ in Mumbai tomorrow?"            │
   └────────────┬────────────────────┘
                │
                ▼
   ┌─────────────────────────────────┐
   │ Query Embedding                 │
   │ (Convert to vector)             │
   └────────────┬────────────────────┘
                │
                ▼
   ┌─────────────────────────────────┐
   │ Search Vector DB                │
   │ (Find similar weather docs)     │
   └────────────┬────────────────────┘
                │
                ▼
   ┌─────────────────────────────────┐
   │ Retrieved Context:              │
   │ • Tomorrow's forecast for Mumbai│
   │ • Temperature: 32°C             │
   │ • Humidity: 75%                 │
   │ • Precipitation: 5mm            │
   └─────────────────────────────────┘

2. GENERATION PHASE
   ┌─────────────────────────────────┐
   │ Build Prompt:                   │
   │ [System] + [Context] + [Query]  │
   └────────────┬────────────────────┘
                │
                ▼
   ┌─────────────────────────────────┐
   │ Gemini 3.5 Flash LLM            │
   │ Generates response grounded in  │
   │ retrieved context               │
   └────────────┬────────────────────┘
                │
                ▼
   ┌─────────────────────────────────┐
   │ Answer:                         │
   │ "Tomorrow in Mumbai will be     │
   │ warm (32°C) with 75% humidity   │
   │ and light rain (5mm). Consider  │
   │ carrying an umbrella."          │
   └─────────────────────────────────┘
```

## 🚀 Quick Start

### Prerequisites
- Python 3.11 or higher
- Google API key from [Google AI Studio](https://aistudio.google.com/apikey)

### Installation

```bash
cd weather-rag-gradio
uv sync
cp .env.example .env
# Add your GOOGLE_API_KEY to .env
```

### Build RAG Index

```bash
uv run python -m weather_rag.ingest --location "Mumbai" --country "India"
```

### Launch Application

```bash
uv run python -m weather_rag.app
```

Open the Gradio URL displayed in your terminal.

## 📖 Documentation

- **[Detailed Setup Guide](weather-rag-gradio/README.md)** - Complete installation and configuration instructions
- **Example Queries** - Weather questions the system can answer
- **Configuration** - Customize locations, models, and parameters

## 🧪 Testing

```bash
cd weather-rag-gradio
uv run pytest
```

## 📓 Demo Notebook

Explore the system interactively:

```bash
cd weather-rag-gradio
uv run jupyter notebook notebooks/weather_rag_demo.ipynb
```

## 🛠️ Technology Stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3.11+ |
| Package Manager | uv |
| Weather Data | Open-Meteo API |
| LLM | Google Gemini 3.5 Flash |
| Embeddings | Google Gemini Embedding 2 |
| Vector DB | Chroma |
| RAG Framework | LangChain |
| Frontend | Gradio |
| Testing | Pytest |
| Notebooks | Jupyter |

## 🌍 Supported Locations

The application works with any Indian city, district, or state including:

- **Major Cities**: Mumbai, Delhi, Bengaluru, Hyderabad, Chennai, Kolkata, Pune, Jaipur
- **States**: Maharashtra, Karnataka, Tamil Nadu, Telangana, West Bengal, Kerala, etc.
- **Districts**: Bettiah, Patna, and 700+ other districts

## 💡 Example Queries

- "What is the weather in Bettiah today?"
- "Will it rain tomorrow in Mumbai?"
- "Compare today's and tomorrow's temperatures in Bengaluru"
- "What are the expected wind conditions?"
- "Should I carry an umbrella?"

## 🔐 Configuration

1. Create `.env` from `.env.example` in `weather-rag-gradio/`
2. Add your Google API key:
   ```env
   GOOGLE_API_KEY=your_key_here
   ```

## 📝 License

This project is open source and available for educational and research purposes.

## 👤 Author

[VikashKumar1424](https://github.com/VikashKumar1424)

---

**Ready to get started?** Head to [weather-rag-gradio/README.md](weather-rag-gradio/README.md) for detailed setup instructions!
