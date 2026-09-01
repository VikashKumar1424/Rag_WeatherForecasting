
from __future__ import annotations

from pathlib import Path
from typing import Any
import hashlib
import re

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import (
    CHROMA_DIR,
    GEMINI_MODEL,
    GOOGLE_API_KEY,
)

from .embeddings import GeminiEmbedding2
from .weather_codes import weather_description


# =========================================================
# WEATHER DOCUMENT CREATION
# =========================================================

def weather_documents(
    forecast: dict[str, Any],
    location_name: str,
    latitude: float,
    longitude: float,
) -> list[Document]:
    """
    Convert Open-Meteo hourly forecast data into LangChain
    Documents suitable for RAG and Chroma.
    """

    hourly = forecast["hourly"]

    times = hourly["time"]

    documents: list[Document] = []

    fields = [
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

    for i, timestamp in enumerate(times):

        values = {
            field: hourly.get(
                field,
                [None] * len(times),
            )[i]
            for field in fields
        }

        code = values["weather_code"]

        text = f"""
Weather forecast for {location_name}

Coordinates: {latitude}, {longitude}

Local time: {timestamp}

Temperature: {values["temperature_2m"]} °C

Apparent temperature: {values["apparent_temperature"]} °C

Relative humidity: {values["relative_humidity_2m"]} %

Precipitation: {values["precipitation"]} mm

Rain: {values["rain"]} mm

Weather condition: {weather_description(code)}

Cloud cover: {values["cloud_cover"]} %

Wind speed: {values["wind_speed_10m"]} km/h

Wind direction: {values["wind_direction_10m"]} degrees

Source: Open-Meteo
""".strip()

        documents.append(
            Document(
                page_content=text,
                metadata={
                    "source": "Open-Meteo",
                    "location": location_name,
                    "timestamp": timestamp,
                    "latitude": latitude,
                    "longitude": longitude,
                },
            )
        )

    return documents


# =========================================================
# COLLECTION NAME
# =========================================================

def _collection_name(
    documents: list[Document],
) -> str:

    location = (
        str(
            documents[0].metadata.get(
                "location",
                "weather",
            )
        )
        if documents
        else "weather"
    )

    safe = re.sub(
        r"[^a-z0-9]+",
        "-",
        location.lower(),
    ).strip("-") or "weather"

    digest = hashlib.sha1(
        location.encode("utf-8")
    ).hexdigest()[:8]

    return f"weather-{safe[:40]}-{digest}"


# =========================================================
# BUILD VECTOR STORE
# =========================================================

def build_vector_store(
    documents: list[Document],
    persist_directory: Path = CHROMA_DIR,
) -> Chroma:
    """
    Split weather documents and store embeddings in Chroma.
    """

    if not documents:
        raise ValueError(
            "Cannot build vector store with no documents."
        )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=100,
    )

    chunks = splitter.split_documents(
        documents
    )

    persist_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    collection_name = _collection_name(
        chunks
    )

    # -----------------------------------------------------
    # Delete existing collection for this location
    # -----------------------------------------------------

    vector_store = Chroma(
        collection_name=collection_name,
        embedding_function=GeminiEmbedding2(),
        persist_directory=str(
            persist_directory
        ),
    )

    try:
        vector_store.delete_collection()
    except Exception:
        pass

    # -----------------------------------------------------
    # Create fresh collection
    # -----------------------------------------------------

    vector_store = Chroma(
        collection_name=collection_name,
        embedding_function=GeminiEmbedding2(),
        persist_directory=str(
            persist_directory
        ),
    )

    vector_store.add_documents(
        chunks
    )

    return vector_store


# =========================================================
# LOAD VECTOR STORE
# =========================================================

def load_vector_store(
    location_name: str,
    persist_directory: Path = CHROMA_DIR,
) -> Chroma:
    """
    Load an existing Chroma collection for a location.
    """

    digest = hashlib.sha1(
        location_name.encode("utf-8")
    ).hexdigest()[:8]

    safe = re.sub(
        r"[^a-z0-9]+",
        "-",
        location_name.lower(),
    ).strip("-") or "weather"

    collection_name = (
        f"weather-{safe[:40]}-{digest}"
    )

    return Chroma(
        collection_name=collection_name,
        embedding_function=GeminiEmbedding2(),
        persist_directory=str(
            persist_directory
        ),
    )


# =========================================================
# GEMINI RESPONSE NORMALIZER
# =========================================================

def _extract_text(
    content: Any,
) -> str:
    """
    Normalize Gemini/LangChain response content.

    Gemini may return either:

        "plain text"

    or:

        [
            {
                "type": "text",
                "text": "..."
            }
        ]

    This function extracts only human-readable text.
    """

    # -----------------------------------------------------
    # Normal string
    # -----------------------------------------------------

    if isinstance(content, str):
        return content.strip()

    # -----------------------------------------------------
    # Structured content
    # -----------------------------------------------------

    if isinstance(content, list):

        text_parts: list[str] = []

        for item in content:

            if isinstance(item, str):

                text_parts.append(
                    item
                )

            elif isinstance(item, dict):

                if item.get("type") == "text":

                    text = item.get(
                        "text"
                    )

                    if text:
                        text_parts.append(
                            str(text)
                        )

        result = "\n".join(
            text_parts
        ).strip()

        if result:
            return result

    # -----------------------------------------------------
    # Fallback
    # -----------------------------------------------------

    return str(content).strip()


# =========================================================
# ANSWER QUESTION
# =========================================================

def answer_question(
    question: str,
    vector_store: Chroma,
    location_name: str,
) -> str:
    """
    Answer a weather question using RAG.

    The retrieved Open-Meteo context is supplied to Gemini.
    Gemini is instructed not to invent weather information.

    The final response is normalized into a plain string
    before being returned to Gradio.
    """

    # -----------------------------------------------------
    # Similarity search
    # -----------------------------------------------------

    docs = vector_store.similarity_search(
        question,
        k=8,
    )

    if not docs:

        return (
            "I could not find relevant weather data. "
            "Please refresh the weather index."
        )

    # -----------------------------------------------------
    # Build context
    # -----------------------------------------------------

    context = "\n\n---\n\n".join(
        doc.page_content
        for doc in docs
    )

    # -----------------------------------------------------
    # Gemini
    # -----------------------------------------------------

    llm = ChatGoogleGenerativeAI(
        model=GEMINI_MODEL,
        google_api_key=GOOGLE_API_KEY,
        temperature=0.2,
    )

    # -----------------------------------------------------
    # Prompt
    # -----------------------------------------------------

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
You are a weather monitoring assistant.

Answer the user's question using ONLY the supplied
Open-Meteo weather context.

Rules:

1. Do not invent weather values.
2. Do not use information outside the supplied context.
3. If the required information is not present, clearly say
   that the forecast information is not available.
4. Always mention the relevant date when answering
   questions about today or tomorrow.
5. Include units such as °C, %, mm, and km/h.
6. For rain questions, clearly state whether rain or
   precipitation is expected.
7. Use concise, user-friendly language.
8. Practical advice is allowed, but distinguish advice
   from actual forecast facts.
9. Return ONLY normal human-readable text.
10. Do NOT return JSON.
11. Do NOT return Python dictionaries.
12. Do NOT return Python lists.
13. Do NOT return metadata.
14. Do NOT return signatures.

Location: {location}
""",
            ),
            (
                "human",
                """
Question:

{question}

Open-Meteo retrieved weather context:

{context}
""",
            ),
        ]
    )

    # -----------------------------------------------------
    # Invoke Gemini
    # -----------------------------------------------------

    chain = prompt | llm

    response = chain.invoke(
        {
            "location": location_name,
            "question": question,
            "context": context,
        }
    )

    # -----------------------------------------------------
    # Normalize response
    # -----------------------------------------------------

    return _extract_text(
        response.content
    )
