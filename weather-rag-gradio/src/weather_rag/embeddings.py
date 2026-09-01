
from __future__ import annotations

from typing import List

from google import genai
from langchain_core.embeddings import Embeddings

from .config import GOOGLE_API_KEY


class GeminiEmbedding2(Embeddings):
    """
    LangChain-compatible Gemini embedding implementation.

    Uses the Google GenAI SDK and GOOGLE_API_KEY.

    Gemini's BatchEmbedContents API supports a maximum of
    100 requests per batch, so documents are automatically
    split into batches.
    """

    DEFAULT_MODEL = "gemini-embedding-001"
    MAX_BATCH_SIZE = 100

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        batch_size: int = MAX_BATCH_SIZE,
    ) -> None:

        if not GOOGLE_API_KEY:
            raise RuntimeError(
                "GOOGLE_API_KEY is not configured."
            )

        if batch_size < 1 or batch_size > 100:
            raise ValueError(
                "batch_size must be between 1 and 100."
            )

        self.model = model
        self.batch_size = batch_size

        self.client = genai.Client(
            api_key=GOOGLE_API_KEY,
        )

    def embed_documents(
        self,
        texts: List[str],
    ) -> List[List[float]]:
        """
        Embed documents in batches.

        Gemini allows at most 100 embedding requests
        in a single BatchEmbedContents request.
        """

        if not texts:
            return []

        all_embeddings: List[List[float]] = []

        for start in range(
            0,
            len(texts),
            self.batch_size,
        ):
            batch = texts[
                start : start + self.batch_size
            ]

            response = self.client.models.embed_content(
                model=self.model,
                contents=batch,
            )

            batch_embeddings = [
                embedding.values
                for embedding in response.embeddings
            ]

            all_embeddings.extend(
                batch_embeddings
            )

        return all_embeddings

    def embed_query(
        self,
        text: str,
    ) -> List[float]:
        """
        Embed a single user query.
        """

        if not text or not text.strip():
            raise ValueError(
                "Query text cannot be empty."
            )

        response = self.client.models.embed_content(
            model=self.model,
            contents=text,
        )

        return response.embeddings[0].values
