from weather_rag.embeddings import GeminiEmbedding2


def main() -> None:
    print("Testing Gemini embedding...")

    embeddings = GeminiEmbedding2()

    result = embeddings.embed_query(
        "What is the weather in Delhi today?"
    )

    print("SUCCESS")
    print("Embedding size:", len(result))
    print("First 5 values:", result[:5])


if __name__ == "__main__":
    main()