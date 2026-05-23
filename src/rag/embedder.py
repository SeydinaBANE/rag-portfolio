import cohere

from src.config import settings

_client: cohere.AsyncClient | None = None


def get_cohere_client() -> cohere.AsyncClient:
    global _client
    if _client is None:
        _client = cohere.AsyncClient(api_key=settings.cohere_api_key)
    return _client


async def embed_texts(texts: list[str]) -> list[list[float]]:
    client = get_cohere_client()
    response = await client.embed(
        texts=texts,
        model=settings.embedding_model,
        input_type="search_document",
    )
    return [list(e) for e in response.embeddings]


async def embed_query(text: str) -> list[float]:
    client = get_cohere_client()
    response = await client.embed(
        texts=[text],
        model=settings.embedding_model,
        input_type="search_query",
    )
    return list(response.embeddings[0])
