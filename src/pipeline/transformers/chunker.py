from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import settings


def chunk_pages(
    pages: list[str], chunk_size: int | None = None, overlap: int | None = None
) -> list[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size if chunk_size is not None else settings.chunk_size,
        chunk_overlap=overlap if overlap is not None else settings.chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks: list[str] = []
    for page in pages:
        chunks.extend(splitter.split_text(page))
    return [c for c in chunks if c.strip()]
