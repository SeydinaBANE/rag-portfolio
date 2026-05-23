import httpx
from bs4 import BeautifulSoup

from src.pipeline.loaders.pdf_loader import RawDocument


async def load_url(url: str) -> RawDocument:
    async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
        response = await client.get(url, headers={"User-Agent": "KnowledgeForge/0.1"})
        response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")
    title = soup.title.string.strip() if soup.title and soup.title.string else None

    # Remove script/style noise
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()

    text = soup.get_text(separator="\n", strip=True)
    return RawDocument(
        source_uri=url,
        doc_type="url",
        title=title,
        pages=[text],
        metadata={"url": url, "status_code": response.status_code},
    )
