import re
import unicodedata


def clean_text(text: str) -> str:
    """Normalize unicode, strip excessive whitespace, remove control characters."""
    text = unicodedata.normalize("NFKC", text)
    # Remove non-printable control characters except newlines and tabs
    text = re.sub(r"[^\S\n\t ]+", " ", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def clean_pages(pages: list[str]) -> list[str]:
    return [clean_text(p) for p in pages if clean_text(p)]
