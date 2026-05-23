from src.pipeline.transformers.cleaner import clean_pages, clean_text


def test_clean_text_strips_whitespace():
    assert clean_text("  hello   world  ") == "hello world"


def test_clean_text_normalizes_newlines():
    text = "line1\n\n\n\nline2"
    result = clean_text(text)
    assert "\n\n\n" not in result


def test_clean_text_removes_control_chars():
    result = clean_text("hello\x00world")
    assert "\x00" not in result


def test_clean_text_normalizes_unicode():
    result = clean_text("café")
    assert result == "café"


def test_clean_pages_filters_empty():
    pages = ["  ", "hello", "", "world"]
    result = clean_pages(pages)
    assert len(result) == 2
    assert "hello" in result
    assert "world" in result
