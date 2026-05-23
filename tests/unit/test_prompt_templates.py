from src.rag.prompt_templates import DEFAULT_VERSION, PROMPT_VERSIONS, get_prompt


def test_default_version_exists():
    assert DEFAULT_VERSION in PROMPT_VERSIONS


def test_get_prompt_returns_template():
    prompt = get_prompt("v1")
    assert prompt is not None


def test_get_prompt_fallback_on_unknown_version():
    prompt = get_prompt("v99")
    assert prompt is not None  # should not raise


def test_prompt_versions_have_content():
    for version, system in PROMPT_VERSIONS.items():
        assert len(system) > 10, f"Prompt version {version} is too short"
