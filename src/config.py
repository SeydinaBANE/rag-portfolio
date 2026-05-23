from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # OpenRouter (LLM)
    openrouter_api_key: str
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    chat_model: str = "nvidia/nemotron-3-super-120b-a12b:free"

    # Cohere (embeddings)
    cohere_api_key: str
    embedding_model: str = "embed-multilingual-v3.0"

    # PostgreSQL
    database_url: str

    # MLflow
    mlflow_tracking_uri: str = "http://localhost:5000"
    mlflow_experiment_name: str = "knowledgeforge-rag"

    # Pipeline
    chunk_size: int = 512
    chunk_overlap: int = 64
    top_k_retrieval: int = 5
    embedding_dimensions: int = 1024

    # App
    log_level: str = "INFO"
    env: str = "development"


settings = Settings()  # type: ignore[call-arg]
