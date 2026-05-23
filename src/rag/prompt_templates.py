from langchain_core.prompts import ChatPromptTemplate

SYSTEM_V1 = """You are a precise technical assistant. Answer the user's question using ONLY the provided context.
If the context does not contain enough information to answer, say so clearly.
Always cite the source documents when relevant.
Be concise and factual."""

SYSTEM_V2 = """You are a concise technical assistant. Use ONLY the provided context to answer.
If the answer is not in the context, respond: "I don't have enough information to answer this question."
Keep answers under 200 words unless more detail is explicitly requested."""

PROMPT_VERSIONS: dict[str, str] = {
    "v1": SYSTEM_V1,
    "v2": SYSTEM_V2,
}

DEFAULT_VERSION = "v1"


def get_prompt(version: str = DEFAULT_VERSION) -> ChatPromptTemplate:
    system = PROMPT_VERSIONS.get(version, SYSTEM_V1)
    return ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "Context:\n{context}\n\nQuestion: {question}"),
        ]
    )
