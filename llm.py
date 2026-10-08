import os
import math

from dotenv import load_dotenv
from openai import OpenAI, APIError, APIConnectionError, APITimeoutError, APIStatusError
from authorization import authorize_results
from graph_rag import graph_context


load_dotenv()

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

if not DEEPSEEK_API_KEY:
    raise ValueError("DEEPSEEK_API_KEY was not found in .env")


def completion_timeout():
    """Reject configuration that could silently disable the provider timeout."""
    value = float(os.getenv("LLM_TIMEOUT_SECONDS", "30"))
    if not math.isfinite(value) or value <= 0:
        raise ValueError("LLM_TIMEOUT_SECONDS must be a finite positive number")
    return value


class GenerationError(Exception):
    """Fixed public error metadata, excluding provider bodies and prompts."""

    def __init__(self, reason, status_code, detail):
        super().__init__(detail)
        self.reason = reason
        self.status_code = status_code
        self.detail = detail


client = OpenAI(
    api_key=DEEPSEEK_API_KEY,
    base_url="https://api.deepseek.com",
    timeout=completion_timeout(),
    max_retries=0,
)


def generate_answer(question, search_results, *, tenant_id, role):
    # Enforce the policy at the context-construction boundary as well as in
    # the API, so direct callers cannot accidentally bypass authorization.
    authorize_results(search_results, tenant_id, role)
    context_parts = []

    for result in search_results:
        source = result.payload["source"]
        chunk_id = result.payload["chunk_id"]
        text = result.payload["text"]

        context_parts.append(
            f"[Source: {source}, Chunk: {chunk_id}]\n{text}"
        )

    context = "\n\n".join(context_parts)

    return _complete_answer(question, context)


def generate_graph_answer(question, search_results, *, tenant_id, role, plan):
    context = graph_context(plan, search_results, tenant_id=tenant_id, role=role)
    return _complete_answer(question, context)


def _complete_answer(question, context):

    prompt = f"""
Answer the user's question using only the context below.

If the answer cannot be found in the context, say:
"I don't know based on the provided documents."

Do not invent information.

Include the source filename and chunk ID you used.

Context:
{context}

Question:
{question}
"""

    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a document question-answering assistant. "
                        "Answer only from the provided context."
                    ),
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            temperature=0,
        )
    except APITimeoutError:
        raise GenerationError("generation_timeout", 504, "Answer provider timed out") from None
    except APIConnectionError:
        raise GenerationError("generation_unavailable", 503, "Answer provider temporarily unavailable") from None
    except APIStatusError as exc:
        if exc.status_code == 429 or exc.status_code >= 500:
            raise GenerationError("generation_unavailable", 503, "Answer provider temporarily unavailable") from None
        raise GenerationError("generation_rejected", 502, "Answer provider could not complete the request") from None
    except APIError:
        raise GenerationError("generation_invalid_response", 502, "Answer provider returned an invalid response") from None

    choices = getattr(response, "choices", None)
    choice = choices[0] if isinstance(choices, (list, tuple)) and choices else None
    content = getattr(getattr(choice, "message", None), "content", None)
    if (not isinstance(content, str) or not content.strip()
            or getattr(choice, "finish_reason", "stop") != "stop"):
        raise GenerationError("generation_invalid_response", 502, "Answer provider returned an invalid response")
    return content

if __name__ == "__main__":
    from search import search_documents

    question = "How many days of annual leave do employees get?"

    results = search_documents(
        question,
        tenant_id="acme",
        role="employee",
        limit=3,
    )

    answer = generate_answer(
        question,
        results,
        tenant_id="acme",
        role="employee",
    )

    print("\nQuestion:")
    print(question)

    print("\nAnswer:")
    print(answer)
