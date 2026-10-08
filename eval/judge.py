import os

from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

client = OpenAI(
    api_key=DEEPSEEK_API_KEY,
    base_url="https://api.deepseek.com",
)


def judge_answer(question, reference_answer, generated_answer):
    prompt = f"""
You are evaluating the correctness of a RAG system.

Question:
{question}

Reference answer:
{reference_answer}

Generated answer:
{generated_answer}

Determine whether the generated answer is factually consistent
with the reference answer.

Extra wording is allowed as long as it does not contradict the
reference answer.

Respond with exactly one word:

CORRECT

or

INCORRECT
"""

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {
                "role": "system",
                "content": "You are a strict answer evaluator.",
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        temperature=0,
    )

    verdict = response.choices[0].message.content.strip().upper()

    return verdict == "CORRECT"