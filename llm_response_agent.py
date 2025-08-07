# llm_response_agent.py

import os
import openai
from typing import List, Dict
from dotenv import load_dotenv

load_dotenv()

class LLMResponseAgent:
    def __init__(self, model: str = "gpt-3.5-turbo"):
        openai.api_key = os.getenv("OPENAI_API_KEY")
        self.model = model

    def format_prompt(self, query: str, context_chunks: List[str]) -> str:
        context = "\n\n".join(context_chunks)
        return (
            "You are an expert assistant helping users understand document content.\n"
            "Based on the following context, answer the user's question concisely and clearly.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {query}\n\n"
            "Answer:"
        )

    def generate_answer(self, query: str, top_chunks: List[Dict]) -> str:
        context_texts = [chunk["chunk"] for chunk in top_chunks]
        prompt = self.format_prompt(query, context_texts)
        try:
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=500,
            )
            return response['choices'][0]['message']['content'].strip()
        except Exception as e:
            return f"Error: {str(e)}"