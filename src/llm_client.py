import requests
import time
from typing import List, Dict, Optional

from src.config import Config

class LLMClient:
    """Client for LLM API"""

    def __init__(self, config: Config):
        self.config = config
        self.api_key = config.get_api_key()
        self.api_url = config.llm.api_url
        self.model = config.llm.model
        self.provider = config.llm.provider

        #set headers
        self.headers = self.get_headers()

        if config.system.verbose:
            print(f"LLM client initialized ({self.provider})")

    def get_headers(self) -> Dict[str, str]:
        """get headers based on the provider"""
        base_headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        if self.provider == "openrouter":
            base_headers.update({
                "HTTP-Referer": "http://localhost:3000",
                "X-Title": "Movie RAG System"
            })

        return base_headers

    def generate_answer(self, contexts: List[str], query: str) -> Dict:
        """generate answer using LLM API"""

        prompt = self.format_prompt(contexts, query)

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a helpful assistant that answers querys about movies "
                        "based on plot information. Provide clear, accurate answers based "
                        "only on the given context."
                    )
                },
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            "temperature": self.config.llm.parameters.temperature,
            "max_tokens": self.config.llm.parameters.max_tokens
        }

        #Retry
        for attempt in range(self.config.llm.retry.max_attempts):
            try:
                response = requests.post(
                    self.api_url,
                    headers=self.headers,
                    json=payload,
                    timeout=self.config.llm.parameters.timeout
                )
                response.raise_for_status()

                result = response.json()
                answer = result['choices'][0]['message']['content'].strip()

                return {
                    'answer': answer,
                    'model': self.model,
                    'usage': result.get('usage', {}),
                    'success': True,
                }
            except requests.exceptions.RequestException as e:
                if attempt < self.config.llm.retry.max_attempts - 1:
                    wait_time = self.config.llm.retry.backoff_factor ** attempt
                    if self.config.system.verbose:
                        print(f"API request failed (attempt {attempt + 1}), "
                              f"retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    return {
                        'answer': f"Error: Failed to generate answer - {str(e)}",
                        'success': False,
                        'error': str(e)
                    }

        return {
            'answer': "Error: Max retry attempts reached",
            'success': False,
            'error': "Max retries exceeded"
        }

    def format_prompt(self, contexts: List[str], query: str) -> str:
        """format a prompt for LLM"""
        # join contexts
        contexts_text = "\n\n".join([
            f"Context {i + 1}:\n{ctx}"
            for i, ctx in enumerate(contexts)
        ])

        prompt = f"""Based on the following movie plot contexts, answer the query about movies.

        {contexts_text}

        query: {query}

        Instructions:
        - Provide a clear, accurate answer based on the given contexts
        - If the answer cannot be fully determined from the contexts, say so
        - Mention specific movie titles when relevant
        - Keep the answer concise but informative

        Answer:"""

        return prompt

    def generate_reasoning(self, contexts: List[str], query: str, answer: str) -> str:
        """generate reasoning based on answer"""
        reasoning_prompt = f"""query: {query}

        Your answer: {answer}

        Briefly explain (in 1-2 sentences) how you formed this answer from the given movie plot contexts. 
        Mention which movies or plot elements were most relevant."""

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": "You provide brief explanations of reasoning."
                },
                {
                    "role": "user",
                    "content": reasoning_prompt,
                }
            ],
            "temperature": 0.5,
            "max_tokens": 150
        }

        try:
            response = requests.post(
                self.api_url,
                headers=self.headers,
                json=payload,
                timeout=self.config.llm.parameters.timeout
            )
            response.raise_for_status()

            result = response.json()
            reasoning = result['choices'][0]['message']['content'].strip()
            return reasoning

        except Exception as e:
            return f"Generated answer based on retrieved movie plot contexts. ({str(e)})"