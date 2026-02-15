# core/llm_interface.py
import json
import hashlib
from typing import List, Dict, Optional
from config import settings
import requests

# ---------------- LOCAL (OLLAMA 0.6.x) ----------------
try:
    import ollama
except ImportError:
    ollama = None  # fallback if Ollama not installed

class LLMClient:
    """
    Unified LLM client supporting:
    - Local Ollama
    - HuggingFace
    - OpenRouter
    - Groq
    """

    def __init__(self, provider: Optional[str] = None, model: Optional[str] = None):
        self.provider = (provider or settings.LLM_PROVIDER).lower()
        self.model = model or getattr(settings, "LLM_MODEL", "llama3:8b")
        self.api_key = None

        if self.provider != "local":
            self.api_key = getattr(settings, f"{self.provider.upper()}_API_KEY", None)
            if self.provider in ["groq", "openrouter"] and not self.api_key:
                raise ValueError(f"API key for {self.provider} not set in .env")

        if self.provider == "local":
            if ollama is None:
                print("⚠️ Ollama not installed. Local mode will use dummy responses.")
            self.local_client = ollama
            self.model = getattr(settings, "LLM_MODEL", self.model)

        self._cache = {}

    # ---------------- GENERATE TEXT ----------------
    def generate(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.2,
        max_tokens: int = 150,
        response_format: Optional[Dict] = None,
        batch_prompt: Optional[str] = None
    ) -> str:
        """Unified generation entry point"""
        key_source = batch_prompt or json.dumps(messages)
        key = hashlib.md5(key_source.encode("utf-8")).hexdigest()
        if key in self._cache:
            return self._cache[key]

        if batch_prompt:
            messages = [{"role": "user", "content": batch_prompt}]

        try:
            if self.provider == "local":
                result = self._local(messages, temperature, max_tokens)
            elif self.provider == "huggingface":
                result = self._huggingface(messages, temperature, max_tokens, response_format)
            elif self.provider == "openrouter":
                result = self._openrouter(messages, temperature, max_tokens, response_format)
            elif self.provider == "groq":
                result = self._groq(messages, temperature, max_tokens, response_format)
            else:
                raise ValueError(f"Unsupported provider: {self.provider}")
        except Exception as e:
            print(f"⚠️ {self.provider} generation failed: {e}")
            # fallback to HuggingFace free tier
            if self.provider != "huggingface":
                fallback_client = LLMClient(provider="huggingface", model=getattr(settings, "EMBEDDING_MODEL", "BAAI/bge-large-en-v1.5"))
                result = fallback_client.generate(messages, temperature, max_tokens, response_format)
            else:
                # return dummy if even HuggingFace fails
                result = self._dummy(messages)

        self._cache[key] = result
        return result

    # ---------------- LOCAL (OLLAMA) ----------------
    def _local(self, messages, temperature=0.2, max_tokens=150, response_format=None):
        if self.local_client is None:
            return self._dummy(messages)

        model_to_use = getattr(settings, "LLM_MODEL", self.model)
        try:
            resp = self.local_client.chat(
                model=model_to_use,
                messages=messages
            )
            if isinstance(resp, dict) and "content" in resp:
                return resp["content"]
            return str(resp)
        except Exception:
            return self._dummy(messages)

    # ---------------- DUMMY FALLBACK ----------------
    def _dummy(self, messages):
        """Return safe dummy questions (JSON list) for local testing."""
        last_msg = ""
        if messages and isinstance(messages, list):
            last = messages[-1]
            if isinstance(last, dict):
                last_msg = last.get("content", "")
        last_msg = str(last_msg)[:50]  # truncate
        return json.dumps([
            {"question": f"Dummy question 1 based on: {last_msg}"},
            {"question": f"Dummy question 2 based on: {last_msg}"},
            {"question": f"Dummy question 3 based on: {last_msg}"}
        ])

    # ---------------- OPENROUTER ----------------
    def _openrouter(self, messages, temperature, max_tokens, response_format=None):
        url = "https://openrouter.ai/api/v1/chat/completions"
        model = self.model
        if ":free" not in model:
            model += ":free"
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        if response_format:
            payload["response_format"] = response_format
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]

    # ---------------- GROQ ----------------
    def _groq(self, messages, temperature, max_tokens, response_format=None):
        url = "https://api.groq.com/openai/v1/chat/completions"
        payload = {"model": self.model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens}
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]

    # ---------------- HUGGINGFACE ----------------
    def _huggingface(self, messages, temperature, max_tokens, response_format=None):
        prompt = "\n".join([m.get("content","") for m in messages])
        url = f"https://api-inference.huggingface.co/models/{self.model}"
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        payload = {"inputs": prompt, "parameters": {"temperature": temperature, "max_new_tokens": max_tokens}}
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        data = response.json()
        if isinstance(data, dict) and "error" in data:
            raise RuntimeError(f"HuggingFace API error: {data['error']}")
        if isinstance(data, list) and "generated_text" in data[0]:
            return data[0]["generated_text"]
        return str(data)
