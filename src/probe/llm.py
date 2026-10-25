"""One configurable OpenAI-compatible client for all agents."""

import json
import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def parse_json_object(text: str) -> dict[str, Any]:
    """Accept a JSON object, optionally enclosed in one Markdown code fence."""
    content = text.strip()
    if content.startswith("```"):
        lines = content.splitlines()
        if len(lines) < 3 or lines[-1].strip() != "```":
            raise ValueError("Unclosed JSON code fence")
        content = "\n".join(lines[1:-1])
    value = json.loads(content)
    if not isinstance(value, dict):
        raise ValueError("The agent must return one JSON object")
    return value


class LLMClient:
    def __init__(self, model: str, *, base_url: str | None = None,
                 temperature: float = 0.7, timeout: float = 300,
                 max_retries: int = 2, trace_path: Path | None = None):
        from openai import OpenAI

        key = os.environ.get("OPENAI_API_KEY")
        if not key:
            raise ValueError("Set OPENAI_API_KEY before running PROBE")
        self.model = model
        self.temperature = temperature
        self.trace_path = trace_path
        self.calls = 0
        self._client = OpenAI(
            api_key=key, base_url=base_url or os.environ.get("OPENAI_BASE_URL") or
            os.environ.get("OPENAI_API_BASE") or "https://api.openai.com/v1",
            timeout=timeout, max_retries=max_retries,
        )

    def complete(self, prompt: str) -> str:
        response = self._client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
        )
        content = response.choices[0].message.content
        if not content:
            raise RuntimeError("Language model returned an empty response")
        self.calls += 1
        if self.trace_path is not None:
            self.trace_path.parent.mkdir(parents=True, exist_ok=True)
            record = {"call": self.calls, "model": self.model, "prompt": prompt,
                      "response": content,
                      "usage": response.usage.model_dump() if response.usage else None}
            with self.trace_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        logger.info("Completed model call %d", self.calls)
        return content

    def json(self, prompt: str) -> dict[str, Any]:
        return parse_json_object(self.complete(prompt))
