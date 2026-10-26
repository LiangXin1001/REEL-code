from __future__ import annotations

import base64
import time
from pathlib import Path
from typing import Any

from openai import OpenAI


class OpenAIVLMClient:
    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str = "EMPTY",
        timeout: float = 240.0,
        max_retries: int = 0,
    ) -> None:
        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key,
            timeout=timeout,
            max_retries=max_retries,
        )
        self.model = model

    def chat(
        self,
        messages: list[dict[str, Any]],
        max_tokens: int = 256,
        temperature: float = 0.0,
        response_format: dict[str, Any] | None = None,
        stop: str | list[str] | None = None,
        stop_token_ids: list[int] | None = None,
        min_tokens: int | None = None,
    ) -> dict:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if response_format is not None:
            kwargs["response_format"] = response_format
        if stop:
            kwargs["stop"] = stop
        extra_body: dict[str, Any] = {}
        if stop_token_ids:
            # vLLM exposes this native sampling option through extra_body even
            # though it is not part of the provider-neutral OpenAI schema.
            extra_body["stop_token_ids"] = stop_token_ids
        if min_tokens is not None:
            extra_body["min_tokens"] = min_tokens
        if extra_body:
            kwargs["extra_body"] = extra_body
        resp = self.client.chat.completions.create(
            **kwargs,
        )
        usage = getattr(resp, "usage", None)
        choice = resp.choices[0]
        return {
            "text": choice.message.content or "",
            "usage": usage.model_dump() if usage else {},
        }

    def chat_stream_timed(
        self,
        messages: list[dict[str, Any]],
        max_tokens: int = 256,
        temperature: float = 0.0,
        response_format: dict[str, Any] | None = None,
        stop: str | list[str] | None = None,
        stop_token_ids: list[int] | None = None,
        min_tokens: int | None = None,
    ) -> dict:
        """Run a streaming chat request and retain request-level timing.

        TTFT is measured at the first non-empty content chunk. TPOT uses the
        provider-reported completion-token count, so it is not affected when a
        server groups multiple tokens into one streamed chunk.
        """
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if response_format is not None:
            kwargs["response_format"] = response_format
        if stop:
            kwargs["stop"] = stop
        extra_body: dict[str, Any] = {}
        if stop_token_ids:
            extra_body["stop_token_ids"] = stop_token_ids
        if min_tokens is not None:
            extra_body["min_tokens"] = min_tokens
        if extra_body:
            kwargs["extra_body"] = extra_body

        started = time.perf_counter()
        first_content_at: float | None = None
        content_parts: list[str] = []
        usage_dict: dict[str, Any] = {}
        for chunk in self.client.chat.completions.create(**kwargs):
            usage = getattr(chunk, "usage", None)
            if usage is not None:
                usage_dict = usage.model_dump()
            for choice in getattr(chunk, "choices", []) or []:
                piece = getattr(getattr(choice, "delta", None), "content", None)
                if piece:
                    now = time.perf_counter()
                    if first_content_at is None:
                        first_content_at = now
                    content_parts.append(piece)

        finished = time.perf_counter()
        if first_content_at is None:
            raise RuntimeError("stream completed without a non-empty content chunk")
        completion_tokens = int(usage_dict.get("completion_tokens") or 0)
        ttft = first_content_at - started
        e2e = finished - started
        tpot = (
            (e2e - ttft) / (completion_tokens - 1)
            if completion_tokens > 1
            else None
        )
        return {
            "text": "".join(content_parts),
            "usage": usage_dict,
            "timing": {
                "ttft_s": ttft,
                "e2e_stream_s": e2e,
                "tpot_s": tpot,
            },
        }


def image_to_data_url(path: str) -> str:
    data = base64.b64encode(Path(path).read_bytes()).decode("ascii")
    return f"data:image/jpeg;base64,{data}"


def build_frame_messages(
    frame_paths: list[str],
    question: str,
    options: list[str],
    prompt: str,
    media_mode: str = "file",
) -> list[dict]:
    content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    for idx, path in enumerate(frame_paths):
        content.append({"type": "text", "text": f"Frame {idx}:"})
        if media_mode == "data-url":
            url = image_to_data_url(path)
        elif media_mode == "file":
            url = f"file://{Path(path).resolve()}"
        else:
            raise ValueError(f"unsupported media mode: {media_mode}")
        content.append({"type": "image_url", "image_url": {"url": url}})

    option_text = "\n".join(options)
    content.append(
        {
            "type": "text",
            "text": f"Question:\n{question}\n\nOptions:\n{option_text}",
        }
    )
    return [{"role": "user", "content": content}]
