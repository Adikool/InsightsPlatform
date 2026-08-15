"""Claude client wrapper.

Two things live here that are easy to get wrong elsewhere:

1. **Schema hardening.** The structured-outputs schema must be closed
   (`additionalProperties: false`, every property `required`) and must not carry
   validation keywords the API does not accept. Pydantic emits a superset of that,
   so we post-process before sending.
2. **Cache layout.** The catalog description is large and identical across
   questions, so it goes in a cached system block *before* the question. Anything
   volatile (the question, the timestamp) goes after the last breakpoint.
"""

from __future__ import annotations

import json
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from ..config import settings
from ..errors import LLMUnavailable, PlatformError

T = TypeVar("T", bound=BaseModel)

# Validation keywords the structured-outputs schema compiler does not accept.
_STRIP_KEYS = {
    "default", "minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum",
    "multipleOf", "minLength", "maxLength", "pattern", "minItems", "maxItems",
    "uniqueItems", "minProperties", "maxProperties", "examples", "format",
}


def strict_schema(model: type[BaseModel]) -> dict[str, Any]:
    """Pydantic model -> a schema the API will accept in `output_config.format`."""

    def harden(node: Any) -> Any:
        if isinstance(node, list):
            return [harden(item) for item in node]
        if not isinstance(node, dict):
            return node

        out = {k: harden(v) for k, v in node.items() if k not in _STRIP_KEYS}

        if out.get("type") == "object" or "properties" in out:
            out.setdefault("type", "object")
            out["additionalProperties"] = False
            props = out.get("properties", {})
            # Strict mode ignores defaults, so every property must be required and
            # the model must fill it explicitly.
            out["required"] = list(props.keys())
        return out

    return harden(model.model_json_schema())


class LLMClient:
    """Thin wrapper. Absent credentials it raises LLMUnavailable so callers can
    fall back to the deterministic parser rather than dying."""

    def __init__(self, model: str | None = None, effort: str | None = None) -> None:
        self.model = model or settings.model
        self.effort = effort or settings.effort
        self._client = None

    @property
    def client(self):
        if self._client is None:
            try:
                import anthropic
            except ImportError as exc:  # pragma: no cover
                raise LLMUnavailable("the `anthropic` package is not installed") from exc
            if not settings.has_llm:
                raise LLMUnavailable(
                    "no ANTHROPIC_API_KEY / ANTHROPIC_AUTH_TOKEN in the environment"
                )
            self._client = anthropic.Anthropic()
        return self._client

    # ------------------------------------------------------------------ calls
    def _create(self, *, max_tokens: int | None, **kwargs) -> Any:
        """The one place that actually talks to the network.

        Every failure mode of the call itself — auth, rate limit, overload,
        connection drop, a 5xx — comes back as a typed `anthropic.APIError`
        subclass, not a `PlatformError`. Every caller in this codebase catches
        `PlatformError` to decide "fall back to the deterministic path"; left
        unguarded, any of those would skip that check and surface as a raw
        traceback (a 500 from the API, an unhandled exception from the CLI)
        instead of a graceful degrade. This is the single choke point, so it's
        the one place that needs the translation.
        """
        import anthropic

        try:
            return self.client.messages.create(
                model=self.model,
                max_tokens=max_tokens or settings.max_tokens,
                thinking={"type": "adaptive"},
                **kwargs,
            )
        except anthropic.APIError as exc:
            raise PlatformError(f"the model call failed: {exc}") from exc

    def structured(
        self,
        *,
        instructions: str,
        context: str,
        question: str,
        output_model: type[T],
        max_tokens: int | None = None,
        effort: str | None = None,
    ) -> T:
        """One request, one validated pydantic object back."""
        system = [
            {"type": "text", "text": instructions},
            # Breakpoint after the schema+catalog: stable across every question.
            {"type": "text", "text": context, "cache_control": {"type": "ephemeral"}},
        ]
        response = self._create(
            max_tokens=max_tokens,
            system=system,
            output_config={
                "effort": effort or self.effort,
                "format": {"type": "json_schema", "schema": strict_schema(output_model)},
            },
            messages=[{"role": "user", "content": question}],
        )

        if response.stop_reason == "refusal":
            detail = getattr(response, "stop_details", None)
            raise PlatformError(
                f"the model declined this request"
                + (f" ({detail.category})" if detail and detail.category else "")
            )
        if response.stop_reason == "max_tokens":
            raise PlatformError(
                "the model hit max_tokens before completing the answer; raise DP_MAX_TOKENS"
            )

        text = next((b.text for b in response.content if b.type == "text"), "")
        if not text.strip():
            raise PlatformError("the model returned no text block")
        try:
            return output_model.model_validate(json.loads(text))
        except (json.JSONDecodeError, ValidationError) as exc:
            raise PlatformError(f"could not parse the model's structured output: {exc}") from exc

    def text(
        self,
        *,
        instructions: str,
        prompt: str,
        max_tokens: int | None = None,
        effort: str | None = None,
    ) -> str:
        """Free-form prose (narratives, summaries)."""
        response = self._create(
            max_tokens=max_tokens,
            system=instructions,
            output_config={"effort": effort or self.effort},
            messages=[{"role": "user", "content": prompt}],
        )
        if response.stop_reason == "refusal":
            raise PlatformError("the model declined this request")
        return "\n".join(b.text for b in response.content if b.type == "text").strip()


def available() -> bool:
    if not settings.has_llm:
        return False
    try:
        import anthropic  # noqa: F401
    except ImportError:
        return False
    return True
