import httpx
import pytest
from fastapi import HTTPException
from openai import APIStatusError, RateLimitError

from budget_agent.service import _guard, _is_transient_upstream


def _resp(status: int) -> httpx.Response:
    return httpx.Response(status, request=httpx.Request("POST", "http://openai"))


def test_is_transient_upstream_only_true_for_429_503():
    def _err(status: int) -> httpx.HTTPStatusError:
        r = httpx.Response(status, request=httpx.Request("GET", "http://agg/accounts"))
        return httpx.HTTPStatusError("boom", request=r.request, response=r)

    assert _is_transient_upstream(_err(429)) is True
    assert _is_transient_upstream(_err(503)) is True
    assert _is_transient_upstream(_err(500)) is False
    assert _is_transient_upstream(ValueError("nope")) is False


def test_guard_maps_rate_limit_to_retryable_429():
    # A 429 from Azure OpenAI (deployment over its TPM budget) must surface as a
    # clear, retryable 429 — not an opaque 500 that reads as "Upstream 500".
    def boom():
        raise RateLimitError("rate limit exceeded", response=_resp(429), body=None)

    with pytest.raises(HTTPException) as ei:
        _guard(boom)
    assert ei.value.status_code == 429
    assert "try again" in ei.value.detail.lower()


def test_guard_maps_openai_status_error_to_502():
    def boom():
        raise APIStatusError("server error", response=_resp(500), body=None)

    with pytest.raises(HTTPException) as ei:
        _guard(boom)
    assert ei.value.status_code == 502
