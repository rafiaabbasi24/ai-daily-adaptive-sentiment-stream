"""Integration tests for the FastAPI application.

These tests verify that the main API endpoints are reachable and return
the expected data structures. They use the FastAPI ``TestClient`` to
exercise the application without needing to run an external server.
"""

from __future__ import annotations

import json
from typing import Any, Dict

import pytest
from fastapi.testclient import TestClient

# The FastAPI app is expected to be defined in ``app/main.py`` as ``app``.
# If the project uses a different module layout, adjust the import accordingly.
try:
    from app.main import app  # type: ignore
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "Could not import the FastAPI app. Ensure that the project defines "
        "`app` in `app/main.py`."
    ) from exc


@pytest.fixture(scope="module")
def client() -> TestClient:
    """Create a single TestClient instance for the whole test module."""
    return TestClient(app)


def test_health_endpoint(client: TestClient) -> None:
    """The health endpoint should return a 200 status and a simple JSON payload."""
    response = client.get("/health")
    assert response.status_code == 200, "Health endpoint did not return HTTP 200"
    data: Dict[str, Any] = response.json()
    # The exact schema may vary; we enforce the presence of a ``status`` key.
    assert isinstance(data, dict), "Health response is not a JSON object"
    assert data.get("status") == "ok", "Health status should be 'ok'"


def test_prediction_endpoint_success(client: TestClient) -> None:
    """
    Test a successful sentiment prediction request.

    The endpoint is expected to accept a JSON body with at least the fields:
    - ``text``: the raw text to analyze.
    - ``lang``: ISO‑639 language code (e.g., ``en`` for English).

    The response should contain:
    - ``sentiment``: a string label (e.g., ``positive``, ``negative``, ``neutral``).
    - ``confidence``: a float between 0 and 1.
    """
    payload = {
        "text": "I love this product! It works great.",
        "lang": "en",
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200, f"Predict endpoint returned {response.status_code}"
    result: Dict[str, Any] = response.json()

    # Basic schema validation
    assert isinstance(result, dict), "Response is not a JSON object"
    assert "sentiment" in result, "Missing 'sentiment' in response"
    assert "confidence" in result, "Missing 'confidence' in response"

    # Type checks
    assert isinstance(result["sentiment"], str), "'sentiment' should be a string"
    assert isinstance(result["confidence"], (float, int)), "'confidence' should be numeric"

    # Value checks
    confidence = float(result["confidence"])
    assert 0.0 <= confidence <= 1.0, "'confidence' must be between 0 and 1"


def test_prediction_endpoint_missing_fields(client: TestClient) -> None:
    """
    The API should return a 422 validation error when required fields are missing.
    """
    # Omit the ``lang`` field
    payload = {"text": "Este es un texto sin idioma."}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422, "Expected 422 for missing fields"


def test_prediction_endpoint_unsupported_language(client: TestClient) -> None:
    """
    When an unsupported language code is supplied, the endpoint should return a
    400 Bad Request with a helpful error message.
    """
    payload = {"text": "Bonjour le monde!", "lang": "xx"}  # 'xx' is not a real ISO‑639 code
    response = client.post("/predict", json=payload)
    assert response.status_code == 400, "Expected 400 for unsupported language"
    error_body = response.json()
    assert isinstance(error_body, dict), "Error response should be JSON"
    assert "detail" in error_body, "Error response missing 'detail' field"


def test_stream_endpoint(client: TestClient) -> None:
    """
    Verify that the streaming endpoint returns a server‑sent events (SSE) stream.

    The endpoint ``/stream`` is expected to keep the connection open and emit
    JSON lines. For testing purposes we request a limited number of events
    by passing a ``max_events`` query parameter (implementation‑specific).
    """
    # Request a short stream to keep the test fast.
    params = {"max_events": 2}
    with client.stream("GET", "/stream", params=params) as response:
        assert response.status_code == 200, "Stream endpoint did not return HTTP 200"
        # The content type for SSE is usually 'text/event-stream'.
        content_type = response.headers.get("content-type", "")
        assert "text/event-stream" in content_type, "Expected SSE content type"

        # Collect the first two events.
        events = []
        for line in response.iter_lines():
            if not line:
                continue  # Skip heartbeat lines
            # SSE payloads are prefixed with "data: ".
            decoded = line.decode("utf-8")
            if decoded.startswith("data:"):
                json_payload = decoded.removeprefix("data:").strip()
                try:
                    events.append(json.loads(json_payload))
                except json.JSONDecodeError:
                    pytest.fail(f"Invalid JSON in SSE payload: {json_payload}")

            if len(events) >= 2:
                break

        assert len(events) == 2, "Did not receive the expected number of stream events"
        for event in events:
            assert isinstance(event, dict), "Each stream event should be a JSON object"
            # Expected keys may include 'text', 'sentiment', 'confidence'.
            assert "sentiment" in event, "Stream event missing 'sentiment'"
            assert "confidence" in event, "Stream event missing 'confidence'"