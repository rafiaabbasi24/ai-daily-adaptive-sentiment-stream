"""tests/test_model.py
Unit tests for the Adaptive Real-time Multilingual Sentiment Stream project.

These tests verify that the sentiment model can be instantiated, produces
reasonable predictions on a small multilingual sample, and that the drift
detector reacts to a simulated distribution shift.

The tests are written with ``pytest`` and rely only on publicly available
packages and the project's own modules.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

# Import the objects under test.  The package name ``adaptive_sentiment`` is
# assumed to be the top‑level module of the project.
from adaptive_sentiment.model import SentimentModel
from adaptive_sentiment.drift import DriftDetector


@pytest.fixture(scope="module")
def sample_dataframe() -> pd.DataFrame:
    """Create a tiny multilingual dataset for quick inference tests.

    Returns
    -------
    pd.DataFrame
        A DataFrame with two columns:
        - ``text``: raw sentences in English, Spanish and French.
        - ``lang``: ISO‑639‑1 language code (used by the model's tokenizer).
    """
    data = {
        "text": [
            "I love this product!",               # English – positive
            "Este producto es terrible.",         # Spanish – negative
            "C'est un excellent service.",        # French – positive
            "Je n'aime pas ce film.",              # French – negative
        ],
        "lang": ["en", "es", "fr", "fr"],
    }
    return pd.DataFrame(data)


@pytest.fixture(scope="module")
def sentiment_model() -> SentimentModel:
    """Instantiate the SentimentModel with deterministic weights.

    The model is loaded in evaluation mode and placed on the CPU to keep the
    test environment lightweight.
    """
    model = SentimentModel()
    model.eval()                     # Ensure deterministic behaviour
    model.to(torch.device("cpu"))    # Force CPU usage for CI environments
    return model


def test_model_prediction_shape_and_range(
    sentiment_model: SentimentModel, sample_dataframe: pd.DataFrame
) -> None:
    """Check that the model returns a probability vector for each input.

    The model's ``predict`` method is expected to return a NumPy array of shape
    ``(n_samples, n_classes)`` with values in the interval ``[0, 1]`` that sum
    to 1 across the class dimension.
    """
    # Run inference
    probs = sentiment_model.predict(sample_dataframe["text"], sample_dataframe["lang"])

    # Basic type / shape checks
    assert isinstance(probs, np.ndarray), "Prediction output must be a NumPy array"
    assert probs.shape == (len(sample_dataframe), 2), (
        f"Expected shape {(len(sample_dataframe), 2)}, got {probs.shape}"
    )

    # Probability constraints
    assert np.all(probs >= 0.0) and np.all(probs <= 1.0), "Probabilities must be in [0, 1]"
    row_sums = probs.sum(axis=1)
    assert np.allclose(row_sums, 1.0, atol=1e-5), "Each row must sum to 1"


def test_model_predict_returns_consistent_labels(
    sentiment_model: SentimentModel, sample_dataframe: pd.DataFrame
) -> None:
    """Ensure that ``predict_labels`` (if implemented) aligns with argmax of probabilities.

    This test guards against mismatched label encoding between the two
    inference entry points.
    """
    probs = sentiment_model.predict(sample_dataframe["text"], sample_dataframe["lang"])
    predicted_labels = sentiment_model.predict_labels(
        sample_dataframe["text"], sample_dataframe["lang"]
    )

    # The label should be the index of the max probability for each sample.
    expected_labels = probs.argmax(axis=1)
    np.testing.assert_array_equal(
        predicted_labels,
        expected_labels,
        err_msg="Label predictions do not match argmax of probability output",
    )


@pytest.fixture
def drift_detector() -> DriftDetector:
    """Create a fresh DriftDetector with default parameters."""
    return DriftDetector()


def test_drift_detector_no_drift_on_stationary_data(drift_detector: DriftDetector) -> None:
    """Feed a stationary stream of predictions and verify that no drift is flagged.

    The synthetic data mimics a stable sentiment distribution (e.g., 70% positive,
    30% negative) across 200 timesteps.
    """
    rng = np.random.default_rng(seed=42)
    # Simulate 200 binary predictions (0 = negative, 1 = positive)
    stationary_preds = rng.binomial(1, 0.7, size=200)

    for pred in stationary_preds:
        drift_detector.update(pred)

    assert not drift_detector.drift_detected, "Drift should not be detected on stationary data"


def test_drift_detector_detects_shift(drift_detector: DriftDetector) -> None:
    """Introduce a sudden shift in the prediction distribution and expect drift detection.

    The test first feeds a stable stream (0.7 positive) and then switches to a
    reversed distribution (0.3 positive).  The detector should flag drift shortly
    after the change.
    """
    rng = np.random.default_rng(seed=0)

    # Phase 1: stable distribution (70% positive)
    stable_preds = rng.binomial(1, 0.7, size=150)
    for pred in stable_preds:
        drift_detector.update(pred)

    assert not drift_detector.drift_detected, "Drift should not be detected before the shift"

    # Phase 2: distribution shift (30% positive)
    shifted_preds = rng.binomial(1, 0.3, size=50)
    drift_flagged = False
    for i, pred in enumerate(shifted_preds, start=1):
        drift_detector.update(pred)
        if drift_detector.drift_detected:
            drift_flagged = True
            # Ensure drift is detected within a reasonable number of steps
            assert i <= 20, f"Drift detected too late (after {i} steps)"
            break

    assert drift_flagged, "Drift detector failed to flag the distribution shift"


def test_drift_detector_reset_functionality(drift_detector: DriftDetector) -> None:
    """After a drift is detected, resetting the detector should clear the flag."""
    # Force a drift by feeding contradictory data
    for pred in [0, 1] * 10:
        drift_detector.update(pred)

    # Assume the detector flags drift after the above pattern
    assert drift_detector.drift_detected, "Drift should be detected before reset"

    drift_detector.reset()
    assert not drift_detector.drift_detected, "Drift flag should be cleared after reset"
    # After reset, feeding stationary data should keep the flag cleared
    for pred in [1] * 30:
        drift_detector.update(pred)
    assert not drift_detector.drift_detected, "Drift should remain cleared after reset with stationary data"
