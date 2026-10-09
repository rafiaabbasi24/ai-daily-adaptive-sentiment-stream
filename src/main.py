import asyncio
import json
import os
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field, validator
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from streamz import Stream

# --------------------------------------------------------------------------- #
# Sentiment analysis utilities
# --------------------------------------------------------------------------- #
class SentimentAnalyzer:
    """
    Wrapper around a scikit‑learn pipeline that performs multilingual sentiment
    analysis.  The pipeline consists of a TF‑IDF vectorizer followed by a
    logistic‑regression classifier.  If a pretrained model file is present it is
    loaded; otherwise a tiny dummy model is trained on a few hard‑coded examples.
    """

    _MODEL_PATH = Path(__file__).parent / "sentiment_model.pkl"

    def __init__(self) -> None:
        if self._MODEL_PATH.exists():
            self.pipeline: Pipeline = pd.read_pickle(self._MODEL_PATH)  # type: ignore
        else:
            self.pipeline = self._train_dummy_model()
            # Persist the model for future runs
            pd.to_pickle(self.pipeline, self._MODEL_PATH)

        # Mapping from numeric class to human readable label
        self._label_map = {0: "negative", 1: "neutral", 2: "positive"}

    @staticmethod
    def _train_dummy_model() -> Pipeline:
        """
        Trains a very small model on handcrafted examples.  This is only used
        when a real model is not provided; in production you would replace this
        with a proper multilingual pretrained model.
        """
        examples = [
            ("I love this product!", 2),
            ("This is terrible.", 0),
            ("It is okay, not great.", 1),
            ("¡Me encanta!", 2),
            ("No me gusta nada.", 0),
            ("C'est moyen.", 1),
            ("Das ist fantastisch!", 2),
            ("Ich mag das nicht.", 0),
        ]
        texts, labels = zip(*examples)

        vectorizer = TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 2),
            max_features=5000,
            lowercase=True,
        )
        classifier = LogisticRegression(max_iter=200, n_jobs=1)

        pipeline = Pipeline([("tfidf", vectorizer), ("clf", classifier)])
        pipeline.fit(list(texts), list(labels))
        return pipeline

    def predict(self, text: str) -> Dict[str, Any]:
        """
        Predict sentiment for a single piece of text.

        Returns
        -------
        dict
            ``label`` – human readable sentiment,
            ``confidence`` – probability of the predicted class.
        """
        if not text:
            raise ValueError("Input text must be non‑empty")

        probs = self.pipeline.predict_proba([text])[0]
        pred_idx = int(probs.argmax())
        label = self._label_map.get(pred_idx, "unknown")
        confidence = float(probs[pred_idx])

        return {"label": label, "confidence": confidence}


# --------------------------------------------------------------------------- #
# FastAPI request models
# --------------------------------------------------------------------------- #
class PredictRequest(BaseModel):
    """
    Request body for the ``/predict`` endpoint.
    """

    text: str = Field(..., min_length=1, description="Text to analyse")
    language: str = Field(
        default="auto",
        description="ISO‑639‑1 language code or 'auto' to let the model guess",
    )

    @validator("language")
    def _validate_language(cls, v: str) -> str:
        if not isinstance(v, str):
            raise ValueError("language must be a string")
        return v.lower()


class PredictResponse(BaseModel):
    """
    Response model for sentiment predictions.
    """

    label: str
    confidence: float


class StatsResponse(BaseModel):
    """
    Aggregated statistics about processed messages.
    """

    total_messages: int
    sentiment_counts: Dict[str, int]


# --------------------------------------------------------------------------- #
# Stream orchestration
# --------------------------------------------------------------------------- #
class StreamOrchestrator:
    """
    Manages a Streamz pipeline that receives raw texts, runs sentiment analysis,
    and updates in‑memory statistics.
    """

    def __init__(self, analyzer: SentimentAnalyzer) -> None:
        self.analyzer = analyzer
        self._stats_df = pd.DataFrame(columns=["label"])
        self._lock = asyncio.Lock()

        # Build the stream: source -> predict -> update stats
        self.source: Stream = Stream()
        self.source.map(self._predict).sink(self._update_stats)

    async def push(self, text: str) -> None:
        """
        Push a new text into the stream.  This method is ``async`` because the
        underlying Streamz ``emit`` method can be awaited when the stream runs
        in asynchronous mode.
        """
        await self.source.emit(text)

    def _predict(self, text: str) -> Dict[str, Any]:
        """
        Synchronous helper used inside the Streamz pipeline.  It returns a dict
        containing the original text and the prediction result.
        """
        try:
            result = self.analyzer.predict(text)
        except Exception as exc:
            result = {"label": "error", "confidence": 0.0}
        return {"text": text, "label": result["label"], "confidence": result["confidence"]}

    def _update_stats(self, result: Dict[str, Any]) -> None:
        """
        Update the in‑memory ``pandas`` DataFrame with the new prediction.
        This method runs in the Streamz worker thread, therefore we guard the
        DataFrame with an ``asyncio`` lock to avoid race conditions when the
        FastAPI handlers read the statistics.
        """
        # ``asyncio`` locks cannot be used directly in a non‑async context,
        # so we schedule the update on the event loop.
        loop = asyncio.get_event_loop()
        loop.create_task(self._async_update(result))

    async def _async_update(self, result: Dict[str, Any]) -> None:
        async with self._lock:
            self._stats_df = pd.concat(
                [self._stats_df, pd.DataFrame([{"label": result["label"]}])],
                ignore_index=True,
            )

    async def get_stats(self) -> StatsResponse:
        """
        Return aggregated statistics about processed messages.
        """
        async with self._lock:
            total = len(self._stats_df)
            counts = (
                self._stats_df["label"]
                .value_counts()
                .to_dict()
                if total > 0
                else {}
            )
        return StatsResponse(total_messages=total, sentiment_counts=counts)


# --------------------------------------------------------------------------- #
# FastAPI application definition
# --------------------------------------------------------------------------- #
app = FastAPI(
    title="Adaptive Real‑time Multilingual Sentiment Stream",
    description=(
        "A FastAPI service that receives text streams, performs multilingual "
        "sentiment analysis in real time, and provides aggregated statistics."
    ),
    version="1.0.0",
)

# Global objects – instantiated once at import time
_analyzer = SentimentAnalyzer()
_orchestrator = StreamOrchestrator(_analyzer)


@app.post(
    "/predict",
    response_model=PredictResponse,
    summary="Predict sentiment for a single piece of text",
)
async def predict_endpoint(payload: PredictRequest) -> PredictResponse:
    """
    Synchronous prediction endpoint.  The request is also forwarded to the
    background Streamz pipeline so that it contributes to the live statistics.
    """
    # Immediate prediction for the client
    try:
        result = _analyzer.predict(payload.text)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    # Forward to the streaming pipeline (fire‑and‑forget)
    asyncio.create_task(_orchestrator.push(payload.text))

    return PredictResponse(label=result["label"], confidence=result["confidence"])


@app.get(
    "/stats",
    response_model=StatsResponse,
    summary="Retrieve aggregated sentiment statistics",
)
async def stats_endpoint() -> StatsResponse:
    """
    Returns the number of processed messages and a breakdown per sentiment
    label.  The statistics are updated asynchronously by the Streamz pipeline.
    """
    return await _orchestrator.get_stats()


@app.get("/", include_in_schema=False)
async def root() -> Dict[str, str]:
    """
    Simple health‑check endpoint.
    """
    return {"message": "Adaptive Real‑time Multilingual Sentiment Stream is running"}


# --------------------------------------------------------------------------- #
# Application entry point
# --------------------------------------------------------------------------- #
def _run() -> None:
    """
    Starts the Uvicorn server.  This function is separated from the ``if
    __name__ == "__main__"`` block to make the module import‑friendly for
    testing.
    """
    uvicorn.run(
        "src.main:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "8000")),
        log_level="info",
        reload=False,
    )


if __name__ == "__main__":
    _run()