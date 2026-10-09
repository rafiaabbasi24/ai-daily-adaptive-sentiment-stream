"""src/model.py
Sentiment model wrapper with online learning and drift detection.

The module provides a production‑ready ``SentimentModel`` class that
encapsulates:

* A TF‑IDF vectorizer for multilingual text.
* An ``SGDClassifier`` (logistic regression) that supports ``partial_fit``
  for online updates.
* A lightweight Page‑Hinkley drift detector that monitors the
  log‑loss of incoming batches and raises a flag when a statistically
  significant increase is observed.

The class is deliberately framework‑agnostic – it can be used directly
from a FastAPI endpoint, a Streamz pipeline or any custom ingestion loop.

Example
-------
>>> from src.model import SentimentModel
>>> model = SentimentModel()
>>> # Fit on an initial labelled batch (list of dicts or pandas DataFrame)
>>> model.fit_initial(initial_data)
>>> # Predict on new texts
>>> preds = model.predict(["I love this!", "C'est mauvais."])
>>> # Update model with newly labelled data and check for drift
>>> model.partial_fit(new_batch)
>>> if model.drift_detected:
...     print("Concept drift detected – consider re‑training.")
"""

from __future__ import annotations

import collections
import logging
import math
from typing import Any, Iterable, List, Mapping, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import log_loss
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class PageHinkley:
    """
    Simple Page‑Hinkley drift detector.

    The algorithm monitors a stream of loss values ``x_t`` and raises a
    drift flag when the cumulative sum of deviations from the running mean
    exceeds a predefined threshold.

    Parameters
    ----------
    delta : float, default 0.005
        Small positive value to offset the mean (prevents false alarms on
        low‑variance streams).
    lambda_ : float, default 50.0
        Threshold for the cumulative sum; larger values make the detector
        less sensitive.
    alpha : float, default 0.999
        Forgetting factor for the exponential moving average of the loss.
    """

    def __init__(self, delta: float = 0.005, lambda_: float = 50.0, alpha: float = 0.999) -> None:
        self.delta = delta
        self.lambda_ = lambda_
        self.alpha = alpha
        self._reset()

    def _reset(self) -> None:
        self._mean = 0.0
        self._cum_sum = 0.0
        self._num_samples = 0
        self.drift = False

    def update(self, loss: float) -> bool:
        """
        Feed a new loss value to the detector.

        Parameters
        ----------
        loss : float
            The loss (e.g., log‑loss) observed on the latest batch.

        Returns
        -------
        bool
            ``True`` if drift is detected, otherwise ``False``.
        """
        self._num_samples += 1
        # Exponential moving average of the loss
        self._mean = self.alpha * self._mean + (1 - self.alpha) * loss
        # Cumulative sum of deviations
        self._cum_sum += loss - self._mean - self.delta
        # Keep the minimum cumulative sum to detect upward shifts
        self._cum_sum = max(0.0, self._cum_sum)

        if self._cum_sum > self.lambda_:
            self.drift = True
            logger.warning(
                "Page-Hinkley drift detected (cum_sum=%.3f > lambda=%.3f).",
                self._cum_sum,
                self.lambda_,
            )
            self._reset()  # Reset after detection
            return True

        self.drift = False
        return False


class SentimentModel:
    """
    Wrapper for an online multilingual sentiment classifier with drift detection.

    The model uses a TF‑IDF representation of the input text and an
    ``SGDClassifier`` trained with ``log`` loss (i.e., logistic regression).
    It supports incremental updates via ``partial_fit`` and monitors the
    log‑loss of each batch to detect concept drift.

    Attributes
    ----------
    vectorizer : TfidfVectorizer
        Converts raw text into a sparse TF‑IDF matrix.
    classifier : SGDClassifier
        Linear model trained with stochastic gradient descent.
    label_encoder : LabelEncoder
        Maps string labels to integer indices.
    drift_detector : PageHinkley
        Detects sudden increases in loss indicative of concept drift.
    drift_detected : bool
        ``True`` if the most recent ``partial_fit`` call triggered drift.
    """

    def __init__(
        self,
        *,
        max_features: int = 50000,
        ngram_range: Tuple[int, int] = (1, 2),
        min_df: int = 2,
        loss: str = "log",
        alpha: float = 1e-4,
        random_state: int = 42,
    ) -> None:
        """
        Initialise the sentiment model.

        Parameters
        ----------
        max_features : int, optional
            Maximum number of features for the TF‑IDF vectorizer.
        ngram_range : tuple, optional
            The lower and upper boundary of the n‑gram range.
        min_df : int, optional
            Minimum document frequency for a term to be kept.
        loss : str, optional
            Loss function for ``SGDClassifier`` (default ``'log'``).
        alpha : float, optional
            Regularisation term for the classifier.
        random_state : int, optional
            Random seed for reproducibility.
        """
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=min_df,
            tokenizer=self._simple_tokenizer,
            lowercase=True,
        )
        self.classifier = SGDClassifier(
            loss=loss,
            penalty="l2",
            alpha=alpha,
            max_iter=5,
            random_state=random_state,
            warm_start=True,
        )
        self.label_encoder = LabelEncoder()
        self.drift_detector = PageHinkley()
        self.drift_detected = False
        self._is_fitted = False

    @staticmethod
    def _simple_tokenizer(text: str) -> List[str]:
        """
        Very light‑weight tokenizer that splits on whitespace and strips punctuation.

        It works reasonably well for many languages without requiring heavy
        external dependencies.

        Parameters
        ----------
        text : str
            Input sentence.

        Returns
        -------
        List[str]
            Token list.
        """
        # Basic punctuation removal
        cleaned = "".join(ch if ch.isalnum() or ch.isspace() else " " for ch in text)
        return cleaned.split()

    def _prepare_batch(
        self, data: Union[pd.DataFrame, Sequence[Mapping[str, Any]]]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert raw input into feature matrix and label vector.

        Parameters
        ----------
        data : pandas.DataFrame or iterable of mappings
            Must contain at least ``'text'`` and ``'label'`` columns/keys.

        Returns
        -------
        X : scipy.sparse.csr_matrix
            TF‑IDF feature matrix.
        y : np.ndarray
            Integer‑encoded label vector.
        """
        if isinstance(data, pd.DataFrame):
            texts = data["text"].astype(str).tolist()
            labels = data["label"].astype(str).tolist()
        else:
            # Assume iterable of dict‑like objects
            texts = [str(item["text"]) for item in data]
            labels = [str(item["label"]) for item in data]

        if not self._is_fitted:
            # Fit vectorizer on the first batch
            X = self.vectorizer.fit_transform(texts)
        else:
            X = self.vectorizer.transform(texts)

        # Encode labels – fit on first batch, then transform
        if not self._is_fitted:
            y = self.label_encoder.fit_transform(labels)
        else:
            # ``LabelEncoder`` does not support unseen labels; raise a clear error
            unseen = set(labels) - set(self.label_encoder.classes_)
            if unseen:
                raise ValueError(f"Encountered unseen labels during update: {unseen}")
            y = self.label_encoder.transform(labels)

        return X, y

    def fit_initial(
        self,
        data: Union[pd.DataFrame, Sequence[Mapping[str, Any]]],
        classes: Sequence[str] | None = None,
    ) -> None:
        """
        Fit the model on an initial labelled batch.

        This method must be called before any ``predict`` or ``partial_fit``
        operation.

        Parameters
        ----------
        data : pandas.DataFrame or iterable of mappings
            Training data containing ``'text'`` and ``'label'``.
        classes : sequence of str, optional
            Explicit list of all possible class names. If omitted, the classes
            are inferred from ``data``.
        """
        X, y = self._prepare_batch(data)
        if classes is None:
            classes = self.label_encoder.classes_
        self.classifier.partial_fit(X, y, classes=np.arange(len(classes)))
        self._is_fitted = True
        logger.info("Initial model fitted on %d samples.", X.shape[0])

    def predict(self, texts: Sequence[str]) -> List[str]:
        """
        Predict sentiment labels for a list of raw texts.

        Parameters
        ----------
        texts : sequence of str
            Input sentences.

        Returns
        -------
        List[str]
            Predicted class names.
        """
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before calling predict().")
        X = self.vectorizer.transform(texts)
        pred_idx = self.classifier.predict(X)
        return self.label_encoder.inverse_transform(pred_idx).tolist()

    def predict_proba(self, texts: Sequence[str]) -> np.ndarray:
        """
        Return class probabilities for the given texts.

        Parameters
        ----------
        texts : sequence of str

        Returns
        -------
        np.ndarray
            Array of shape (n_samples, n_classes) with probability estimates.
        """
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before calling predict_proba().")
        X = self.vectorizer.transform(texts)
        if hasattr(self.classifier, "predict_proba"):
            return self.classifier.predict_proba(X)
        # SGDClassifier with 'log' loss provides decision_function; convert via sigmoid
        decision = self.classifier.decision_function(X)
        if decision.ndim == 1:
            # Binary case
            prob_pos = 1 / (1 + np.exp(-decision))
            return np.vstack([1 - prob_pos, prob_pos]).T
        else:
            # Multiclass – apply softmax
            exp_logits = np.exp(decision - np.max(decision, axis=1, keepdims=True))
            return exp_logits / exp_logits.sum(axis=1, keepdims=True)

    def partial_fit(
        self,
        data: Union[pd.DataFrame, Sequence[Mapping[str, Any]]],
    ) -> None:
        """
        Incrementally update the model with a new labelled batch and check for drift.

        Parameters
        ----------
        data : pandas.DataFrame or iterable of mappings
            New training samples containing ``'text'`` and ``'label'``.
        """
        if not self._is_fitted:
            raise RuntimeError("Call ``fit_initial`` before ``partial_fit``.")
        X, y = self._prepare_batch(data)

        # Compute loss before the update for drift detection
        probas = self.predict_proba([item["text"] for item in data] if not isinstance(data, pd.DataFrame) else data["text"].astype(str).tolist())
        loss = log_loss(y, probas, labels=np.arange(len(self.label_encoder.classes_)))
        self.drift_detected = self.drift_detector.update(loss)

        # Perform the incremental update
        self.classifier.partial_fit(X, y)
        logger.info(
            "Model updated with %d samples; loss=%.4f; drift=%s",
            X.shape[0],
            loss,
            self.drift_detected,
        )

    def stream_process(
        self,
        source: Any,
        batch_size: int = 32,
        *,
        max_batches: int | None = None,
    ) -> Any:
        """
        Convenience wrapper to connect a Streamz source to the model.

        The method yields dictionaries ``{'texts': [...], 'predictions': [...]}``
        for each processed batch. It also updates the model when a ``'label'``
        field is present.

        Parameters
        ----------
        source : Streamz Stream
            The upstream stream yielding dictionaries with at least a ``'text'``
            key. Optional ``'label'`` key triggers an online update.
        batch_size : int, default 32
            Number of records to accumulate before processing.
        max_batches : int or None, optional
            Stop after this many batches (useful for testing).

        Yields
        ------
        dict
            Mapping with keys ``'texts'`` and ``'predictions'`` (and optionally
            ``'drift'``).
        """
        from streamz import Stream

        if not isinstance(source, Stream):
            raise TypeError("source must be a streamz.Stream instance")

        buffer: List[Mapping[str, Any]] = []

        def _process_batch(batch: List[Mapping[str, Any]]) -> dict:
            texts = [str(item["text"]) for item in batch]
            preds = self.predict(texts)
            result = {"texts": texts, "predictions": preds}
            # If labels are present, perform an online update
            if any("label" in item for item in batch):
                self.partial_fit(batch)
                result["drift"] = self.drift_detected
            return result

        # Build the streaming pipeline
        stream = (
            source.partition(batch_size)
            .map(_process_batch)
        )

        # Consume the stream and yield results
        count = 0
        for out in stream:
            yield out
            count += 1
            if max_batches is not None and count >= max_batches:
                break

    def save(self, path: str) -> None:
        """
        Persist the model, vectorizer and label encoder to disk.

        Parameters
        ----------
        path : str
            Destination directory. The method creates three files:
            ``vectorizer.pkl``, ``classifier.pkl`` and ``label_encoder.pkl``.
        """
        import joblib
        import os

        os.makedirs(path, exist_ok=True)
        joblib.dump(self.vectorizer, os.path.join(path, "vectorizer.pkl"))
        joblib.dump(self.classifier, os.path.join(path, "classifier.pkl"))
        joblib.dump(self.label_encoder, os.path.join(path, "label_encoder.pkl"))
        logger.info("Model saved to %s", path)

    def load(self, path: str) -> None:
        """
        Load a previously saved model from ``path``.

        Parameters
        ----------
        path : str
            Directory containing the three ``*.pkl`` files created by ``save``.
        """
        import joblib
        import os

        self.vectorizer = joblib.load(os.path.join(path, "vectorizer.pkl"))
        self.classifier = joblib.load(os.path.join(path, "classifier.pkl"))
        self.label_encoder = joblib.load(os.path.join(path, "label_encoder.pkl"))
        self._is_fitted = True
        logger.info("Model loaded from %s", path)