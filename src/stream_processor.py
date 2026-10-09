import logging
import asyncio
from datetime import datetime
from typing import Callable, Iterable, List, Dict, Any, Optional

import pandas as pd
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torchtext.data.utils import get_tokenizer
from streamz import Stream

# Configure a module level logger
logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        fmt="%(asctime)s %(levelname)s %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


def default_tokenizer(text: str) -> List[str]:
    """
    Tokenize a string using the basic English tokenizer from ``torchtext``.
    The function is deliberately simple to keep the pipeline lightweight and
    language‑agnostic (it merely splits on whitespace and punctuation).

    Parameters
    ----------
    text: str
        Raw input text.

    Returns
    -------
    List[str]
        List of token strings.
    """
    tokenizer = get_tokenizer("basic_english")
    return tokenizer(text)


class SimpleSentimentModel(nn.Module):
    """
    A minimal feed‑forward sentiment classifier.

    The model expects an input tensor of shape ``(batch_size, vocab_size)`` that
    contains token counts (i.e., a bag‑of‑words representation).  It outputs a
    single sigmoid‑activated score per sample representing the probability of a
    positive sentiment.

    The architecture is deliberately tiny to keep the example runnable on
    modest hardware.  In a production setting you would replace this with a
    transformer or a more sophisticated architecture.
    """

    def __init__(self, vocab_size: int, hidden_dim: int = 64):
        super().__init__()
        self.fc1 = nn.Linear(vocab_size, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x: torch.Tensor
            Tensor of shape ``(batch_size, vocab_size)`` with token counts.

        Returns
        -------
        torch.Tensor
            Tensor of shape ``(batch_size, 1)`` with probabilities.
        """
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return self.sigmoid(out)


def build_vocab(
    texts: Iterable[str],
    tokenizer: Callable[[str], List[str]] = default_tokenizer,
    min_freq: int = 1,
) -> Dict[str, int]:
    """
    Build a token‑to‑index vocabulary from an iterable of texts.

    Parameters
    ----------
    texts: Iterable[str]
        Collection of raw text strings.
    tokenizer: Callable[[str], List[str]], optional
        Tokenizer function.  Defaults to ``default_tokenizer``.
    min_freq: int, optional
        Minimum frequency a token must have to be kept.  Defaults to ``1``.

    Returns
    -------
    Dict[str, int]
        Mapping from token string to integer index.
    """
    from collections import Counter

    counter = Counter()
    for txt in texts:
        counter.update(tokenizer(txt))

    # Reserve index 0 for unknown tokens
    vocab = {"<UNK>": 0}
    for token, freq in counter.items():
        if freq >= min_freq:
            vocab[token] = len(vocab)
    logger.info("Built vocabulary with %d tokens (including <UNK>)", len(vocab))
    return vocab


def vectorize_batch(
    texts: List[str],
    vocab: Dict[str, int],
    tokenizer: Callable[[str], List[str]] = default_tokenizer,
) -> torch.Tensor:
    """
    Convert a batch of raw texts into a bag‑of‑words tensor.

    Parameters
    ----------
    texts: List[str]
        List of raw text strings.
    vocab: Dict[str, int]
        Token‑to‑index mapping.
    tokenizer: Callable[[str], List[str]], optional
        Tokenizer function.

    Returns
    -------
    torch.Tensor
        Tensor of shape ``(batch_size, vocab_size)`` with integer counts.
    """
    vocab_size = len(vocab)
    batch_tensor = torch.zeros((len(texts), vocab_size), dtype=torch.float32)

    for i, txt in enumerate(texts):
        tokens = tokenizer(txt)
        for token in tokens:
            idx = vocab.get(token, vocab["<UNK>"])
            batch_tensor[i, idx] += 1.0
    return batch_tensor


class StreamProcessor:
    """
    Encapsulates a Streamz pipeline that ingests raw text, preprocesses it,
    runs a sentiment model, and emits a ``pandas.DataFrame`` with predictions.

    Typical usage::

        >>> vocab = build_vocab(initial_corpus)
        >>> model = SimpleSentimentModel(len(vocab))
        >>> processor = StreamProcessor(model, vocab)
        >>> source = Stream()
        >>> pipeline = processor.start(source)
        >>> source.emit({"text": "I love this product!"})
        >>> # The pipeline will eventually produce a DataFrame row.
    """

    def __init__(
        self,
        model: nn.Module,
        vocab: Dict[str, int],
        batch_size: int = 32,
        tokenizer: Callable[[str], List[str]] = default_tokenizer,
        device: Optional[torch.device] = None,
    ):
        """
        Parameters
        ----------
        model: nn.Module
            A PyTorch model that accepts a bag‑of‑words tensor and returns a
            probability score.
        vocab: Dict[str, int]
            Token‑to‑index mapping used for vectorisation.
        batch_size: int, optional
            Number of records to accumulate before a forward pass.  Defaults to
            ``32``.
        tokenizer: Callable[[str], List[str]], optional
            Tokenizer used for preprocessing.  Defaults to ``default_tokenizer``.
        device: torch.device, optional
            Device on which to run the model.  If ``None``, ``cpu`` is used.
        """
        self.model = model.eval()
        self.vocab = vocab
        self.batch_size = batch_size
        self.tokenizer = tokenizer
        self.device = device or torch.device("cpu")
        self.model.to(self.device)

        logger.info(
            "Initialized StreamProcessor with batch_size=%d on device=%s",
            self.batch_size,
            self.device,
        )

    def _predict_batch(self, batch: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Run sentiment inference on a batch of records.

        Parameters
        ----------
        batch: List[Dict[str, Any]]
            Each dict must contain at least a ``'text'`` key.

        Returns
        -------
        pandas.DataFrame
            DataFrame with columns ``timestamp``, ``text`` and ``sentiment``.
        """
        texts = [record["text"] for record in batch]
        timestamps = [
            record.get("timestamp", datetime.utcnow().isoformat()) for record in batch
        ]

        logger.debug("Vectorising %d texts", len(texts))
        bow_tensor = vectorize_batch(texts, self.vocab, self.tokenizer).to(self.device)

        with torch.no_grad():
            scores = self.model(bow_tensor).squeeze(-1).cpu().numpy()

        df = pd.DataFrame(
            {
                "timestamp": timestamps,
                "text": texts,
                "sentiment": scores,
            }
        )
        logger.info("Produced %d prediction rows", len(df))
        return df

    def start(self, source: Stream) -> Stream:
        """
        Attach the processing steps to a Streamz source.

        Parameters
        ----------
        source: streamz.Stream
            The upstream stream that yields dictionaries with a ``'text'`` field.

        Returns
        -------
        streamz.Stream
            The downstream stream that emits ``pandas.DataFrame`` objects.
        """
        # Buffer records until we have ``batch_size`` items or a timeout occurs.
        # ``buffer`` will emit a list of records.
        buffered = source.buffer(self.batch_size)

        # Apply the prediction function to each buffered list.
        processed = buffered.map(self._predict_batch)

        # Flatten the stream so that downstream consumers receive one DataFrame
        # per batch rather than a list of DataFrames.
        flattened = processed.sink(self._log_output)

        logger.info("Stream pipeline constructed")
        return buffered

    @staticmethod
    def _log_output(df: pd.DataFrame) -> None:
        """
        Default sink that logs the DataFrame.  Users can replace the sink with
        custom logic (e.g., persisting to a database).

        Parameters
        ----------
        df: pandas.DataFrame
            DataFrame produced by the pipeline.
        """
        logger.debug("Sink received DataFrame with %d rows", len(df))
        # For demonstration we log the first few rows at INFO level.
        logger.info("\n%s", df.head().to_string(index=False))


# --------------------------------------------------------------------------- #
# Example usage (executed when the module is run directly)
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    # Sample corpus to build a vocabulary
    sample_corpus = [
        "I love this product!",
        "Terrible experience, will not buy again.",
        "It was okay, nothing special.",
        "Absolutely fantastic service.",
    ]

    # Build vocab and instantiate model
    vocab = build_vocab(sample_corpus)
    model = SimpleSentimentModel(len(vocab))

    # Create the processor
    processor = StreamProcessor(model=model, vocab=vocab, batch_size=2)

    # Create a Streamz source
    source_stream = Stream()

    # Attach the pipeline
    processor.start(source_stream)

    # Simulate asynchronous emission of records
    async def emit_examples():
        examples = [
            {"text": "I really enjoyed this!"},
            {"text": "Worst thing ever."},
            {"text": "Mediocre at best."},
            {"text": "Excellent quality, highly recommend."},
        ]
        for rec in examples:
            source_stream.emit(rec)
            await asyncio.sleep(0.1)  # simulate a small delay

    asyncio.run(emit_examples())
