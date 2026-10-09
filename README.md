# Adaptive Real-time Multilingual Sentiment Stream

## Table of Contents
- [Project Overview](#project-overview)
- [Features](#features)
- [Architecture Diagram](#architecture-diagram)
- [Tech Stack](#tech-stack)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Running the Service](#running-the-service)
- [API Endpoints](#api-endpoints)
- [Streaming Workflow](#streaming-workflow)
- [Model Training & Evaluation](#model-training--evaluation)
- [Testing](#testing)
- [Deployment](#deployment)
- [Contributing](#contributing)
- [License](#license)

---

## Project Overview
**Adaptive Real-time Multilingual Sentiment Stream** is a production‑ready system that ingests textual data from live sources (e.g., social media firehose, chat logs), detects the language on‑the‑fly, and produces sentiment predictions in near real‑time. The pipeline is built to be **scalable**, **low‑latency**, and **extensible** to additional languages or custom sentiment models.

Key goals:
- **Multilingual support** (initially English, Spanish, French, German, Mandarin).
- **Adaptive inference** – the model can be swapped or fine‑tuned without downtime.
- **Streaming processing** using `streamz` for back‑pressure handling.
- **RESTful API** for ingesting single messages or bulk batches.
- **Docker‑compatible** for cloud or edge deployments.

---

## Features
- **Language detection** using fastText embeddings.
- **Pre‑trained multilingual BERT** (via `torch`/`torchtext`) fine‑tuned for sentiment classification.
- **Real‑time streaming** with `streamz` and asynchronous FastAPI endpoints.
- **Batch inference** fallback for high‑throughput bursts.
- **Metrics & health checks** exposed via `/metrics` (Prometheus format) and `/health`.
- **Configurable pipelines** via a single `config.yaml` file.
- **Dockerfile** and `docker-compose.yml` for easy orchestration.
- **Comprehensive test suite** with `pytest`.

---

## Architecture Diagram
```
+----------------+      +-------------------+      +-------------------+
|   Data Source  | ---> |   FastAPI Server  | ---> |   Streamz Pipeline |
+----------------+      +-------------------+      +-------------------+
                               |                         |
                               v                         v
                        +------------+            +------------+
                        | Language   |            | Sentiment  |
                        | Detector   |            | Classifier |
                        +------------+            +------------+
                               |                         |
                               v                         v
                        +-----------------------------------+
                        |   PostgreSQL / Redis (optional)   |
                        +-----------------------------------+
```

---

## Tech Stack
| Layer                | Library / Tool                     |
|----------------------|------------------------------------|
| Language Detection   | `fasttext` (pre‑trained lid.176)   |
| NLP Modeling          | `torch`, `torchtext`, `transformers` |
| Data Manipulation     | `pandas`                           |
| Streaming Engine      | `streamz`                          |
| API Framework         | `fastapi`                          |
| ASGI Server           | `uvicorn`                          |
| Testing               | `pytest`, `httpx`                  |
| Containerization      | `Docker`, `docker-compose`         |
| Monitoring            | `prometheus-client`                |

All dependencies are available on PyPI and listed in `requirements.txt`.

---

## Prerequisites
- Python **3.10** or newer
- `git` (to clone the repository)
- Docker & Docker Compose (optional, for containerized deployment)
- Access to a GPU (CUDA 11.8+) for faster inference (CPU fallback supported)

---

## Installation

```bash
# Clone the repository
git clone https://github.com/your-org/adaptive-realtime-multilingual-sentiment-stream.git
cd adaptive-realtime-multilingual-sentiment-stream

# Create a virtual environment
python -m venv .venv
source .venv/bin/activate   # On Windows: .venv\Scripts\activate

# Install Python dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Download the fastText language detection model (≈ 126 MB)
mkdir -p models
wget -O models/lid.176.bin https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin
```

*If you plan to run inside Docker, skip the virtual‑env steps and use the provided `Dockerfile`.*

---

## Configuration
All runtime settings live in `config.yaml`. A minimal example:

```yaml
api:
  host: 0.0.0.0
  port: 8000
  log_level: info

stream:
  max_batch_size: 64
  max_latency_ms: 200

model:
  name: "bert-base-multilingual-cased"
  checkpoint_path: "models/sentiment_bert.pt"
  device: "cuda"   # set to "cpu" if no GPU is available

language_detector:
  model_path: "models/lid.176.bin"

metrics:
  enabled: true
  port: 9100
```

Edit the file to match your environment (e.g., change `device` to `"cpu"` on a CPU‑only host).

---

## Running the Service

### Local Development
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### Production (Gunicorn + Uvicorn Workers)
```bash
gunicorn app.main:app \
  -k uvicorn.workers.UvicornWorker \
  --workers 4 \
  --bind 0.0.0.0:8000 \
  --log-level info
```

### Docker Compose
```bash
docker compose up --build -d
```

The API will be reachable at `http://localhost:8000`.

---

## API Endpoints

| Method | Path | Description | Request Body | Response |
|--------|------|-------------|--------------|----------|
| `POST` | `/predict` | Predict sentiment for a single text string. | `{ "text": "Your message" }` | `{ "language": "en", "sentiment": "positive", "score": 0.92 }` |
| `POST` | `/batch_predict` | Predict sentiment for a list of texts (max `max_batch_size`). | `{ "texts": ["msg1", "msg2", ...] }` | `{ "results": [{...}, {...}] }` |
| `GET`  | `/health` | Liveness probe. Returns `200 OK` if the service is up. | – | `{ "status": "healthy" }` |
| `GET`  | `/metrics` | Prometheus‑compatible metrics (if enabled). | – | Plain‑text metrics |

All endpoints return JSON and validate input with Pydantic models. Errors are reported with standard HTTP status codes and a descriptive JSON payload.

---

## Streaming Workflow

1. **Ingestion** – Clients push messages to `/predict` or `/batch_predict`.  
2. **FastAPI** forwards the payload to a `streamz` source node.  
3. **Language Detection** – The first stream operator runs fastText on the raw text.  
4. **Batching** – Messages are grouped respecting `max_batch_size` and `max_latency_ms`.  
5. **Model Inference** – Batched tensors are passed to the multilingual BERT classifier on the configured device.  
6. **Post‑processing** – Probabilities are converted to human‑readable labels (`positive`, `neutral`, `negative`).  
7. **Response** – Results are emitted back to the originating request context and optionally persisted.

The pipeline is fully asynchronous; back‑pressure is automatically applied when downstream operators become saturated.

---

## Model Training & Evaluation

### Training Script
```bash
python scripts/train.py \
  --train-data data/train.csv \
  --val-data data/val.csv \
  --model-name bert-base-multilingual-cased \
  --output-dir models/ \
  --epochs 5 \
  --batch-size 32 \
  --learning-rate 2e-5 \
  --device cuda
```

- **Input format**: CSV with columns `text`, `language`, `sentiment` (`positive|neutral|negative`).  
- The script tokenizes with `torchtext`, fine‑tunes the BERT model, and saves `sentiment_bert.pt` in the `models/` directory.

### Evaluation
```bash
python scripts/evaluate.py \
  --test-data data/test.csv \
  --model-path models/sentiment_bert.pt \
  --device cpu
```

The script prints a classification report (precision, recall, F1) per language and overall.

---

## Testing

Run the full test suite with:

```bash
pytest -vv
```

Key test modules:
- `tests/test_api.py` – integration tests using `httpx.AsyncClient`.
- `tests/test_pipeline.py` – unit tests for language detection and batching logic.
- `tests/test_model.py` – sanity checks on the BERT classifier output shapes.

All tests are deterministic and do not require external services (mocked FastAPI requests and a small in‑memory dataset are used).

---

## Deployment

The project can be deployed to any container‑orchestrated environment (Kubernetes, ECS, Cloud Run). Recommended steps:

1. **Build the Docker image**  
   ```bash
   docker build -t your-registry/adaptive-sentiment:latest .
   ```

2. **Push to a container registry**  
   ```bash
   docker push your-registry/adaptive-sentiment:latest
   ```

3. **Create a Kubernetes Deployment** (example `deployment.yaml`):
   ```yaml
   apiVersion: apps/v1
   kind: Deployment
   metadata:
     name: sentiment-stream
   spec:
     replicas: 3
     selector:
       matchLabels:
         app: sentiment-stream
     template:
       metadata:
         labels:
           app: sentiment-stream
       spec:
         containers:
           - name: app
             image: your-registry/adaptive-sentiment:latest
             ports:
               - containerPort: 8000
             env:
               - name: CONFIG_PATH
                 value: "/app/config.yaml"
             volumeMounts:
               - name: config
                 mountPath: /app/config.yaml
                 subPath: config.yaml
         volumes:
           - name: config
             configMap:
               name: sentiment-config
   ```

4. **Expose via a Service** and optionally an Ingress for external traffic.

Monitoring can be enabled by scraping the `/metrics` endpoint with Prometheus and visualizing with Grafana.

---

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository.
2. Create a feature branch (`git checkout -b feature/awesome-thing`).
3. Write code **with tests** and ensure `pytest` passes.
4. Update documentation (README, docstrings, API spec) as needed.
5. Submit a Pull Request with a clear description of the change.

Please adhere to the **PEP 8** style guide and run `ruff`/`black` before committing.

---

## License

This project is licensed under the **Apache License 2.0**. See the `LICENSE` file for full details.