# Deployment Guide

Complete deployment guide for EXAI-ResumeIntel across local, containerised, and cloud environments.

**Author:** Mithin Sagar S ([@mithinsagar](https://github.com/mithinsagar))

## Table of Contents

- [Local Deployment](#local-deployment)
- [Docker](#docker)
- [Docker Compose](#docker-compose)
- [Kubernetes](#kubernetes)
- [Cloud Platforms](#cloud-platforms)
  - [Render](#render)
  - [Railway](#railway)
  - [Hugging Face Spaces](#hugging-face-spaces)
  - [AWS ECS](#aws-ecs)
  - [Google Cloud Run](#google-cloud-run)
- [GitHub Pages (Static UI only)](#github-pages)
- [Nginx Reverse Proxy](#nginx-reverse-proxy)
- [Environment Variables](#environment-variables)

## Local Deployment

The simplest deployment runs three services on separate ports:

```bash
# Terminal 1: FastAPI backend on port 8765
make serve

# Terminal 2: Static UI on port 8080
make serve-ui

# Terminal 3: Streamlit app on port 8501
make serve-app
```

Access points:
- API docs: http://localhost:8765/docs
- Landing page: http://localhost:8080/landing.html
- Analyzer: http://localhost:8080/index.html
- Streamlit: http://localhost:8501

## Docker

### Build the API image

```bash
docker build -t exai-resumeintel:latest -f deployment/Dockerfile .
```

### Run the API container

```bash
docker run -d \
  --name exai-api \
  -p 8765:8765 \
  -v $(pwd)/models:/app/models:ro \
  -v $(pwd)/data:/app/data:ro \
  exai-resumeintel:latest
```

### Streamlit image

```bash
docker build -t exai-streamlit:latest -f deployment/Dockerfile.streamlit .
docker run -d --name exai-app -p 8501:8501 exai-streamlit:latest
```

## Docker Compose

The full stack launches with:

```bash
docker-compose -f deployment/docker-compose.yml up -d
```

Services started:

| Service | Port | Description |
|:---|:---:|:---|
| `api` | 8765 | FastAPI backend |
| `ui` | 8080 | Static HTML dashboard |
| `app` | 8501 | Streamlit application |
| `nginx` | 80 | Reverse proxy |

Access everything through http://localhost after the stack starts.

Stop the stack:

```bash
docker-compose -f deployment/docker-compose.yml down
```

## Kubernetes

Apply the deployment manifest:

```bash
kubectl apply -f deployment/kubernetes/deployment.yaml
```

This creates:
- A `Deployment` with 2 replicas of the API
- A `Service` exposing port 8765 internally
- An `Ingress` for external HTTPS access

Check status:

```bash
kubectl get pods -l app=exai-resumeintel
kubectl logs -l app=exai-resumeintel --tail=100
```

## Cloud Platforms

### Render

The easiest managed deployment. Free tier available.

1. Push the repo to GitHub.
2. On [render.com](https://render.com), create a new Web Service and connect the repo.
3. Configure:
   - Environment: `Python 3.11`
   - Build Command: `pip install -r requirements.txt && bash scripts/download_data.sh`
   - Start Command: `uvicorn api.server:app --host 0.0.0.0 --port $PORT`
4. Add environment variables from `.env.example`.
5. Deploy.

The static UI can be deployed as a separate static site on Render's static hosting.

### Railway

Similar to Render but with different pricing.

1. Install Railway CLI: `npm install -g @railway/cli`
2. `railway login`
3. `railway init` inside the repo
4. `railway up`

Set `API_HOST=0.0.0.0` and `API_PORT=$PORT` in the Railway dashboard.

### Hugging Face Spaces

Best fit for the Streamlit application because Spaces natively supports Streamlit and can serve large models from Hugging Face Hub without egress limits.

1. Create a new Space at [huggingface.co/new-space](https://huggingface.co/new-space).
2. Select "Streamlit" as the SDK.
3. Push the repository to the Space's git remote.
4. Add `models/` and `data/` as symlinks to your dataset repos (or download in the Space entrypoint).
5. The Space auto-deploys on push.

**Cost:** Free CPU tier for public Spaces.

### AWS ECS

For production workloads with autoscaling.

1. Push the Docker image to ECR:
   ```bash
   aws ecr create-repository --repository-name exai-resumeintel
   docker tag exai-resumeintel:latest <account>.dkr.ecr.<region>.amazonaws.com/exai-resumeintel:latest
   docker push <account>.dkr.ecr.<region>.amazonaws.com/exai-resumeintel:latest
   ```
2. Create an ECS task definition using the pushed image.
3. Create an ECS service behind an Application Load Balancer.
4. Store datasets in S3, mount via EFS or download at task startup.

### Google Cloud Run

Serverless, pay-per-request. Best for low-traffic public deployments.

```bash
gcloud builds submit --tag gcr.io/PROJECT_ID/exai-resumeintel
gcloud run deploy exai-resumeintel \
  --image gcr.io/PROJECT_ID/exai-resumeintel \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 2Gi \
  --cpu 2
```

## GitHub Pages

The static landing page can be deployed to GitHub Pages for free.

1. In your repo settings, enable GitHub Pages, source: `main` branch, `/ui` folder.
2. The landing page will be available at:
   `https://mithinsagar.github.io/EXAI-ResumeIntel/landing.html`
3. Configure `EXAI_API_BASE` in the URL: `?api=https://your-api-domain.com`

## Nginx Reverse Proxy

The `deployment/nginx.conf` file provides a reverse proxy that routes:

- `/`                -> static UI (port 8080)
- `/api/`            -> FastAPI (port 8765)
- `/app/`            -> Streamlit (port 8501)
- `/docs`            -> Swagger UI
- `/redoc`           -> ReDoc

Use case: single-domain deployment where all services live behind one hostname with TLS termination.

## Environment Variables

All environment variables are documented in `.env.example`. Key ones for deployment:

| Variable | Default | Description |
|:---|:---|:---|
| `API_HOST` | 0.0.0.0 | Interface for FastAPI to bind |
| `API_PORT` | 8765 | FastAPI port |
| `API_LOG_LEVEL` | info | Uvicorn log level |
| `CORS_ORIGINS` | localhost:8080,8501 | Comma-separated allowed origins |
| `EMBEDDING_MODEL_PATH` | models/embedding_engine.pkl | Path to trained embedding engine |
| `HF_DATA_REPO` | mithinsagar/exai-resumeintel-data | HF dataset repo |
| `HF_MODEL_REPO` | mithinsagar/exai-resumeintel-models | HF model repo |
| `HF_TOKEN` | (empty) | Optional HF token for private repos |
| `SBERT_MODEL_NAME` | sentence-transformers/all-mpnet-base-v2 | SBERT model |

## Data and Model Handling

### Large file strategies

The datasets and precomputed models can be handled three ways:

1. **Hugging Face Hub** (recommended, default in this repo). Datasets and models are hosted on HF and downloaded by `scripts/download_data.sh`. Free, no bandwidth limits for public repos.
2. **Git LFS**. Add `.gitattributes` to track `*.csv`, `*.pkl`, `*.memmap` with LFS. Watch the free 1 GB storage / 1 GB monthly bandwidth cap.
3. **S3 / GCS / Azure Blob**. For enterprise deployments, replace the download script with a `boto3` / `google-cloud-storage` / `azure-storage-blob` call.

### First-time model generation

If you have raw CSVs but no trained model:

```bash
python -m training.train
```

Takes 2-5 minutes on CPU for the 2,484-resume corpus.

## Health Checks

For load balancers and Kubernetes liveness probes:

```
GET /health
```

Returns `{"status": "ok", "model_loaded": true|false, "version": "1.0.0"}` with HTTP 200 when healthy.

Recommended probe configuration:
- Initial delay: 15 seconds (allow embedding model to load)
- Period: 30 seconds
- Timeout: 5 seconds
- Failure threshold: 3

## Monitoring

The FastAPI server logs to stdout in the format:

```
2026-01-15 10:23:45 [INFO] exai.server: EXAI-ResumeIntel v1.0.0 ready
2026-01-15 10:24:12 [INFO] uvicorn.access: 127.0.0.1:52134 - "POST /analyze HTTP/1.1" 200
```

Ship logs to CloudWatch / Stackdriver / Loki via the standard container logging drivers.

For metrics, wrap FastAPI with `prometheus-fastapi-instrumentator`:

```bash
pip install prometheus-fastapi-instrumentator
```

Then expose `/metrics` in `api/server.py`.
