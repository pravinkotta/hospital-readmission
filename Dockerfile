# CPU-only service image: API + trained artefacts + cohort data.
#
#   docker build -t readmission-api .
#   docker run --rm -p 8000:8000 --env-file .env readmission-api
#
# torch is installed from the CPU wheel index first: the default Linux wheel
# pulls in CUDA libraries (~2.5 GB) and this service never touches a GPU.
FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN pip install --upgrade pip \
    && pip install torch --index-url https://download.pytorch.org/whl/cpu

# Serving-only dependency set: the full requirements.txt drags in xgboost (+
# a 340 MB CUDA wheel), MLflow, Evidently and the notebook stack that the API
# never imports — too much for an 8 GB micro's disk. See the file's header.
COPY requirements-service.txt .
RUN pip install -r requirements-service.txt

# artefacts (model, preprocessor, SHAP background, metrics) and the cleaned
# cohort the API serves are committed on purpose — the image needs no training.
# artifacts/chroma carries the guideline RAG index; guidelines/ is the source
# corpus it was built from (python -m src.rag --build re-creates it on-box).
COPY src/ ./src/
COPY artifacts/ ./artifacts/
COPY guidelines/ ./guidelines/
COPY data/diabetic_data.csv data/IDS_mapping.csv ./data/

EXPOSE 8000

# --env-file .env supplies OPENAI_API_KEY (for /alert) and ALERT_API_KEY
CMD ["python", "-m", "uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
