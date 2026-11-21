
# syntax=docker/dockerfile:1

# 1. Use a Python base image
FROM python:3.13-slim

# 2. Set working directory
WORKDIR /app

# 3. Configure Python
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# 4. Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy application files
COPY main.py .
COPY download_model.py .

# 6. Download the latest model from DagsHub
# The token is provided securely during the build
RUN --mount=type=secret,unset DAGSHUB_TOKEN=dagshub_token,required=true \
    MLFLOW_TRACKING_USERNAME=Abdel380 \
    MLFLOW_TRACKING_PASSWORD="$(cat /run/secrets/dagshub_token)" \
    python download_model.py

# 7. Expose FastAPI port
EXPOSE 8000

# 8. Start FastAPI
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]