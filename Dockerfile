FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000

WORKDIR /app

# Install dependencies first for better layer caching.
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir .

EXPOSE 8000

# Honour the platform-provided $PORT (Container Apps sets it); default 8000.
CMD ["sh", "-c", "uvicorn budget_agent.service:app --host 0.0.0.0 --port ${PORT:-8000}"]
