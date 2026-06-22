# Image de l'agent/chat pour Cloud Run.
FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml ./
COPY src ./src
COPY data ./data

RUN pip install --no-cache-dir -e .

# Cloud Run fournit $PORT
ENV PORT=8080
EXPOSE 8080

# Chainlit écoute sur $PORT
CMD ["sh", "-c", "chainlit run src/agentic4api/chat/app.py --host 0.0.0.0 --port ${PORT}"]
