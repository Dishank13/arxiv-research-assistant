FROM python:3.11-slim

WORKDIR /app

# Install curl for health check in start.sh
RUN apt-get update && apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir pydantic-settings slowapi

COPY . .

RUN chmod +x start.sh

# HF Spaces expects port 7860
EXPOSE 7860

CMD ["bash", "start.sh"]
