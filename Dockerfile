FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir pydantic-settings slowapi

COPY . .

# HF Spaces expects port 7860
EXPOSE 7860

CMD ["streamlit", "run", "frontend/app.py", "--server.port=7860", "--server.address=0.0.0.0", "--server.headless=true"]
