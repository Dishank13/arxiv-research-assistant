#!/bin/bash
# Start both FastAPI backend and Streamlit frontend for HF Spaces

# Start FastAPI in the background
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 &

# Wait for API to be ready
echo "Waiting for API server to start..."
for i in $(seq 1 30); do
    if curl -s http://localhost:8000/health > /dev/null 2>&1; then
        echo "API server is ready!"
        break
    fi
    sleep 1
done

# Start Streamlit on port 7860 (HF Spaces default)
python -m streamlit run frontend/app.py \
    --server.port=7860 \
    --server.address=0.0.0.0 \
    --server.headless=true \
    --browser.gatherUsageStats=false
