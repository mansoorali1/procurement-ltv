FROM python:3.11-slim

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app/ ./app/
COPY models/ ./models/
COPY data/processed/featured/ ./data/processed/featured/
COPY data/processed/cleaned/ ./data/processed/cleaned/
COPY mlruns/ ./mlruns/
COPY src/ ./src/

# Expose Streamlit port
EXPOSE 7860

# Hugging Face Spaces uses port 7860
ENV STREAMLIT_SERVER_PORT=7860
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Health check
HEALTHCHECK CMD curl --fail http://localhost:7860/_stcore/health || exit 1

# Run
ENTRYPOINT ["streamlit", "run", "app/main.py", \
            "--server.port=7860", \
            "--server.address=0.0.0.0", \
            "--server.headless=true"]