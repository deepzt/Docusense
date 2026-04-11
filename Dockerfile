# Base image (example)
FROM python:3.11-slim

# Install system deps for OCR + PDFs + healthcheck
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    poppler-utils \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps
COPY requirements.txt .
RUN echo "==== requirements in image ====" && cat requirements.txt && \
    pip install --no-cache-dir --no-deps -r requirements.txt

# Copy app code
COPY . .

# Run as non-root to limit blast radius if FAISS pickle deserialization is exploited
RUN useradd -m appuser && chown -R appuser /app
USER appuser

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:7860/ || exit 1

# Run only the UI
CMD ["python", "ui_app.py"]