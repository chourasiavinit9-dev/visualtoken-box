FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Download SmolLM2-135M model weights into container
RUN python download_model.py

# Hugging Face Spaces uses port 7860
EXPOSE 7860

# Launch GlassBox server
CMD ["python", "-m", "uvicorn", "glassbox.server.app:app", "--host", "0.0.0.0", "--port", "7860"]