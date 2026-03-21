# Base image
FROM python:3.11-slim

# Working directory
WORKDIR /app

# Copy project files
COPY . .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose API port
EXPOSE 8000

# Run FastAPI service
CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]