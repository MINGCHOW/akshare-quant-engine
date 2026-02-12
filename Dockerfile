FROM python:3.9-slim

WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY api/ api/

# Expose port
EXPOSE 8080

# Start command
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8080"]
