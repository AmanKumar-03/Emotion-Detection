# Emotion Detection - FastAPI Backend Dockerfile

# Use official Python image
FROM python:3.11-slim

# Prevent Python from creating .pyc files
ENV PYTHONDONTWRITEBYTECODE=1

# Enable real-time logging
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Copy requirements file
COPY requirements.txt .

# Upgrade pip and install dependencies
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy all backend source files
COPY . .

# Expose FastAPI port
EXPOSE 8000

# Start FastAPI server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]