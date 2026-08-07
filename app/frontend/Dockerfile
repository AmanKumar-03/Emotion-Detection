# Emotion Detection - Streamlit Frontend Dockerfile

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
RUN pip install --no-cache-dir -r requirements.txt

# Copy all frontend source files
COPY . .

# Expose Streamlit port
EXPOSE 8501

# Start Streamlit application
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]