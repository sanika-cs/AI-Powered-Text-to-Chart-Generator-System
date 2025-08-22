# Use official Python 3.12 slim image
FROM python:3.12-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set work directory
WORKDIR /app

# Copy requirements.txt first for caching
COPY requirements.txt .

# Install system dependencies for audio, plotting, and build tools
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        ffmpeg \
        build-essential \
        libsndfile1 \
        git \
        pkg-config \
        python3-dev \
        && rm -rf /var/lib/apt/lists/*

# Upgrade pip, setuptools, wheel before installing dependencies
RUN pip install --upgrade pip setuptools wheel

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the app
COPY . .

# Expose the port Streamlit runs on
EXPOSE 8501

# Command to run the app
CMD ["streamlit", "run", "home.py", "--server.port=8501", "--server.address=0.0.0.0"]

