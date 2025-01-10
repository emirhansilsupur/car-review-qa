# Base image
FROM python:3.12-slim

# Set the working directory
WORKDIR /app

# Set PYTHONPATH to include /app/src
ENV PYTHONPATH="${PYTHONPATH}:/app/src"

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    chromium-driver \
    && rm -rf /var/lib/apt/lists/*

# install app dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy in the source code
COPY . .

# Create necessary directories
RUN mkdir -p articles/raw/expert_review articles/raw/long_term_reviews vector_db

# Run the app
EXPOSE 8501
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["streamlit", "run", "src/app.py", "--server.port=8501", "--server.address=0.0.0.0"] 