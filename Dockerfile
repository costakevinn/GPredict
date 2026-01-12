# Dockerfile for GPredict

# 1. Use official lightweight Python image
FROM python:3.11-slim

# 2. Set working directory
WORKDIR /app

# 3. Copy project files
COPY . /app

# 4. Install dependencies
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# 5. Set environment variable to prevent Python from buffering stdout
ENV PYTHONUNBUFFERED=1

# 6. Default command to run examples
CMD ["python", "main.py"]
