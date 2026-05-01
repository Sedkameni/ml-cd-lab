FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies required for PyTorch
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire webapp directory
COPY ./webapp /app

# Expose port 5000
EXPOSE 5000

# Run the Flask application
CMD ["python", "app.py"]