# Use a base image with Python
FROM python:3.9-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the project files into the container
COPY . .

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip setuptools wheel
RUN pip install --no-cache-dir numpy==1.21.6 pandas==1.3.5 scikit-learn==1.0.2
RUN pip install --no-cache-dir torch==2.5.1 torchaudio==2.5.1 torchvision==0.20.1
RUN pip install --no-cache-dir -r requirements.txt

# Make the run.sh script executable
RUN chmod +x run.sh

# Expose the port Streamlit will use
EXPOSE 8501

# Use /bin/bash to avoid compatibility issues with the shebang
CMD ["/bin/bash", "./run.sh"]
