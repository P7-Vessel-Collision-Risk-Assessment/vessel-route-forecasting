# Use the official TensorFlow image with GPU support
FROM tensorflow/tensorflow:latest-gpu

# Set the working directory
WORKDIR /app

# Install additional dependencies if needed
RUN apt-get update && apt-get install -y \
    build-essential \
    libssl-dev \
    libffi-dev \
    wget \
    git \
    && apt-get clean


COPY requirements.txt /app/requirements.txt

# Install specific Python packages
RUN pip install --upgrade pip
RUN pip install --no-cache-dir --ignore-installed -r requirements.txt


# Copy the local files to the container
COPY . /app

EXPOSE 8080

# Command to run your app (e.g., a Python script or Jupyter Notebook)
CMD ["python", "src/app.py", "--port", "80"]
