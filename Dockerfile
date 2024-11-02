# Use the official TensorFlow image with GPU support
FROM tensorflow/tensorflow:latest-gpu

# Set the working directory
WORKDIR /app

# Install dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libssl-dev \
    libffi-dev \
    wget \
    git \
    && apt-get clean

# Copy the requirements file
COPY requirements.txt /app/requirements.txt

# Install Python packages
RUN pip install --upgrade pip
RUN pip install --no-cache-dir --ignore-installed -r requirements.txt


# Copy the local files to the container
COPY . /app

EXPOSE 8080

CMD ["python", "src/app.py", "--port", "8080"]
