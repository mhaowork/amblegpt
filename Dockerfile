# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container to /app
WORKDIR /app

# Copy the necessary contents into the container at /app
COPY mqtt_client.py /app
COPY requirements.txt /app

# Install any needed packages specified in requirements.txt
RUN pip install --upgrade pip

RUN apt-get update \
    && apt-get install -y --no-install-recommends gcc python3-dev \
    && rm -rf /var/lib/apt/lists/* \
    && pip install psutil \
    && apt-get purge -y --auto-remove gcc python3-dev

RUN pip install --no-cache-dir -r requirements.txt

# Run mqtt_client.py when the container launches
CMD ["python", "./mqtt_client.py"]
