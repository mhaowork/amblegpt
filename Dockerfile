# Use an official Python runtime as a parent image
FROM python:3.9

# Set the working directory in the container to /app
WORKDIR /app

# Copy the necessary contents into the container at /app
COPY mqtt_client.py /app
COPY requirements.txt /app
COPY .env /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Run mqtt_client.py when the container launches
CMD ["python", "./mqtt_client.py"]