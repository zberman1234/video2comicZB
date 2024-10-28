# Use the official Python image as the base image
FROM python:3.11

# Install Redis server and other dependencies
RUN apt-get update && \
    apt-get install -y redis-server && \
    rm -rf /var/lib/apt/lists/*

# Set environment variables
ENV REDIS_URL=redis://localhost PYTHONUNBUFFERED=1

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Initialize Reflex (if needed)
RUN reflex init

# Build the frontend assets
RUN reflex export --frontend-only --no-zip

# Expose the port that the app runs on
EXPOSE 8000

# Set the command to run your app using the PORT environment variable provided by Render
CMD ["/bin/bash", "-c", "redis-server --daemonize yes && reflex run --env prod --port ${PORT:-8000}"]
