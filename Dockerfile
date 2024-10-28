# Use the official Python image as the base image
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    curl redis-server libgl1-mesa-glx libglib2.0-0 util-linux && \
    rm -rf /var/lib/apt/lists/*

# Install Node.js
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - && \
    apt-get install -y nodejs

# Install Bun
RUN curl -fsSL https://bun.sh/install | bash && \
    mv /root/.bun /usr/local/bin/bun && \
    export BUN_INSTALL="/usr/local/bin/bun" && \
    export PATH="$BUN_INSTALL/bin:$PATH"

# Set environment variables
ENV REDIS_URL=redis://localhost \
    PYTHONUNBUFFERED=1 \
    PATH="/usr/local/bin/bun/bin:$PATH"

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install reflex-chakra

# Initialize Reflex
RUN reflex init

# Build the frontend assets without prerendering
RUN reflex export --frontend-only --no-zip --no-prerender

# Expose the port that the app runs on
EXPOSE 8000

# Command to start Redis and run the app
CMD ["bash", "-c", "redis-server --daemonize yes && reflex run --env prod"]
