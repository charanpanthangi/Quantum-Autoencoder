# Use a slim Python base image to keep the container lightweight.
FROM python:3.11-slim

# Set a working directory inside the container.
WORKDIR /app

# Copy dependency list first to leverage Docker layer caching.
COPY requirements.txt ./

# Install Python dependencies without caching to reduce final size.
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project files into the image.
COPY . .

# Default command runs the CLI demo. Users can override this in docker run.
CMD ["python", "app/main.py"]
