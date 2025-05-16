# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container at /app
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
# Also, download the 'punkt' tokenizer for NLTK, which is commonly used.
RUN pip install --no-cache-dir -r requirements.txt && \
    python -m nltk.downloader punkt

# Copy the rest of the web_app (current directory '.') into the container at /app
COPY . .

# Make port 5000 available to the world outside this container
EXPOSE 5000

# Environment variables for Gunicorn and application
# PYTHONUNBUFFERED ensures that Python output (like print statements) is sent
# straight to terminal without being first buffered, which is good for logging.
ENV PYTHONUNBUFFERED=1
ENV GUNICORN_CMD_ARGS="--workers=2 --threads=4 --worker-class=gthread"

# Define environment variables
# FLASK_APP is used by 'flask run' command.
# FLASK_RUN_HOST allows the app to be accessible externally.
ENV FLASK_APP=run.py
ENV FLASK_RUN_HOST=0.0.0.0
ENV FLASK_RUN_PORT=5000
ENV FLASK_DEBUG=False

# Run run.py when the container launches
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "run:application"]