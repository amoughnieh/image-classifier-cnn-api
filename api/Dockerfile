FROM python:3.11-slim

# Set working directory
WORKDIR /code

# Copy requirements
COPY ./requirements.txt /code/requirements.txt

# Install dependencies and clean up in the same layer to reduce image size
RUN pip install --no-cache-dir -r /code/requirements.txt \
    && rm -rf /root/.cache/pip

# Copy the rest of the application
COPY ./app /code/app
COPY ./best_model/CNN_best_model-f1_0.7887.pth /code/best_model/

# Expose port from environment variable (for documentation)
EXPOSE ${PORT:-8080}

# Command to run the application
CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8080}