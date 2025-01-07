FROM python:3.10-slim

# Set the working directory
WORKDIR /app

# Copy the requirements file
COPY . /app

# Install dependencies including optional ones
RUN pip install --upgrade pip && \
    pip install -e .[app]w

# Run the tests
CMD ["pytest", "-v"]
