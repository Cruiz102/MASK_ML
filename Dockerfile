FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && \
    apt-get install -y git libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender1 libfontconfig1 libice6 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN apt-get update && \
    apt-get install -y build-essential python3-dev


RUN git clone --recurse-submodules https://github.com/Cruiz102/MASK_ML.git
# Set the working directory to the project folder
WORKDIR /MASK_ML
# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install -e .
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
RUN pip install ultralytics
RUN pip install -e /MASK_ML/third_party/Cutie
RUN python third_party/Cutie/cutie/utils/download_models.py

# Copy and set up the entrypoint script
COPY docker_entrypoint.sh /docker_entrypoint.sh
RUN chmod +x /docker_entrypoint.sh

ENTRYPOINT ["/docker_entrypoint.sh"]

