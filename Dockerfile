FROM nvcr.io/nvidia/pytorch:22.09-py3

WORKDIR /src/home/rishi/projects

COPY requirements.txt .

RUN pip install -r requirements.txt

# RUN pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116

