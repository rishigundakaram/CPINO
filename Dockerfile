FROM python:3.10

WORKDIR /usr/src/app

COPY requirements.txt .

RUN pip install -r requirements.txt

RUN pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116
