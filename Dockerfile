# Usar uma imagem base leve do Python 3.9 
FROM python:3.9-slim

# 1. Instalar dependências do sistema (necessárias para dlib e opencv)
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    libopenblas-dev \
    liblapack-dev \
    libx11-dev \
    libgtk-3-dev \
    git \
    wget \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# 2. Configurar diretório de trabalho
WORKDIR /app

# 3. Instalar PyTorch versão CPU (Muito importante para não baixar GBs de drivers NVIDIA inúteis)
# Isso economiza espaço e RAM no seu notebook
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# 4. Instalar outras dependências pesadas
RUN pip install dlib
RUN pip install opencv-python-headless
RUN pip install git+https://github.com/FacePerceiver/facer.git@main
RUN pip install timm==0.4.5 scipy pandas scikit-learn tensorboard submitit

# 5. Copiar o código fonte para dentro do container
COPY ./src /app/src

# 6. Definir variável de ambiente para usar CPU
ENV CUDA_VISIBLE_DEVICES=""
ENV OMP_NUM_THREADS=1
