Abaixo está o **Plano de Implementação Remodelado para WSL 2 com Docker (CPU-Only)**.

-----

### 1\. Estratégia Geral: "Mini-FSFM Containerizado"

Como seu hardware é limitado (i5 sem GPU dedicada), nós vamos criar um container Docker "leve" que roda tudo em CPU.

  * **Ambiente:** Ubuntu (via Docker) rodando dentro do WSL 2.
  * **Dados:** Usaremos o dataset **LFW (Labeled Faces in the Wild)** reduzido (apenas \~50 imagens) em vez dos datasets gigantes do artigo.
  * **Pipeline:**
    1.  Baixar Dataset LFW (direto pelo terminal Linux).
    2.  Pré-processar (Extração e Parsing) dentro do container.
    3.  Pré-treinar (FSFM) por poucas épocas (CPU).
    4.  Fine-tuning (Detecção) com dados sintéticos simples.

-----

### 2\. Preparação dos Arquivos (No Windows/WSL)

Abra seu terminal WSL (Ubuntu) e crie uma pasta para o projeto.

```bash
mkdir fsfm_project
cd fsfm_project
# Clone o repositório oficial
git clone https://github.com/wolo-wolo/FSFM-CVPR25.git
mv FSFM-CVPR25 src
```

Agora, crie um arquivo chamado `Dockerfile` na raiz da pasta `fsfm_project`. Este arquivo vai configurar todo o ambiente automaticamente.

**Conteúdo do `Dockerfile` (Copie e cole):**

```dockerfile
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
RUN pip install facer timm scipy pandas scikit-learn tensorboard

# 5. Copiar o código fonte para dentro do container
COPY ./src /app/src

# 6. Definir variável de ambiente para usar CPU
ENV CUDA_VISIBLE_DEVICES=""
ENV OMP_NUM_THREADS=1
```

-----

### 3\. Modificações no Código (Cirurgia para CPU)

Você precisará editar alguns arquivos Python dentro da pasta `src` que você clonou. Você pode usar o VS Code (conectado ao WSL) para isso.

#### A. Desativar Treinamento Distribuído (`src/fsfm-3c/pretrain/main_pretrain.py`)

O código original tenta iniciar conexões complexas entre GPUs. Precisamos forçar o modo "standalone".

1.  Abra `src/fsfm-3c/pretrain/main_pretrain.py`.
2.  Localize a função `main(args)`.
3.  Logo no início da função `main`, **adicione** estas linhas para forçar o desligamento do modo distribuído:

<!-- end list -->

```python
def main(args):
    # --- MODIFICAÇÃO PARA CPU START ---
    args.distributed = False
    import torch
    device = torch.device('cpu')
    # --- MODIFICAÇÃO PARA CPU END ---
    
    # Comente ou apague a linha original que dizia: 
    # misc.init_distributed_mode(args)
    
    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    # ... resto do código
```

4.  Busque onde o código define `device = torch.device(args.device)` e garanta que ele está usando a variável `device` que definimos acima (CPU).
5.  Busque por `model = torch.nn.parallel.DistributedDataParallel(model, ...)` e **comente** ou envolva em um `if args.distributed:`. Como setamos para `False`, ele usará o modelo puro.

#### B. Ajuste no Carregamento de Dados (`src/datasets/pretrain/dataset_preprocess.py`)

Simplifique este script. Em vez de usar o script complexo deles, vamos fazer algo manual na etapa de execução, pois o dataset será pequeno.

-----

### 4\. Construindo e Rodando o Container

No terminal do WSL, dentro da pasta `fsfm_project`:

1.  **Construir a Imagem:**

    ```bash
    docker build -t fsfm-cpu .
    ```

    *(Isso vai demorar uns minutos instalando o dlib e torch, mas só precisa ser feito uma vez).*

2.  **Rodar o Container:**
    Vamos montar o volume para que os dados persistam e você possa ver os logs.

    ```bash
    docker run -it --rm -v $(pwd)/src:/app/src -v $(pwd)/datasets:/app/datasets fsfm-cpu /bin/bash
    ```

    *Agora você está dentro do terminal do container Linux, pronto para rodar os scripts.*

-----

### 5\. O Pipeline de Execução (Dentro do Container)

Agora, dentro do container, execute passo a passo:

#### Passo 1: Baixar Mini-Dataset (LFW)

Vamos baixar o LFW e preparar uma estrutura "fake" que o código aceite.

```bash
# Crie a estrutura de pastas
mkdir -p /app/datasets/pretrain_datasets/mini_real/images
cd /app/datasets

# Baixar LFW (Labelled Faces in the Wild) - versão pequena
wget http://vis-www.cs.umass.edu/lfw/lfw-funneled.tgz
tar -xzf lfw-funneled.tgz

# Copiar apenas 50 imagens para nossa pasta de treino
# (Isso pega as primeiras 50 imagens encontradas nas subpastas)
find lfw_funneled -name "*.jpg" | head -n 50 | xargs -I {} cp {} /app/datasets/pretrain_datasets/mini_real/images/

# Criar uma pasta de validação vazia ou com poucas imagens para não quebrar o script
mkdir -p /app/datasets/pretrain_datasets/mini_real/val
cp /app/datasets/pretrain_datasets/mini_real/images/*.jpg /app/datasets/pretrain_datasets/mini_real/val/
```

#### Passo 2: Face Parsing (Crítico para o FSFM)

O FSFM precisa dos mapas de segmentação (`.npy`).

```bash
cd /app/src/datasets/pretrain
# Precisamos ajustar o script face_parse.py para apontar para nossa pasta
# Execute o script (ele vai baixar o modelo do facer automaticamente)
python face_parse.py --dataset_path /app/datasets/pretrain_datasets/mini_real/images
```

*Nota:* Se o `face_parse.py` pedir argumentos específicos do argparse original (como `dataset_name`), você pode precisar editar o final do arquivo `face_parse.py` para aceitar um caminho direto ou simplesmente hardcodar o caminho da pasta `mini_real`.

#### Passo 3: Pré-Treinamento (Execução Principal)

Aqui rodamos o modelo. Devido à sua RAM (16GB compartilhada), seremos conservadores.

```bash
cd /app/src/fsfm-3c/pretrain

# Comando adaptado para CPU Single-Core
python main_pretrain.py \
  --batch_size 4 \
  --accum_iter 1 \
  --epochs 5 \
  --model fsfm_vit_base_patch16 \
  --mask_ratio 0.75 \
  --pretrain_data_path /app/datasets/pretrain_datasets/mini_real \
  --output_dir ./output_cpu_test \
  --num_workers 0
```

  * **`--batch_size 4`**: Essencial para não estourar sua RAM.
  * **`--num_workers 0`**: Essencial para evitar erros de multiprocessamento em emulação.
  * **`--epochs 5`**: Suficiente para mostrar que "rodou" (o loss vai aparecer no terminal).

#### Passo 4: Simulação de Fine-Tuning (Prova de Conceito)

Para provar que o modelo treinado funciona, rodamos o script de detecção.

1.  Crie uma pasta `datasets/finetune` com duas subpastas: `real` e `fake`.
2.  Copie algumas imagens do LFW para `real`.
3.  Copie as mesmas imagens para `fake` e rabisque algo nelas (ou inverta cores) usando algum script Python rápido, só para serem diferentes matematicamente.
4.  Rode o `main_finetune.py` apontando para o checkpoint gerado no Passo 3 (`output_cpu_test/checkpoint-5.pth`).
