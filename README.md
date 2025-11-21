# FSFM-CVPR25 CPU-Only Setup Guide

This guide outlines the steps to set up and run the FSFM-CVPR25 project in a CPU-only environment using Docker and WSL 2, following the "Plano de Implementação Remodelado para WSL 2 com Docker (CPU-Only)".

---

## 1. Preparação dos Arquivos (No Windows/WSL) 

1.  **Clone o repositório oficial e configure a pasta do projeto:**
    ```bash
    git clone https://github.com/wolo-wolo/FSFM-CVPR25.git
    mv FSFM-CVPR25 src
    ```

2.  **Crie o `Dockerfile` na raiz do projeto:**
    Crie um arquivo chamado `Dockerfile` com o seguinte conteúdo:

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
    RUN pip install git+https://github.com/FacePerceiver/facer.git@main
    RUN pip install timm==0.4.5 scipy pandas scikit-learn tensorboard submitit torchsummary # Corrigido: Versão timm, submitit e torchsummary
    
    # 5. Copiar o código fonte para dentro do container
    COPY ./src /app/src

    # 6. Definir variável de ambiente para usar CPU
    ENV CUDA_VISIBLE_DEVICES=""
    ENV OMP_NUM_THREADS=1
    ```

---

## 2. Modificações no Código (Cirurgia para CPU)

As seguintes modificações foram aplicadas ao arquivo `src/fsfm-3c/pretrain/main_pretrain.py`:

1.  **Desativar Treinamento Distribuído e Forçar CPU:**
    *   No início da função `main(args)`, o código foi modificado para forçar o uso da CPU e desabilitar o modo distribuído. A linha `misc.init_distributed_mode(args)` e a definição `device = torch.device(args.device)` foram comentadas e substituídas pela definição `device = torch.device('cpu')`.

2.  **Desabilitar `DistributedDataParallel` para Múltiplas GPUs:**
    *   O bloco `if torch.cuda.device_count() > 1:` foi modificado para incluir `and args.distributed`, garantindo que a lógica de paralelização para múltiplas GPUs não seja ativada em ambiente CPU.

3.  **Execução Local do Script Principal:**
    *   O bloco `if __name__ == '__main__':` foi ajustado para remover a dependência de `submitit` e chamar `main(args)` diretamente, adaptando o script para execução local em ambiente CPU.

Adicionalmente, os seguintes scripts foram modificados para corrigir erros e garantir a compatibilidade com o ambiente:

*   **`src/datasets/pretrain/preprocess/face_parse.py`:**
    1.  **Correção da Importação do `facer`:** A linha `from tools.facer import facer` foi alterada para `import facer`, pois a biblioteca `facer` é instalada como um pacote Python padrão.
    2.  **Ajuste do Argument Parser e Execução Principal:** As funções `get_args_parser` e o bloco `if __name__ == '__main__':` foram redefinidos para aceitar o argumento `--dataset_path` e direcionar a saída para o diretório pai do caminho das imagens.
    3.  **Compatibilidade com Python 3.9 (`facer`):** A biblioteca `facer`, instalada via `git+https`, utiliza a sintaxe de `Union` (operador `|`) para type hints. Como o container utiliza Python 3.9, foram aplicadas correções via `sed` em `/usr/local/lib/python3.9/site-packages/facer/face_parsing/farl.py`.

*   **`src/fsfm-3c/models_fsfm.py`:**
    1.  **Remoção de `qk_scale`:** O argumento `qk_scale=None` foi removido das chamadas do construtor `Block`, pois a versão `timm==0.4.5` (original do projeto) não o aceita.

*   **`src/fsfm-3c/util/pos_embed.py`:**
    1.  **Correção de `np.float`:** A expressão `np.float` foi substituída por `float` para compatibilidade com versões mais recentes do NumPy.

*   **`src/fsfm-3c/pretrain/main_pretrain.py`:**
    1.  **Comentar `torchsummary`:** A chamada para `summary.summary` foi comentada, pois causava um `ValueError` com o formato de entrada.
    2.  **Correção da passagem do modelo:** A função `train_one_epoch` agora recebe `model_without_ddp` em vez de `model`, garantindo que o modelo não encapsulado por `DistributedDataParallel` seja usado.

*   **`src/fsfm-3c/pretrain/engine_pretrain.py`:**
    1.  **Correção de `model.module`:** As referências a `model.module` foram alteradas para `model` diretamente ao iterar sobre os parâmetros, pois o modelo não está em modo distribuído.
    2.  **Comentar `torch.cuda.synchronize()`:** A chamada `torch.cuda.synchronize()` foi comentada, pois causa um erro em ambientes CPU-only.

---

## 3. Construindo e Rodando o Container

1.  **Construa a Imagem Docker:**
    ```bash
    docker build -t fsfm-cpu .
    ```

2.  **Inicie o Container Docker (modo detached):**
    Este comando inicia um container em segundo plano, nomeado `fsfm_container`, que permanecerá ativo para que os comandos seguintes possam ser executados.
    ```bash
    docker run -d --name fsfm_container -v $(pwd):/app fsfm-cpu /bin/bash -c "sleep infinity"
    ```
    *Obs: O diretório raiz do projeto local (`$(pwd)`) é montado como `/app` dentro do container, garantindo que o `config_pretraining.sh` e outros arquivos do projeto estejam acessíveis.*

---

## 4. O Pipeline de Execução (Dentro do Container)

Todos os comandos abaixo devem ser executados **dentro do container** (usando `docker exec fsfm_container ...`).

### **Passo 1: Baixar Mini-Dataset (LFW)**

O download direto do dataset LFW do `vis-www.cs.umass.edu` e `www.cs.cmu.edu` falhou devido a problemas de rede/DNS ou erro 404. O dataset foi baixado e preparado usando a biblioteca `scikit-learn` dentro do container.

1.  **Crie a estrutura de diretórios necessária:**
    ```bash
    docker exec fsfm_container mkdir -p /app/datasets/pretrain_datasets/mini_real/images
    ```

2.  **Crie um script Python para baixar o dataset LFW (usando `scikit-learn`):**
    Crie o arquivo `src/download_lfw.py` com o seguinte conteúdo:
    ```python
    from sklearn.datasets import fetch_lfw_people
    import os

    print("Downloading LFW dataset...")
    lfw_people = fetch_lfw_people(min_faces_per_person=1, resize=0.4)
    print("Download complete.")

    from sklearn.datasets import get_data_home
    data_home = get_data_home()
    print(f"Scikit-learn data home: {data_home}")

    lfw_dir = os.path.join(data_home, 'lfw_home')
    print(f"LFW data should be in: {lfw_dir}")

    if os.path.exists(lfw_dir):
        print(f"Found LFW home directory at: {lfw_dir}")
    else:
        print("LFW home directory not found where expected.")
    ```

3.  **Execute o script para baixar o dataset:**
    ```bash
    docker exec fsfm_container python /app/src/download_lfw.py
    ```
    *(Este comando fará o download do dataset para `/root/scikit_learn_data/lfw_home` dentro do container.)*

4.  **Copie 50 imagens do dataset LFW para a pasta de treinamento:**
    ```bash
    docker exec fsfm_container bash -c "find /root/scikit_learn_data/lfw_home/lfw_funneled -name '*.jpg' | head -n 50 | xargs -I {} cp --parents {} /app/datasets/pretrain_datasets/mini_real/images/"
    ```

5.  **Crie a pasta de validação e copie as imagens para ela:**
    ```bash
    docker exec fsfm_container bash -c "mkdir -p /app/datasets/pretrain_datasets/mini_real/val && find /app/datasets/pretrain_datasets/mini_real/images -name '*.jpg' -print0 | xargs -0 -I {} cp {} /app/datasets/pretrain_datasets/mini_real/val/"
    ```

### **Passo 2: Face Parsing (Crítico para o FSFM)**

1.  **Instale a dependência `yacs`:**
    ```bash
    docker exec fsfm_container pip install yacs
    ```

2.  **Aplique as correções de sintaxe no `farl.py` (necessário para Python 3.9):**
    A biblioteca `facer`, instalada via `git+https`, utiliza a sintaxe de `Union` (operador `|`) para type hints, que é compatível apenas com Python 3.10 ou superior. Como o container utiliza Python 3.9, é necessário corrigir o arquivo `farl.py` diretamente.
    ```bash
    docker exec fsfm_container sed -i "s/from typing import Optional, Dict, Any/from typing import Optional, Dict, Any, Union/" /usr/local/lib/python3.9/site-packages/facer/face_parsing/farl.py
    docker exec fsfm_container sed -i "s/images: torch.Tensor|np.ndarray|list/images: Union[torch.Tensor, np.ndarray, list]/" /usr/local/lib/python3.9/site-packages/facer/face_parsing/farl.py
    ```

3.  **Execute o script `face_parse.py` (com o caminho corrigido):**
    ```bash
    docker exec fsfm_container bash -c "cd /app/src/datasets/pretrain/preprocess && python face_parse.py --dataset_path /app/datasets/pretrain_datasets/mini_real/images"
    ```
    *(Este comando fará o processamento das imagens e salvará os mapas de segmentação facial.)*



### **Passo 3: Pré-Treinamento (Execução Principal)**

O pré-treinamento do modelo é executado com o script `main_pretrain.py`. Os parâmetros de execução foram centralizados em um arquivo de configuração para facilitar a modificação.

1.  **Crie o arquivo de configuração `config_pretraining.sh` na raiz do seu projeto:**
    ```bash
    #!/bin/bash

    # Configuration for pre-training the FSFM model on CPU

    # Batch size: Essential to avoid running out of RAM
    export BATCH_SIZE=4

    # Accumulate gradient iterations: For increasing effective batch size under memory constraints
    export ACCUM_ITER=1

    # Number of epochs: Sufficient to show that it "ran"
    export EPOCHS=5

    # Model name
    export MODEL_NAME="fsfm_vit_base_patch16"

    # Mask ratio
    export MASK_RATIO=0.75

    # Path to the pre-training dataset
    export PRETRAIN_DATA_PATH="/app/datasets/pretrain_datasets/mini_real"

    # Output directory for checkpoints and logs
    export OUTPUT_DIR="./output_cpu_test"

    # Number of data loading workers: Essential for avoiding multiprocessing errors in emulation
    export NUM_WORKERS=0
    ```
    *Obs: Certifique-se de que o arquivo `config_pretraining.sh` esteja presente na raiz do seu projeto local. Ele será montado automaticamente dentro do container no diretório `/app`.*

2.  **Execute o script de pré-treinamento dentro do container, utilizando o script `run_pretrain.sh`:**
    Para garantir que os parâmetros do `config_pretraining.sh` sejam utilizados corretamente, criamos um script wrapper `run_pretrain.sh` que faz a leitura do arquivo de configuração e executa o comando `python main_pretrain.py`.
    ```bash
    docker exec fsfm_container /app/run_pretrain.sh
    ```
    *   **Parâmetros de Configuração**: Você pode ajustar os valores de `BATCH_SIZE`, `EPOCHS`, etc., editando o arquivo `config_pretraining.sh` diretamente. As mudanças serão refletidas na próxima execução do `run_pretrain.sh`.
    *   **Saída**: O script irá gerar logs e checkpoints no diretório especificado por `OUTPUT_DIR` (padrão: `./output_cpu_test`).
