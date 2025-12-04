# FSFM-CVPR25 (CPU-Only Docker Version)

Esta é uma versão modificada do projeto FSFM, adaptada para rodar exclusivamente em ambiente **CPU via Docker**. O objetivo é facilitar a execução, teste e experimentação do modelo sem a necessidade de uma GPU.

## Pré-requisitos

- **Docker:** Certifique-se de que o Docker esteja instalado e em execução em seu sistema.
- **Git:** Necessário para clonar o repositório.
- **Conexão com a Internet:** Para clonar o repositório e construir a imagem Docker.

---

## Guia de Início Rápido

Siga estes passos para colocar o projeto em funcionamento.

### Passo 1: Clonar o Repositório

Primeiro, clone este repositório para a sua máquina local:

```bash
git clone https://github.com/denilsonbomjesus/fsfm-cpu-docker.git
cd fsfm-cpu-docker
```

### Passo 2: Baixar e Estruturar os Datasets

Este projeto utiliza um conjunto de datasets para pré-treinamento e fine-tuning. Para facilitar, todos os datasets necessários foram agrupados em um único arquivo.

1.  **Baixe o arquivo `datasets.zip`** do seguinte link:
    - [Link para Download dos Datasets (Google Drive)](https://drive.google.com/file/d/1YfP7uN_Lb7DMMNo-Xs-5-SoXe3kdGS3i/view?usp=sharing)

2.  **Descompacte e estruture os datasets:**
    Após o download, descompacte o arquivo `datasets.zip` na raiz do projeto. A estrutura de pastas final deve ser a seguinte:

    ```
    fsfm-cpu-docker/
    ├── datasets/
    │   ├── finetune/
    │   │   ├── DfD/
    │   │   │   └── set_1_lfw_mock/
    │   │   ├── DiFF/
    │   │   │   └── set_1_lfw_mock/
    │   │   └── FAS/
    │   │       └── set_1_lfw_mock/
    │   └── pretrain_datasets/
    │       ├── set_1_lfw/
    │       └── set_2_celeba/
    ├── src/
    ├── Dockerfile
    ├── README.md
    └── ... (outros arquivos do projeto)
    ```

### Passo 3: Construir o Ambiente Docker

Com os datasets no lugar, construa a imagem Docker. Este comando utiliza o `Dockerfile` do projeto para criar um ambiente com todas as dependências necessárias.

```bash
docker build -t fsfm-cpu .
```

### Passo 4: Pré-treinamento do Modelo

O pré-treinamento é o primeiro passo para treinar o modelo FSFM.

1.  **Inicie o contêiner Docker (se ainda não estiver rodando):**
    Este comando inicia o contêiner em segundo plano e monta o diretório do projeto, permitindo que você execute os scripts.

    ```bash
    docker run -d --name fsfm_container -v $(pwd):/app fsfm-cpu /bin/bash -c "sleep infinity"
    ```

2.  **Execute o script de pré-treinamento:**
    Utilizamos o script `run_pretrain.sh` para iniciar o processo. Por padrão, ele usará o dataset `set_1_lfw`.

    ```bash
    docker exec fsfm_container bash /app/run_pretrain.sh set_1_lfw
    ```

    - **Para usar outro dataset**, como o `set_2_celeba`, basta alterar o argumento:
      ```bash
      docker exec fsfm_container bash /app/run_pretrain.sh set_2_celeba
      ```
    - Os logs e checkpoints serão salvos em `src/fsfm-3c/pretrain/output_DATASET_ID/`.

### Passo 5: Fine-Tuning do Modelo

Após o pré-treinamento, você pode especializar o modelo para tarefas específicas usando os scripts de fine-tuning.

#### Cenário 1: Cross Dataset DfD (DeepFake Detection)

- **Comando:**
  ```bash
  docker exec fsfm_container bash /app/run_finetune.sh set_1_lfw_mock
  ```
- **O que faz:** Executa o fine-tuning para detecção de DeepFakes.
- **Saída:** Logs e checkpoints serão salvos em `src/fsfm-3c/finuetune/cross_dataset_DfD/output_finetune_set_1_lfw_mock/`.

#### Cenário 2: Cross Dataset Unseen DiFF (Unseen DeepFake Detection)

- **Comando:**
  ```bash
  docker exec fsfm_container bash /app/run_finetune_diff.sh set_1_lfw_mock
  ```
- **O que faz:** Executa o fine-tuning para detecção de DeepFakes de métodos "não vistos" durante o treinamento.
- **Saída:** Logs e checkpoints serão salvos em `src/fsfm-3c/finuetune/cross_dataset_unseen_DiFF/output_finetune_set_1_lfw_mock/`.

#### Cenário 3: Cross Domain FAS (Face Anti-Spoofing)

- **Comando:**
  ```bash
  docker exec fsfm_container bash /app/run_finetune_fas.sh set_1_lfw_mock
  ```
- **O que faz:** Executa o fine-tuning para detecção de ataques de "spoofing" facial (ex: foto de uma foto, vídeo de uma face).
- **Saída:** Logs e checkpoints serão salvos em `src/fsfm-3c/finuetune/cross_domain_FAS/output_finetune_set_1_lfw_mock/`.

---

Com isso, você pode executar tanto o pré-treinamento quanto os diferentes cenários de fine-tuning do modelo FSFM em seu próprio ambiente, de forma simplificada e sem a necessidade de hardware especializado.
