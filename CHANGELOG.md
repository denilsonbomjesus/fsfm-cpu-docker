# Histórico de Modificações (Changelog) - FSFM CPU-Only Docker

Este documento detalha as principais modificações aplicadas ao projeto FSFM original para criar esta versão adaptada, focada em rodar em ambiente **CPU-only via Docker**.

O objetivo principal das alterações foi remover a dependência de GPUs NVIDIA, simplificar o setup e facilitar a experimentação em hardware de propósito geral.

---

## Modificações por Componente

### 1. Ambiente e Dependências (`Dockerfile`)

- **Imagem Base Leve:** A imagem Docker foi baseada em `python:3.9-slim` para manter o tamanho final reduzido.
- **Instalação de PyTorch para CPU:** A dependência do PyTorch foi explicitamente alterada para a versão CPU-only (`torch --index-url https://download.pytorch.org/whl/cpu`). Isso evita o download de bibliotecas CUDA, economizando gigabytes de espaço.
- **Dependências do Sistema:** Foram adicionadas as dependências de sistema (`build-essential`, `cmake`, `libopenblas-dev`, etc.) necessárias para a compilação de bibliotecas como `dlib` e `opencv`.
- **Variáveis de Ambiente:** A variável `CUDA_VISIBLE_DEVICES` foi definida como `""` para garantir que o PyTorch não tente utilizar uma GPU, mesmo que disponível no host.

### 2. Scripts de Automação (`.sh`)

- **Criação de Scripts de Execução:** Foram criados os scripts `run_pretrain.sh`, `run_finetune.sh`, `run_finetune_diff.sh`, e `run_finetune_fas.sh`.
    - **Função:** Automatizar a execução das diferentes etapas do projeto (pré-treinamento e os três cenários de fine-tuning).
    - **Benefício:** Simplificam a linha de comando, evitando a necessidade de digitar múltiplos argumentos para os scripts Python.
- **Seleção Dinâmica de Datasets:** Os scripts foram refatorados para aceitar um `DATASET_ID` como argumento, permitindo ao usuário alternar facilmente entre diferentes conjuntos de dados (ex: `set_1_lfw`, `set_2_celeba`).
- **Caminhos Absolutos e Relativos:** Foram feitos ajustes nos caminhos dos datasets e diretórios de saída para garantir que funcionassem corretamente dentro do ambiente Docker, independentemente do diretório de onde o script é chamado.

### 3. Código-Fonte Python (`.py`)

O foco principal foi remover ou contornar qualquer código que dependesse diretamente da arquitetura CUDA.

- **`main_pretrain.py` e `engine_pretrain.py`:**
    - **Desativação do Modo Distribuído:** O código de inicialização de treinamento distribuído (`misc.init_distributed_mode`) foi desativado para permitir a execução em um único processo na CPU.
    - **Remoção de `torch.cuda.synchronize()`:** Chamadas a esta função, que são específicas para sincronização de operações em GPU, foram comentadas ou removidas para evitar erros em ambiente CPU.
    - **Acesso Direto ao Modelo:** Referências a `model.module`, um padrão comum ao usar `DistributedDataParallel`, foram alteradas para `model`, pois o modelo não é mais encapsulado para paralelismo de GPU.

- **`face_parse.py` (Pré-processamento):**
    - **Correção de Importação:** A importação da biblioteca `facer` foi ajustada.
    - **Robustez a Falhas de Parsing:** O script foi modificado para ser mais robusto. Agora, quando a biblioteca `facer` falha ao processar uma imagem, um arquivo de mapa de parsing "dummy" (um array numpy preenchido com `-1`) é criado. Isso garante que o `FaceParsingDataset` não encontre um `FileNotFoundError` durante o pré-treinamento, permitindo que o processo continue mesmo que algumas imagens não possam ser processadas.

- **`download_dataset.py` e Utilitários de Dataset:**
    - **Script de Download Genérico:** Foi criado o `download_dataset.py` para baixar e extrair datasets de URLs, visando facilitar a adição de novos conjuntos de dados.
    - **Instalação de `gdown`:** Adicionada a necessidade de instalar o pacote `gdown`, pois `torchvision` o utiliza para baixar datasets do Google Drive (como o CelebA).
    - **Estrutura de Diretórios:** O código foi adaptado para a nova estrutura de datasets (`datasets/pretrain_datasets/SET_ID/` e `datasets/finetune/TASK/SET_ID/`), garantindo que os scripts encontrem os dados nos locais corretos.

### 4. Documentação

- **`README.md` Simplificado:** O `README.md` principal foi completamente reescrito para servir como um guia de início rápido, focado em levar um novo usuário do "git clone" à execução dos treinamentos da forma mais direta possível.
- **`CONFIG_PROJECT.md`:** O `README.md` original, mais técnico e detalhado, foi renomeado para `CONFIG_PROJECT.md` para servir como documentação de referência para configurações avançadas.

---

Em resumo, o projeto foi sistematicamente modificado para ser autocontido, robusto e fácil de usar em um ambiente Docker CPU-only, democratizando o acesso à experimentação com o modelo FSFM.
