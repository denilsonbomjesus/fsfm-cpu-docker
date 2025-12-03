# Plano de Implementação e Evolução do Projeto FSFM

Este documento detalha o plano de ação para expandir e robustecer o projeto FSFM, com base na análise da implementação atual e nos objetivos de evolução. O plano está estruturado em fases sequenciais para garantir um desenvolvimento incremental, organizado e testável.

---

## Fase 1: Implementação do Finetuning `cross_dataset_unseen_DiFF`

**Objetivo:** Implementar e testar o cenário de finetuning para detecção de DeepFakes "não vistos" durante o treinamento, medindo a capacidade de generalização do modelo contra novos métodos de ataque.

**Passos:**

1.  **Criar o Script de Automação:**
    *   [ ] Criar o arquivo `run_finetune_diff.sh` na raiz do projeto.
    *   [ ] Copiar o conteúdo do `run_finetune.sh` existente como base.
    *   [ ] Modificar o script para executar o `main_finetune_DiFF.py`.
        *   Alterar a linha de comando `python` para chamar `./src/fsfm-3c/finuetune/cross_dataset_unseen_DiFF/main_finetune_DiFF.py`.
    *   [ ] Ajustar o `OUTPUT_DIR` para salvar os logs em um diretório específico, como `./src/fsfm-3c/finuetune/cross_dataset_unseen_DiFF/output_finetune_cpu_test_diff/`.

2.  **Adaptação para CPU e Teste Inicial:**
    *   [ ] Revisar o script `main_finetune_DiFF.py` para garantir que ele aceite e utilize argumentos para rodar em CPU (ex: `--device cpu`), similar ao que foi feito no `main_finetune_DfD.py`.
    *   [ ] No script `run_finetune_diff.sh`, configurar o `--data_path` para apontar para o dataset `lfw` já existente, a fim de realizar um primeiro teste funcional.
    *   [ ] Adicionar todos os argumentos necessários para a execução em modo CPU e com poucas épocas (ex: `--num_workers 0`, `--epochs 1`, `--batch_size 4`).

3.  **Execução e Validação:**
    *   [ ] Executar o script: `./run_finetune_diff.sh`.
    *   [ ] Verificar se a execução ocorre sem erros e se os arquivos de log (`log.txt`, `log_detail.txt`, etc.) são criados corretamente no diretório de saída definido.
    *   [ ] Analisar brevemente os logs para confirmar que as métricas (ACC, AUC) estão sendo geradas.

4.  **Atualizar Documentação:**
    *   [ ] Editar o arquivo `README.md` principal do projeto.
    *   [ ] Adicionar uma nova seção explicando como executar o finetuning para o cenário `DiFF` usando o novo script `run_finetune_diff.sh`.

---

## Fase 2: Implementação do Finetuning `cross_domain_FAS`

**Objetivo:** Implementar e testar o cenário de "Face Anti-Spoofing", expandindo a aplicação do modelo para um domínio relacionado de detecção de fraudes.

**Passos:**

1.  **Criar o Script de Automação:**
    *   [ ] Criar o arquivo `run_finetune_fas.sh` na raiz do projeto.
    *   [ ] Usar a estrutura dos outros scripts `.sh` como base.
    *   [ ] Identificar o script Python de entrada principal para esta tarefa (provavelmente `train_vit.py` dentro de `src/fsfm-3c/finuetune/cross_domain_FAS/`).
    *   [ ] Modificar o script para executar o `train_vit.py` com os argumentos corretos.
    *   [ ] Ajustar o `OUTPUT_DIR` para `./src/fsfm-3c/finuetune/cross_domain_FAS/output_finetune_cpu_test_fas/`.

2.  **Adaptação para CPU e Teste Inicial:**
    *   [ ] Revisar o `train_vit.py` e seus arquivos de configuração (`config.py`) para garantir a compatibilidade com a execução em CPU.
    *   [ ] Configurar o `run_finetune_fas.sh` para usar o dataset `lfw` no primeiro teste e ajustar os parâmetros para uma execução leve.

3.  **Execução e Validação:**
    *   [ ] Executar o script: `./run_finetune_fas.sh`.
    *   [ ] Validar a criação e o conteúdo dos logs no diretório de saída.

4.  **Atualizar Documentação:**
    *   [ ] Adicionar uma seção ao `README.md` explicando como executar o finetuning para o cenário `FAS` com o script `run_finetune_fas.sh`.

---

## Fase 3: Implementação do Sistema de Seleção Dinâmica de Datasets

**Objetivo:** Refatorar os scripts de automação para permitir a escolha dinâmica do conjunto de dados a ser usado, facilitando a experimentação e a avaliação de desempenho com datasets progressivamente maiores.

**Passos:**

1.  **Reestruturar os Diretórios de Datasets:**
    *   [ ] Organizar os datasets seguindo a estrutura proposta na análise:
        ```
        datasets/
        ├── pretrain/
        │   └── set_1_lfw/
        └── finetune/
            ├── DfD/
            │   └── set_1_lfw_mock/  # (usando lfw como placeholder)
            ├── DiFF/
            │   └── set_1_lfw_mock/
            └── FAS/
                └── set_1_lfw_mock/
        ```
    *   [ ] Mover os arquivos do dataset `lfw` para a pasta `datasets/pretrain/set_1_lfw/`.

2.  **Adaptar Scripts `.sh` para Aceitar Argumentos:**
    *   [ ] **Modificar `run_pretrain.sh`:**
        *   Alterar o script para aceitar um ID de dataset como primeiro argumento (`DATASET_ID=$1`).
        *   Tornar os caminhos `DATA_PATH` e `OUTPUT_DIR` dinâmicos usando a variável `$DATASET_ID`.
        *   Exemplo de `DATA_PATH`: `./datasets/pretrain/${DATASET_ID}`.
        *   Exemplo de `OUTPUT_DIR`: `./src/fsfm-3c/pretrain/output_${DATASET_ID}`.
    *   [ ] **Modificar `run_finetune.sh`, `run_finetune_diff.sh`, e `run_finetune_fas.sh`:**
        *   Aplicar a mesma lógica para que aceitem um `DATASET_ID=$1`.
        *   Ajustar `DATA_PATH` e `OUTPUT_DIR` de acordo. Exemplo para `run_finetune.sh`:
            *   `DATA_PATH`: `./datasets/finetune/DfD/${DATASET_ID}`
            *   `OUTPUT_DIR`: `./src/fsfm-3c/finuetune/cross_dataset_DfD/output_finetune_${DATASET_ID}`

3.  **Atualizar Documentação:**
    *   [ ] Reescrever as seções de "Como Executar" no `README.md`.
    *   [ ] Explicar a nova estrutura de diretórios de datasets.
    *   [ ] Fornecer exemplos claros de como executar os scripts com o novo parâmetro de ID do dataset.
        *   Ex: `./run_pretrain.sh set_1_lfw`
        *   Ex: `./run_finetune.sh set_1_lfw_mock`

---

## Fase 4: Otimização e Melhoria Contínua

**Objetivo:** Estabelecer um ciclo de melhoria para os resultados do modelo, utilizando a nova estrutura de datasets dinâmicos.

**Passos:**

1.  **Expor Hiperparâmetros nos Scripts `.sh`:**
    *   [ ] Modificar os scripts `.sh` para transformar hiperparâmetros importantes (como `--lr`, `--batch_size`, `--epochs`) em variáveis no topo do arquivo. Isso facilitará a experimentação sem precisar alterar a linha de comando `python`.

2.  **Ciclo de Teste com Novos Datasets:**
    *   [ ] **Pesquisar e Baixar:** Encontrar um dataset de pré-treinamento ligeiramente maior que o LFW (ex: uma amostra do CelebA).
    *   [ ] **Estruturar:** Criar a pasta `datasets/pretrain/set_2_celeba/` e colocar os dados lá.
    *   [ ] **Executar:** Rodar o pré-treinamento com o novo dataset: `./run_pretrain.sh set_2_celeba`.
    *   [ ] **Analisar:** Comparar os logs de `output_set_1_lfw` e `output_set_2_celeba`. Avaliar o impacto no tempo de treinamento e na queda da `loss`.
    *   [ ] **Repetir:** Seguir o mesmo processo para os datasets de finetuning (ex: usando amostras do FaceForensics++), sempre comparando os resultados (ACC, AUC) e o consumo de recursos (CPU, RAM).

3.  **Ajuste de Hiperparâmetros:**
    *   [ ] Com um dataset fixo (ex: `set_2_celeba`), iniciar experimentos alterando um hiperparâmetro de cada vez (ex: aumentar o número de épocas de `1` para `5`, depois ajustar a `learning rate`).
    *   [ ] Documentar cada resultado para encontrar a combinação ótima que o seu hardware suporta.

Este plano fornece um caminho claro e estruturado para alcançar seus objetivos, transformando a implementação funcional atual em uma plataforma de experimentação robusta e escalável.