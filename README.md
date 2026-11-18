# Monitoramento de um serviço de inferência (ResNet-50 + BentoML)

Projeto de MLOps (CESAR School, AV2 — Parte 1). O serviço de classificação de imagens
[servico-classificacao-imagens-bentoml](https://github.com/phelipecscosta/servico-classificacao-imagens-bentoml)
(commit `8f59d10`, tag `sr1.1`) foi copiado para este repositório e ganhou **monitoramento**:

- **Log em arquivo** (`logs/inference.jsonl`): uma linha JSON por pedido, com características
  numéricas/categóricas da imagem recebida (entrada) e o resultado da inferência (saída). O arquivo
  é append-only, tem data e hora ISO 8601 com microssegundos e fuso, rotação segura no Windows e
  nenhum dado pessoal. É suficiente para montar histogramas, boxplots, KS e PSI;
- **`GET /metrics`** no formato Prometheus, com um guia e uma planilha mostrando como calcular a
  taxa de erros e o histograma e o boxplot da latência, e por quanto tempo os dados ficam no servidor.

A escolha das variáveis e do formato do log está justificada numa análise CRISP-DM curta:
[`docs/crisp_dm.md`](docs/crisp_dm.md).

---

## Sumário

- [Monitoramento de um serviço de inferência (ResNet-50 + BentoML)](#monitoramento-de-um-serviço-de-inferência-resnet-50--bentoml)
  - [Sumário](#sumário)
  - [1. Requisitos](#1-requisitos)
  - [2. Do `git clone` ao log crescendo](#2-do-git-clone-ao-log-crescendo)
    - [Passo 1 — Clonar, instalar e validar (terminal 1)](#passo-1--clonar-instalar-e-validar-terminal-1)
    - [Passo 2 — Subir o serviço (terminal 1)](#passo-2--subir-o-serviço-terminal-1)
    - [Passo 3 — Acompanhar o log (terminal 2)](#passo-3--acompanhar-o-log-terminal-2)
    - [Passo 4 — Gerar carga com um comando (terminal 3)](#passo-4--gerar-carga-com-um-comando-terminal-3)
    - [Passo 5 — (Opcional) Analisar o log](#passo-5--opcional-analisar-o-log)
  - [3. O log em arquivo](#3-o-log-em-arquivo)
    - [Onde fica e como é gravado](#onde-fica-e-como-é-gravado)
    - [Exemplo de linha (real)](#exemplo-de-linha-real)
    - [Dicionário de campos](#dicionário-de-campos)
    - [O que NÃO é gravado](#o-que-não-é-gravado)
    - [Ler o log em Python](#ler-o-log-em-python)
  - [4. Verificação dos dados coletados](#4-verificação-dos-dados-coletados)
  - [5. O `/metrics`](#5-o-metrics)
  - [6. Contrato da API](#6-contrato-da-api)
    - [`POST /classify`](#post-classify)
    - [Rotas automáticas do BentoML](#rotas-automáticas-do-bentoml)
  - [7. Comandos disponíveis](#7-comandos-disponíveis)
  - [8. Estrutura do repositório](#8-estrutura-do-repositório)
  - [9. Decisões de projeto e limitações](#9-decisões-de-projeto-e-limitações)
  - [10. Problemas comuns](#10-problemas-comuns)
  - [11. Créditos](#11-créditos)
  - [12. Uso de IA](#12-uso-de-ia)
  - [13. Licença](#13-licença)

---

## 1. Requisitos

| Ferramenta | Versão testada | Obrigatória? |
|---|---|---|
| Python | 3.11.4 | Sim — mas **não precisa instalar**: o `uv` baixa um Python 3.11 se a máquina não tiver |
| git | 2.50.1 | Sim |
| uv | 0.12.23 | Sim |
| just | 1.58.0 | Não — todos os comandos têm equivalente sem ele (seção 7) |

- **Sistema testado:** Windows 11 (x64). Só ferramentas multiplataforma (Linux/macOS também funcionam).
- **Hardware:** roda em **CPU**; GPU não é necessária nem usada (escolha intencional, para reproduzir em qualquer máquina).
- **Internet:** necessária na primeira execução (dependências ~300 MB e modelo ~100 MB).

```powershell
# Windows (PowerShell)
winget install --id=astral-sh.uv -e
winget install --id=Casey.Just -e      # opcional
```

```bash
# Linux / macOS
curl -LsSf https://astral.sh/uv/install.sh | sh
uv tool install rust-just               # opcional
```

> Após instalar, **feche e reabra o terminal** (ou o VS Code) para que o PATH seja atualizado.

---

## 2. Do `git clone` ao log crescendo

São necessários **três terminais**: um para o serviço, um para acompanhar o log e um para gerar carga.
No VS Code, cada terminal novo é aberto com o botão `+` do painel *Terminal*.

### Passo 1 — Clonar, instalar e validar (terminal 1)

```bash
git clone https://github.com/phelipecscosta/mlops-monitoramento-inferencia-bentoml.git
cd mlops-monitoramento-inferencia-bentoml
just all            # sem just: uv sync --locked && uv run pytest -v
```

Instala o ambiente com **exatamente** as versões do `uv.lock` e roda os 34 testes (serviço em memória,
predições reais, formato do log, privacidade, append-only e rotação). Esperado ao final: `35 passed`. Tempo medido num clone limpo, com caches vazios e sem Python 3.11 instalado (download do Python, das
dependências e do modelo incluídos): **cerca de 3 minutos**. Nas execuções seguintes, segundos.
Na primeira vez, o modelo (~100 MB) é baixado do Hugging Face.

### Passo 2 — Subir o serviço (terminal 1)

```bash
just serve          # sem just: uv run bentoml serve service:ImageClassifier
```

Aguarde a linha `Starting production HTTP BentoServer ... listening on http://localhost:3000`.
O terminal fica ocupado pelo servidor. Para parar: `Ctrl+C`.

### Passo 3 — Acompanhar o log (terminal 2)

```bash
just tail-log       # sem just (Windows): Get-Content logs/inference.jsonl -Wait -Tail 3
                    # sem just (Linux/macOS): tail -n 3 -f logs/inference.jsonl
```
O arquivo é criado vazio quando o serviço sobe; o comando fica aguardando as linhas do passo 4.

### Passo 4 — Gerar carga com um comando (terminal 3)

```bash
just load           # sem just: uv run python scripts/generate_load.py
```

Envia **900 pedidos** (3 cenários × 300, cerca de 1,5 a 2 min em CPU). O terminal 2 mostra as linhas
chegando em tempo real. Resultado esperado — idêntico em qualquer máquina, porque a carga usa
semente fixa:

```text
[reference] concluído em ... s; respostas: {200: 294, 400: 6}
[normal] concluído em ... s; respostas: {200: 287, 400: 13}
[drift] concluído em ... s; respostas: {200: 292, 400: 8}
```

| Cenário | O que envia | Para quê |
|---|---|---|
| `reference` | Variações leves (recorte, espelhamento, rotação, brilho, contraste, saturação, resolução) das 8 imagens de `samples/` | Conjunto de referência para KS e PSI |
| `normal` | A mesma distribuição, com outra semente | Controle: **não** deve acusar drift |
| `drift` | As mesmas variações + escurecimento, desfoque, dessaturação e compressão forte | **Deve** acusar drift |

Cerca de 3% dos pedidos são inválidos de propósito (arquivo que não é imagem ou `top_k` fora da faixa)
para que o log também registre erros. Opções: `just load --scenario drift --n 100`.

### Passo 5 — (Opcional) Analisar o log

Abra [`notebooks/verificacao_log.ipynb`](notebooks/verificacao_log.ipynb) no VS Code, selecione o
kernel do `.venv` e execute **Run All**. Ver seção 4.

---

## 3. O log em arquivo

### Onde fica e como é gravado

| Item | Valor |
|---|---|
| Arquivo | `logs/inference.jsonl` (UTF-8), relativo à pasta onde o serviço foi iniciado |
| Formato | **JSON Lines**: um objeto JSON completo por linha |
| Granularidade | **Uma linha por pedido** a `POST /classify`, com sucesso **ou** erro (inclusive os `400` de validação) |
| Data e hora | `ts` em ISO 8601 com microssegundos e fuso: `2026-11-17T23:07:52.337689-03:00` (instante em que o pedido terminou) |
| Append-only | Novos eventos só acrescentam linhas; o arquivo nunca é reescrito (coberto por teste) |
| Rotação | Ao passar de 10 MB, o arquivo vira `inference.jsonl.1` (os antigos sobem para `.2`, `.3`…) e um novo é criado; são mantidos 5 arquivos antigos. Cada linha tem ~0,7 KB, ou seja, ~14 mil pedidos por arquivo |
| Arquivo `.__inference.jsonl.lock` | Trava usada pela rotação; vazio e normal |

Configuração por variáveis de ambiente (opcionais; ver [`.env.example`](.env.example)):

| Variável | Padrão | Significado |
|---|---|---|
| `INFERENCE_LOG_PATH` | `logs/inference.jsonl` | Caminho do arquivo |
| `INFERENCE_LOG_MAX_BYTES` | `10485760` (10 MB) | Tamanho que dispara a rotação |
| `INFERENCE_LOG_BACKUP_COUNT` | `5` | Quantos arquivos antigos guardar |

### Exemplo de linha (real)

```json
{"ts":"2026-11-17T23:07:52.337689-03:00","event":"inference","schema_version":"1","request_id":"b5d6c40226474723a8af48f844ae0765","endpoint":"/classify","scenario":"normal","model_id":"microsoft/resnet-50","model_revision":"34c2154c194f829b11125337b98c8f5f9965ff19","top_k":5,"file_bytes":96772,"img_format":"JPEG","img_mode":"RGB","width":640,"height":480,"aspect_ratio":1.3333,"exif_rotated":false,"mean_L":115.4401,"std_L":41.3727,"mean_R":115.0459,"mean_G":116.6894,"mean_B":110.0711,"sharpness":2309.5229,"colorfulness":46.201,"status":"ok","http_status":200,"error_type":null,"top1_label":"sandbar, sand bar","top1_score":0.5707,"margin_top1_top2":0.2182,"entropy":1.2499,"features_ms":25.246,"inference_ms":164.9,"latency_ms":390.751}
```

### Dicionário de campos

Toda linha tem **as mesmas chaves, na mesma ordem**; campos sem valor ficam `null` (ex.: características
de uma imagem inválida). A justificativa de cada variável está em [`docs/crisp_dm.md`](docs/crisp_dm.md).

| Grupo | Campo | Tipo | Descrição |
|---|---|---|---|
| Evento | `ts` | data/hora | ISO 8601, µs e fuso |
| | `event` | texto | Tipo do evento (`inference`) |
| | `schema_version` | texto | Versão do formato do log (`1`) |
| | `request_id` | texto | UUID4 aleatório; também devolvido ao cliente no cabeçalho `X-Request-ID` |
| | `endpoint` | texto | `/classify` |
| | `scenario` | categórica | `reference`, `normal`, `drift` (cabeçalho `X-Scenario`) ou `unspecified` |
| | `model_id`, `model_revision` | texto | Modelo e commit exato que respondeu |
| Entrada | `top_k` | inteiro | Parâmetro do pedido |
| | `file_bytes` | inteiro | Tamanho do arquivo enviado |
| | `img_format`, `img_mode` | categórica | Formato (JPEG, PNG…) e modo de cor (RGB, L, RGBA…) |
| | `exif_rotated` | booleano | A foto tinha rotação no EXIF |
| | `width`, `height`, `aspect_ratio` | numérica | Geometria da imagem (já em pé) |
| | `mean_L`, `std_L` | numérica | Brilho médio e contraste (luma BT.601, 0–255) |
| | `mean_R`, `mean_G`, `mean_B` | numérica | Médias dos canais de cor (0–255) |
| | `sharpness` | numérica | Nitidez: variância do Laplaciano |
| | `colorfulness` | numérica | Colorido de Hasler e Süsstrunk (0 = cinza) |
| Saída | `status`, `http_status`, `error_type` | categórica | `ok`/`error`, código HTTP e causa do erro (`invalid_image`, `validation_error`, `internal_error`, …) |
| | `top1_label`, `top1_score` | categórica / numérica | Classe mais provável e sua confiança |
| | `margin_top1_top2` | numérica | Diferença de confiança entre a 1ª e a 2ª classe |
| | `entropy` | numérica | Incerteza do softmax nas 1000 classes (0 a ln 1000 ≈ 6,91) |
| | `features_ms`, `inference_ms`, `latency_ms` | numérica | Tempo das características, do modelo e total do pedido no servidor |

As características de pixel são calculadas sobre uma cópia reduzida a 512 px no maior lado, a mesma
imagem (já girada pelo EXIF e em RGB) que o modelo recebe.

### O que NÃO é gravado

A imagem (bytes ou pixels), o nome original do arquivo, o IP e o User-Agent do cliente, a lista completa
de classes e a mensagem/traceback dos erros. O cabeçalho `X-Scenario` só é aceito numa lista fechada de
valores. A privacidade é coberta por teste (um arquivo chamado `foto_da_maria_silva.jpg` não deixa
rastro no log).

### Ler o log em Python

```python
import pandas as pd
from pathlib import Path

files = sorted(p for p in Path("logs").glob("inference.jsonl*") if not p.name.endswith(".lock"))
df = pd.concat([pd.read_json(f, lines=True) for f in files], ignore_index=True)
```

---

## 4. Verificação dos dados coletados

O notebook [`notebooks/verificacao_log.ipynb`](notebooks/verificacao_log.ipynb) verifica os critérios de
suficiência definidos no CRISP-DM sobre a carga do passo 4:

| Critério | Resultado |
|---|---|
| Leitura direta pelo pandas, somando arquivos rotacionados | OK |
| Campos nulos nas linhas de sucesso / erros sem causa / `request_id` repetidos | 0 / 0 / 0 |
| Volume mínimo de 200 sucessos por cenário | 287 a 305 |
| `reference` × `normal` (mesma distribuição) | PSI < 0,1 nas variáveis de dados: **sem alarme falso** |
| `reference` × `drift` | PSI de 4 a 8 nas entradas de imagem; geometria (não alterada) estável: **sensível e específico** |
| Efeito do drift no modelo | Mediana da confiança de ~0,8 para 0,25; da entropia de ~0,7 para 4,0 |

Achado: `latency_ms` acusa diferença até entre cenários equivalentes, porque depende do estado da máquina
(aquecimento, carga concorrente). Latência se monitora por limites absolutos e pelo `/metrics`, não por
drift contra uma referência.

---

## 5. O `/metrics`

`GET http://localhost:3000/metrics` expõe as métricas nativas do BentoML no formato Prometheus. O guia
[`docs/metrics.md`](docs/metrics.md) e a planilha [`docs/metrics_analise.xlsx`](docs/metrics_analise.xlsx)
mostram, com números reais, como:

- calcular a **taxa de erros** (total e só de 5xx; acumulada e por intervalo) a partir de `bentoml_service_request_total`;
- montar o **histograma** da latência subtraindo buckets cumulativos de `bentoml_service_request_duration_seconds`;
- estimar o **boxplot** (quartis e percentis) por interpolação dentro dos buckets, o mesmo método do `histogram_quantile` do Prometheus;
- saber **por quanto tempo os dados ficam**: desde a subida do servidor até o próximo reinício, quando são apagados. Não há janela de tempo nem histórico.

Evidências: [`docs/evidencias/metrics_apos_carga.txt`](docs/evidencias/metrics_apos_carga.txt) e
[`docs/evidencias/metrics_apos_reinicio.txt`](docs/evidencias/metrics_apos_reinicio.txt).

---

## 6. Contrato da API

### `POST /classify`

Entrada `multipart/form-data`:

| Campo | Tipo | Obrigatório | Descrição |
|---|---|---|---|
| `image` | arquivo | Sim | JPEG, PNG, WEBP ou BMP, validado pelo **conteúdo** |
| `top_k` | inteiro | Não (padrão `5`) | Quantas classes devolver, de 1 a 10 |

Cabeçalhos de monitoramento:

| Cabeçalho | Direção | Uso |
|---|---|---|
| `X-Scenario` | pedido (opcional) | Marca o cenário no log: `reference`, `normal` ou `drift`; outro valor vira `unspecified` |
| `X-Request-ID` | resposta | Id do pedido no log, para auditoria |

```powershell
# Windows (PowerShell) — use curl.exe, NÃO curl
curl.exe -X POST http://localhost:3000/classify -H "X-Scenario: normal" -F "image=@samples/06_praia.jpg" -F "top_k=3"
```

Saída `200 OK` (real):

```json
{"model":{"id":"microsoft/resnet-50","revision":"34c2154c194f829b11125337b98c8f5f9965ff19"},"inference_ms":164.9,"predictions":[{"label":"sandbar, sand bar","score":0.5707},{"label":"seashore, coast, seacoast, sea-coast","score":0.3525},{"label":"promontory, headland, head, foreland","score":0.0109}]}
```

Erros de entrada (arquivo que não é imagem, imagem ausente, `top_k` fora da faixa) devolvem `400` no
formato `{"error", "detail"}`, com o campo inválido em `loc`, e também são registrados no log.

### Rotas automáticas do BentoML

| Rota | Uso |
|---|---|
| `GET /` | Swagger (documentação interativa) |
| `GET /metrics` | Métricas no formato Prometheus (seção 5) |
| `GET /readyz` | `200` quando o modelo está carregado |

---

## 7. Comandos disponíveis

| Comando | Sem `just` | O que faz |
|---|---|---|
| `just` | — | Lista as receitas |
| `just setup` | `uv sync --locked` | Instala o ambiente exato do `uv.lock` |
| `just all` | `uv sync --locked && uv run pytest -v` | Instala e roda os 35 testes |
| `just test` | `uv run pytest -v` | Testes (serviço em memória; não precisa do servidor) |
| `just serve` | `uv run bentoml serve service:ImageClassifier` | Sobe o serviço em `http://localhost:3000` |
| `just load` | `uv run python scripts/generate_load.py` | Gera carga nos 3 cenários |
| `just tail-log` | Windows: `Get-Content logs/inference.jsonl -Wait -Tail 3` · Linux/macOS: `tail -n 3 -f logs/inference.jsonl` | Acompanha o log em tempo real |
| `just demo` | `uv run python scripts/demo.py` | Envia as amostras ao serviço rodando e resume as respostas |
| `just check-model` | `uv run python scripts/check_model.py` | Testa o modelo isolado, fora do BentoML |

O `justfile` desliga a telemetria de uso do BentoML (`BENTOML_DO_NOT_TRACK=true`). Sem `just`, defina a
variável antes de `bentoml serve` — PowerShell: `$env:BENTOML_DO_NOT_TRACK="true"`; Linux/macOS:
`export BENTOML_DO_NOT_TRACK=true`.

---

## 8. Estrutura do repositório

```
├── service.py                    # Serviço BentoML: /classify + ligação do monitoramento
├── monitoring/
│   ├── features.py               # Características de entrada (pixels, metadados) e de saída (confiança, margem, entropia)
│   ├── logging_setup.py          # Logger JSON Lines: ISO 8601 µs + fuso, rotação segura no Windows
│   └── middleware.py             # Middleware ASGI: uma linha de log por pedido, inclusive erros
├── scripts/
│   ├── generate_load.py          # Carga reproduzível: reference, normal, drift
│   ├── demo.py                   # Demonstração contra o serviço rodando
│   ├── check_model.py            # Diagnóstico do modelo isolado
│   ├── classify_images.py        # Classifica uma pasta de imagens (herdado da base)
│   └── prepare_samples.py        # Sanitiza fotos (EXIF/GPS) (herdado da base)
├── tests/
│   ├── conftest.py               # Um serviço em memória para todos os testes; log em pasta temporária
│   ├── test_service.py           # Contrato da API e erros (15 testes)
│   ├── test_features.py          # Testes de sanidade das características (10)
│   └── test_logging.py           # Formato, privacidade, append-only e rotação do log (10)
├── notebooks/verificacao_log.ipynb  # Verificação de suficiência: histogramas, boxplots, KS, PSI
├── samples/                      # 8 imagens + CREDITS.md
├── docs/
│   ├── crisp_dm.md               # Análise CRISP-DM: variáveis e formato do log
│   ├── metrics.md                # Guia de análise do /metrics
│   ├── metrics_analise.xlsx      # Planilha: taxa de erro, histograma e boxplot da latência, retenção
│   ├── problemas_e_ajustes.md    # Problemas encontrados, diagnóstico e decisões
│   └── evidencias/               # Respostas e /metrics reais
├── justfile                      # Receitas
├── pyproject.toml / uv.lock      # Dependências declaradas e travadas (Python 3.11, torch CPU)
└── .env.example                  # Variáveis opcionais (só nomes)
```

`logs/` e `data/` são gerados em tempo de execução e não são versionados.

---

## 9. Decisões de projeto e limitações

| Decisão | Motivo |
|---|---|
| JSON Lines | Append-only seguro, aceita novos campos, preserva tipos e é lido direto por pandas, Loki e Elasticsearch |
| `concurrent-log-handler` em vez do `RotatingFileHandler` padrão | No Windows, renomear arquivo aberto por outro programa (ex.: `Get-Content -Wait`) falha; este handler adia a rotação sem perder eventos (teste automatizado) |
| Logger configurado no `__init__` do serviço | O `bentoml serve` importa o módulo em dois processos; só o que atende pedidos deve abrir o arquivo |
| Middleware ASGI + `contextvars` | Erros `400` de validação acontecem antes do método; o middleware registra todos os pedidos |
| Características sobre cópia de 512 px | Custo baixo (~10 ms) e valores comparáveis entre resoluções |
| Carga gerada em memória com semente fixa | Reproduzível em qualquer máquina e sem gravar derivados de imagens CC BY-SA |
| Telemetria do BentoML desligada | Nada do serviço sai da máquina sem necessidade |

| Limitação | Consequência |
|---|---|
| Referência derivada de só 8 imagens | Pouca diversidade de conteúdo; não substitui o conjunto de treino |
| `X-Scenario` vem do cliente | Só se justifica num ambiente de demonstração |
| `top1_label` tem até 1000 categorias | Para o PSI, agrupar as mais frequentes e juntar o resto em "outras" |
| `ts` é o fim do pedido | O início é `ts − latency_ms` |
| Log interno do BentoML registra 4xx como `ERROR` com traceback | Ruído no console (limitação herdada; não afeta `logs/inference.jsonl`) |

Os problemas reais encontrados no desenvolvimento estão em [`docs/problemas_e_ajustes.md`](docs/problemas_e_ajustes.md).

---

## 10. Problemas comuns

| Sintoma | Causa | Solução |
|---|---|---|
| `curl` no PowerShell dá erro de parâmetro | `curl` é apelido de `Invoke-WebRequest` | Usar **`curl.exe`** |
| `uv` ou `just` "not recognized" | PATH não atualizado | Fechar e reabrir o terminal / VS Code |
| `just load` diz "Serviço indisponível" | Servidor não está rodando | `just serve` em outro terminal |
| `just tail-log` diz que o arquivo não existe | O serviço não foi iniciado nesta pasta | `just serve` (o arquivo é criado na subida) |
| Aviso `VIRTUAL_ENV=... does not match the project environment path` | Outro `.venv` está ativado no terminal (o VS Code ativa automaticamente) | Inofensivo: o `uv` ignora o ambiente ativado e usa o `.venv` do projeto; para sumir, `deactivate` |
| Porta 3000 ocupada | Outro processo usa a porta | `uv run bentoml serve service:ImageClassifier --port 3001` e `just load --url http://localhost:3001` |
| Mensagem `rename failed. File in use?` | O log estava aberto por outro programa na hora da rotação | Inofensivo: a rotação é adiada e nenhum evento se perde |
| Não consigo apagar o `.venv` (arquivo em uso) | Um Python do projeto continua rodando (ex.: kernel do notebook) | Fechar o notebook e os terminais; ou encerrar os processos do `.venv` (ver `docs/problemas_e_ajustes.md`) |
| Primeira execução lenta | Download do modelo (~100 MB) | Só na primeira vez |

O modelo fica em cache em `~/.cache/huggingface/hub/models--microsoft--resnet-50` (~100 MB) e os pacotes
no cache do `uv` (`uv cache dir`). Apagar a pasta do projeto não remove esses caches.

---

## 11. Créditos

- **Serviço base:** [servico-classificacao-imagens-bentoml](https://github.com/phelipecscosta/servico-classificacao-imagens-bentoml) (mesmo autor).
- **Características de imagem:** adaptadas de [mlops-bloco2-atividade1-processamento-imagens](https://github.com/phelipecscosta/mlops-bloco2-atividade1-processamento-imagens) (mesmo autor), que segue o repositório de referência [lgallindo/bentoML-processamento-imagens](https://github.com/lgallindo/bentoML-processamento-imagens).
- **Imagens:** fotos próprias e imagens do Wikimedia Commons sob CC BY-SA 4.0, CC0 1.0 e CC BY 2.0; autores e links em [`samples/CREDITS.md`](samples/CREDITS.md).
- **Modelo:** ResNet-50 (`microsoft/resnet-50`, Apache-2.0), baixado em tempo de execução e não redistribuído.

---

## 12. Uso de IA

**Ferramenta:** Claude (Anthropic), modelo Claude Opus 5.5, acessado por meio da plataforma claude.ai, em um projeto configurado com instruções específicas de orientação técnica e apoio didático.

**O que foi solicitado:** orientação estruturada para evolução de um serviço base para um serviço monitorado, incluindo análise baseada em CRISP-DM, definição das variáveis e do formato de registro dos logs, implementação do módulo de características, do sistema de logging com rotação, do middleware, dos testes automatizados, do gerador de carga e do notebook de validação; elaboração do guia e da planilha de análise do endpoint `/metrics`; além da produção de uma versão inicial deste README.

**Avaliação crítica:**

*Onde auxiliou.*

A principal contribuição ocorreu na identificação antecipada de problemas que poderiam surgir apenas durante a execução local do projeto. As soluções propostas foram previamente avaliadas em um ambiente de testes próprio, utilizando BentoML 1.4.39 e um modelo simulado, permitindo a identificação de comportamentos que dificilmente seriam previstos sem experimentação.

Entre os pontos identificados destacam-se: a importação do arquivo `service.py` em dois processos distintos durante a execução do `bentoml serve`, fator determinante para a definição da configuração de logging; o fato de erros de validação `400` não alcançarem o método `classify`, tornando necessária a implementação de um middleware; a duplicação das métricas do Prometheus quando o serviço é iniciado mais de uma vez dentro do mesmo processo de testes; e o envio padrão de telemetria para `t.bentoml.com`.

Também foi identificado antecipadamente o risco associado à rotação de arquivos de log em ambientes Windows quando o arquivo está aberto por outro processo, sendo sugerida posteriormente uma validação automatizada especificamente para esse cenário.

Para a análise da retenção das métricas expostas em `/metrics`, a ferramenta consultou o código-fonte do BentoML em vez de assumir um comportamento padrão, propondo posteriormente uma demonstração prática por meio da reinicialização do servidor e observação da reinicialização dos contadores.

*Onde apresentou limitações ou exigiu correções.*

- Ao sugerir a recriação do ambiente `.venv`, não foi considerado que o kernel do notebook permanecia em execução. Como consequência, o comando `Remove-Item` removeu parcialmente o ambiente virtual, resultando no erro `No pyvenv.cfg file`. A causa foi identificada posteriormente por meio da análise da mensagem de erro e do encerramento manual dos processos associados ao ambiente virtual (problema 15 documentado em `docs/problemas_e_ajustes.md`).

- A primeira versão do README propunha apenas contornar um problema existente. O comando `just tail-log` falhava antes da realização da primeira requisição, e a documentação apenas alertava sobre essa condição. O problema foi identificado durante testes em um clone limpo do repositório, levando posteriormente à correção adequada por meio da criação automática do arquivo durante a inicialização do serviço, acompanhada de testes específicos.

- A proposta inicial da análise CRISP-DM incluía o uso dos indicadores KS e PSI para monitoramento de latência. A avaliação experimental demonstrou que essas métricas indicavam alterações significativas mesmo entre cenários equivalentes, em função da dependência do estado da máquina. A conclusão foi revisada no notebook, passando a considerar limites absolutos e métricas disponibilizadas pelo endpoint `/metrics` como mecanismos mais adequados para monitoramento de latência.

- Em algumas etapas, as instruções fornecidas não incluíam critérios claros de validação. Nesses casos, foi necessário solicitar orientações adicionais para determinar como verificar corretamente o funcionamento das implementações realizadas.

*Onde houve direcionamento do aluno.*

Foi solicitada a inclusão de uma seção específica descrevendo as fases do método CRISP-DM. Também foi requerida uma avaliação sobre o reaproveitamento de um repositório previamente desenvolvido, resultando na decisão de manter o serviço atual e reutilizar apenas as funções relacionadas à extração de características.

Além disso, considerando a relação entre esforço necessário e pontuação atribuída, optou-se por não realizar as atividades opcionais.

*O que foi verificado de forma independente.*

Cada bloco de código foi executado individualmente, com validação das saídas produzidas antes do prosseguimento para as etapas seguintes.

A geração de carga utilizando semente fixa produziu resultados consistentes em três execuções independentes, incluindo um teste realizado em clone limpo do repositório, com caches vazios e sem Python 3.11 previamente instalado, por meio da execução de `just all`, concluída em aproximadamente três minutos.

Os testes automatizados foram executados com sucesso em ambiente Windows, incluindo o cenário específico de rotação de logs com o arquivo aberto por um programa externo.

A persistência das métricas expostas em `/metrics` foi avaliada por meio da reinicialização do serviço, confirmando a reinicialização dos contadores. Adicionalmente, os experimentos conduzidos no notebook demonstraram que as variáveis selecionadas são capazes de detectar situações de *drift* sem gerar alarmes indevidos.

*Aprendizados obtidos.*

A atividade evidenciou que o monitoramento de modelos vai além da medição de latência, exigindo a definição criteriosa das informações a serem registradas em cada requisição e a validação experimental de que tais variáveis são capazes de detectar mudanças relevantes no comportamento do sistema.

Também foi possível compreender, de forma prática, a diferença entre o registro detalhado em arquivos de log e a exposição de métricas por meio do endpoint `/metrics`. Enquanto o primeiro mantém registros individuais persistentes entre reinicializações, o segundo fornece agregações padronizadas e de baixo custo computacional, porém reiniciadas a cada novo ciclo do serviço.

Por fim, a atividade reforçou a importância de validar comportamentos específicos do sistema operacional em vez de assumi-los como corretos, especialmente em cenários que envolvem manipulação de arquivos, bloqueios de acesso e compatibilidade entre diferentes ambientes de execução.

---

## 13. Licença

Código sob licença **MIT** (ver [`LICENSE`](LICENSE)).
O modelo ResNet-50 (`microsoft/resnet-50`) é distribuído pela Microsoft sob licença **Apache-2.0**.
