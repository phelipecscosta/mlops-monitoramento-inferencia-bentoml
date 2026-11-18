# Windows: receitas executadas pelo PowerShell (Linux/macOS: sh, o padrão do just)
set windows-shell := ["powershell.exe", "-NoLogo", "-NoProfile", "-Command"]

# Carrega variáveis do .env (se existir) em todas as receitas
set dotenv-load := true

# Desliga a telemetria de uso do BentoML (nenhum dado sai da máquina)
export BENTOML_DO_NOT_TRACK := "true"

# Lista as receitas disponíveis
default:
    @just --list

# Instala o ambiente EXATO do uv.lock (cria o .venv; falha se o lock estiver desatualizado)
setup:
    uv sync --locked

# Testa o modelo isolado, fora do BentoML (diagnóstico)
check-model:
    uv run python scripts/check_model.py

# Sobe o serviço em http://localhost:3000 — Swagger na mesma URL (Ctrl+C para parar)
serve:
    uv run bentoml serve service:ImageClassifier

# Roda os testes automatizados (sobe o serviço em memória; não precisa de 'just serve')
test:
    uv run pytest -v

# Envia as amostras ao serviço rodando (antes, rode 'just serve' em outro terminal)
demo:
    uv run python scripts/demo.py

# COMANDO ÚNICO: instala o ambiente e valida o serviço de ponta a ponta com predições reais
all: setup test


# Gera carga nos 3 cenários (reference, normal, drift), 300 pedidos cada. Ex.: just load --scenario drift --n 100
load *args:
    uv run python scripts/generate_load.py {{args}}

# Acompanha o log crescendo em tempo real (Ctrl+C para sair)
[windows]
tail-log:
    Get-Content logs/inference.jsonl -Wait -Tail 3

[unix]
tail-log:
    tail -n 3 -f logs/inference.jsonl
