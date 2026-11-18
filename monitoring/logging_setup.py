"""Configuração do log de inferência: JSON Lines, append-only, com rotação segura no Windows."""

import datetime as dt
import json
import logging
import os
from pathlib import Path

from concurrent_log_handler import ConcurrentRotatingFileHandler

LOGGER_NAME = "inference"  # logger próprio: não se mistura com o log interno do BentoML

# Valores padrão, todos sobrescrevíveis por variável de ambiente (ver .env.example)
DEFAULT_PATH = "logs/inference.jsonl"
DEFAULT_MAX_BYTES = 10 * 1024 * 1024  # 10 MB por arquivo
DEFAULT_BACKUP_COUNT = 5              # arquivos antigos guardados: inference.jsonl.1 ... .5


class JsonLinesFormatter(logging.Formatter):
    """Transforma cada registro em UMA linha JSON, com data e hora ISO 8601 (µs + fuso)."""

    def format(self, record: logging.LogRecord) -> str:
        # record.created = instante do registro (time.time()); astimezone() acrescenta o fuso local
        ts = dt.datetime.fromtimestamp(record.created).astimezone().isoformat(timespec="microseconds")
        event = getattr(record, "event", {"message": record.getMessage()})  # campos vêm em extra={"event": ...}
        # ensure_ascii=False: acentos legíveis; separators: linha compacta, sem espaços
        return json.dumps({"ts": ts, **event}, ensure_ascii=False, separators=(",", ":"))


def setup_inference_logger(
    path: str | os.PathLike | None = None,
    max_bytes: int | None = None,
    backup_count: int | None = None,
) -> logging.Logger:
    """Cria (ou recria) o logger de inferência. Chamar UMA vez, no processo que atende os pedidos."""
    path = Path(path or os.getenv("INFERENCE_LOG_PATH", DEFAULT_PATH))
    max_bytes = max_bytes if max_bytes is not None else int(os.getenv("INFERENCE_LOG_MAX_BYTES", DEFAULT_MAX_BYTES))
    backup_count = (
        backup_count if backup_count is not None else int(os.getenv("INFERENCE_LOG_BACKUP_COUNT", DEFAULT_BACKUP_COUNT))
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    # Cria o arquivo vazio já na subida (se não existir), para que "just tail-log" funcione antes do
    # primeiro pedido. touch() num arquivo existente só atualiza a data: nada é reescrito (append-only).
    path.touch(exist_ok=True)

    logger = logging.getLogger(LOGGER_NAME)
    for old in list(logger.handlers):  # reconfiguração (ex.: testes): fecha o arquivo anterior
        logger.removeHandler(old)
        old.close()

    handler = ConcurrentRotatingFileHandler(
        str(path),
        mode="a",                 # append: nunca reescreve o que já existe
        maxBytes=max_bytes,       # ao passar deste tamanho, o arquivo é renomeado e um novo é criado
        backupCount=backup_count,
        encoding="utf-8",
    )
    handler.setFormatter(JsonLinesFormatter())
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False  # não repete as linhas no console nem no log do BentoML
    return logger
