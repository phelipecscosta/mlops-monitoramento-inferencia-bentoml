"""Testes do log de inferência: formato, conteúdo, privacidade, append-only e rotação."""

import json
import re
from pathlib import Path

import pytest

from monitoring.logging_setup import setup_inference_logger
from monitoring.middleware import FIELDS

SAMPLES = Path(__file__).parent.parent / "samples"
IMAGE = SAMPLES / "06_praia.jpg"
# ISO 8601 com microssegundos (6 dígitos) e fuso (+HH:MM ou -HH:MM)
ISO_US_TZ = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{6}[+-]\d{2}:\d{2}$")


def read_lines(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def post(client, path=IMAGE, data=None, headers=None, name=None):
    with path.open("rb") as f:
        return client.post("/classify", files={"image": (name or path.name, f)}, data=data or {}, headers=headers or {})


def last_event(log_path: Path) -> dict:
    return read_lines(log_path)[-1]


# ------------------------------- Formato e conteúdo -------------------------------
def test_sucesso_gera_linha_completa(client, log_path):
    resp = post(client, headers={"X-Scenario": "reference"})
    assert resp.status_code == 200
    ev = last_event(log_path)
    assert list(ev) == ["ts", *FIELDS]           # todas as chaves, sempre na mesma ordem
    assert ISO_US_TZ.match(ev["ts"])             # ISO 8601 com µs e fuso
    assert ev["request_id"] == resp.headers["x-request-id"]  # o cliente recebe o mesmo id
    assert ev["scenario"] == "reference"
    assert (ev["status"], ev["http_status"], ev["error_type"]) == ("ok", 200, None)
    assert ev["file_bytes"] == IMAGE.stat().st_size
    assert ev["img_format"] == "JPEG" and ev["width"] == 640
    assert ev["top1_label"] == resp.json()["predictions"][0]["label"]
    assert ev["top1_score"] == pytest.approx(resp.json()["predictions"][0]["score"], abs=1e-4)
    assert ev["latency_ms"] >= ev["inference_ms"] > 0


def test_cenario_fora_da_lista_vira_unspecified(client, log_path):
    post(client, headers={"X-Scenario": "<script>qualquer coisa</script>"})
    assert last_event(log_path)["scenario"] == "unspecified"


# ------------------------------- Erros também são registrados -------------------------------
def test_arquivo_invalido_registra_400(client, log_path, tmp_path):
    fake = tmp_path / "nao_imagem.jpg"
    fake.write_text("isto não é uma imagem")
    assert post(client, path=fake).status_code == 400
    ev = last_event(log_path)
    assert (ev["status"], ev["http_status"], ev["error_type"]) == ("error", 400, "invalid_image")
    assert ev["top1_label"] is None and ev["mean_L"] is None  # saída e características ficam nulas


def test_top_k_invalido_registra_400(client, log_path):
    assert post(client, data={"top_k": "50"}).status_code == 400
    ev = last_event(log_path)
    assert (ev["http_status"], ev["error_type"]) == (400, "validation_error")


def test_rotas_que_nao_sao_de_inferencia_nao_geram_linha(client, log_path):
    antes = len(read_lines(log_path))
    client.get("/healthz")  # rota de saúde do BentoML: não é inferência
    assert len(read_lines(log_path)) == antes


# ------------------------------- Privacidade -------------------------------
def test_nome_do_arquivo_nao_vai_para_o_log(client, log_path):
    post(client, name="foto_da_maria_silva.jpg")
    assert "maria" not in log_path.read_text(encoding="utf-8").lower()


# ------------------------------- Append-only -------------------------------
def test_novos_pedidos_so_acrescentam_linhas(client, log_path):
    antes = log_path.read_bytes()
    post(client)
    depois = log_path.read_bytes()
    assert depois.startswith(antes) and len(depois) > len(antes)  # o início do arquivo não muda


# ------------------------------- Rotação -------------------------------
def test_rotacao_nao_perde_eventos(tmp_path, restore_logger):
    path = tmp_path / "rot.jsonl"
    logger = setup_inference_logger(path, max_bytes=2_000, backup_count=50)
    for i in range(200):
        logger.info("e", extra={"event": {"i": i, "pad": "x" * 50}})
    files = sorted(tmp_path.glob("rot.jsonl*"))
    assert len(files) > 1  # houve rotação
    eventos = sorted(e["i"] for f in files for e in read_lines(f))
    assert eventos == list(range(200))  # nenhum evento perdido nem duplicado


def test_rotacao_com_arquivo_aberto_por_leitor(tmp_path, restore_logger):
    # No Windows, renomear um arquivo aberto por outro programa (ex.: Get-Content -Wait) falha.
    # O handler deve continuar gravando sem perder eventos, mesmo que a rotação seja adiada.
    path = tmp_path / "rot.jsonl"
    logger = setup_inference_logger(path, max_bytes=2_000, backup_count=50)
    logger.info("e", extra={"event": {"i": -1}})
    with path.open("r", encoding="utf-8"):  # "leitor" segurando o arquivo aberto
        for i in range(100):
            logger.info("e", extra={"event": {"i": i, "pad": "x" * 50}})
    for i in range(100, 150):  # leitor fechou: a rotação volta a acontecer
        logger.info("e", extra={"event": {"i": i, "pad": "x" * 50}})
    eventos = sorted(e["i"] for f in tmp_path.glob("rot.jsonl*") for e in read_lines(f))
    assert eventos == list(range(-1, 150))


def test_arquivo_existe_desde_a_subida(tmp_path, restore_logger):
    # "just tail-log" precisa do arquivo antes do primeiro pedido
    path = tmp_path / "novo" / "inference.jsonl"
    setup_inference_logger(path)
    assert path.exists() and path.stat().st_size == 0
