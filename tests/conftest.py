"""Fixtures compartilhadas: UM serviço em memória para todos os testes, com log em pasta temporária."""

from pathlib import Path

import pytest
from starlette.testclient import TestClient


@pytest.fixture(scope="session")
def log_path(tmp_path_factory) -> Path:
    # Os testes nunca escrevem no logs/ real do projeto
    return tmp_path_factory.mktemp("logs") / "inference.jsonl"


@pytest.fixture(scope="session")
def client(log_path):
    # scope="session": o modelo é carregado UMA vez para todos os testes. Subir o serviço duas
    # vezes no mesmo processo duplicaria as métricas do Prometheus (DuplicateTimeseries).
    mp = pytest.MonkeyPatch()
    mp.setenv("INFERENCE_LOG_PATH", str(log_path))  # lido pelo __init__ do serviço
    from service import ImageClassifier

    # to_asgi(): o serviço roda em memória, sem servidor e sem porta
    with TestClient(ImageClassifier.to_asgi()) as c:
        yield c
    mp.undo()


@pytest.fixture
def restore_logger(log_path):
    # Testes de rotação reconfiguram o logger; ao final, ele volta a apontar para o log dos testes
    yield
    from monitoring.logging_setup import setup_inference_logger

    setup_inference_logger(log_path)
