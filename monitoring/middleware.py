"""Middleware ASGI: grava UMA linha de log por pedido ao endpoint de inferência, com sucesso ou erro."""

import contextvars
import logging
import time
import uuid

from monitoring.logging_setup import LOGGER_NAME

SCHEMA_VERSION = "1"
SCENARIOS = {"reference", "normal", "drift"}  # lista fechada: qualquer outro valor vira "unspecified"

# Todos os campos do evento, na ordem do docs/crisp_dm.md. Campos sem valor ficam null:
# toda linha tem as mesmas chaves, o que simplifica a leitura com pandas.
FIELDS = (
    "event", "schema_version", "request_id", "endpoint", "scenario", "model_id", "model_revision",
    "top_k", "file_bytes", "img_format", "img_mode", "width", "height", "aspect_ratio", "exif_rotated",
    "mean_L", "std_L", "mean_R", "mean_G", "mean_B", "sharpness", "colorfulness",
    "status", "http_status", "error_type", "top1_label", "top1_score", "margin_top1_top2", "entropy",
    "features_ms", "inference_ms", "latency_ms",
)

# "Prancheta" do pedido atual: o middleware cria, o validador e o endpoint preenchem
_current: contextvars.ContextVar[dict | None] = contextvars.ContextVar("inference_event", default=None)


def current_event() -> dict:
    """Devolve o evento do pedido em andamento (ou um dict descartável, se não houver pedido)."""
    event = _current.get()
    return event if event is not None else {}


class InferenceLogMiddleware:
    def __init__(self, app, paths: tuple[str, ...] = ("/classify",), model_id=None, model_revision=None):
        self.app = app
        self.paths = set(paths)
        self.model_id = model_id
        self.model_revision = model_revision

    async def __call__(self, scope, receive, send):
        # Só monitora POST nos endpoints de inferência; /metrics, /docs etc. passam direto
        if scope["type"] != "http" or scope["method"] != "POST" or scope["path"] not in self.paths:
            await self.app(scope, receive, send)
            return

        t0 = time.perf_counter()
        headers = {k.decode("latin-1").lower(): v.decode("latin-1") for k, v in scope["headers"]}
        scenario = headers.get("x-scenario", "").strip().lower()
        event = dict.fromkeys(FIELDS)  # todas as chaves, valores None
        event.update(
            event="inference",
            schema_version=SCHEMA_VERSION,
            request_id=uuid.uuid4().hex,  # aleatório: identifica o pedido, não a pessoa
            endpoint=scope["path"],
            scenario=scenario if scenario in SCENARIOS else "unspecified",
            model_id=self.model_id,
            model_revision=self.model_revision,
        )
        response = {}

        async def send_wrapper(message):
            if message["type"] == "http.response.start":
                response["status"] = message["status"]
                # Devolve o request_id ao cliente: permite achar o pedido no log (auditoria)
                message["headers"] = [*message.get("headers", []), (b"x-request-id", event["request_id"].encode())]
            await send(message)

        token = _current.set(event)
        try:
            await self.app(scope, receive, send_wrapper)
        finally:
            _current.reset(token)
            self._finish(event, response.get("status"), t0)

    @staticmethod
    def _finish(event: dict, http_status: int | None, t0: float) -> None:
        event["http_status"] = http_status
        event["status"] = "ok" if http_status is not None and http_status < 400 else "error"
        if event["status"] == "error" and event["error_type"] is None:
            if http_status is None:
                event["error_type"] = "no_response"  # ex.: cliente desconectou
            elif http_status < 500:
                event["error_type"] = "validation_error"  # 4xx sem causa mais específica (ex.: top_k inválido)
            else:
                event["error_type"] = "internal_error"
        event["latency_ms"] = round((time.perf_counter() - t0) * 1000, 3)
        logging.getLogger(LOGGER_NAME).info("inference", extra={"event": event})
