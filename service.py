"""Serviço BentoML de classificação de imagens com ResNet-50 (ImageNet-1k)."""

import time
from typing import Annotated, Any

import bentoml
import torch
from PIL import Image, ImageOps, UnidentifiedImageError
from pydantic import BaseModel, Field, ValidatorFunctionWrapHandler, WrapValidator
from transformers import AutoImageProcessor, AutoModelForImageClassification

from monitoring.features import image_metadata, pixel_features, prediction_features
from monitoring.logging_setup import setup_inference_logger
from monitoring.middleware import InferenceLogMiddleware, current_event

MODEL_ID = "microsoft/resnet-50"
MODEL_REVISION = "34c2154c194f829b11125337b98c8f5f9965ff19"  # versão fixada do modelo


# ---------------------- Contrato de saída (aparece no Swagger) ----------------------
class Prediction(BaseModel):
    label: str = Field(description="Classe do ImageNet-1k", examples=["ballpoint, ballpoint pen, ballpen, Biro"])
    score: float = Field(ge=0.0, le=1.0, description="Confiança do softmax (0 a 1); NÃO é probabilidade de acerto", examples=[0.9876])


class ModelInfo(BaseModel):
    id: str = Field(description="Repositório do modelo no Hugging Face", examples=[MODEL_ID])
    revision: str = Field(description="Commit exato do modelo", examples=[MODEL_REVISION])


class ClassificationResponse(BaseModel):
    model: ModelInfo
    inference_ms: float = Field(description="Tempo de pré-processamento + inferência, em ms", examples=[42.7])
    predictions: list[Prediction] = Field(description="Classes ordenadas da mais para a menos provável")


# ------------- Entrada: imagem decodificada pelo BentoML, com falha virando 400 -------------
def decode_or_400(value: Any, handler: ValidatorFunctionWrapHandler) -> Image.Image:
    """Envolve a decodificação do BentoML: arquivo inválido vira erro de VALIDAÇÃO (400), não 500."""
    event = current_event()
    event["file_bytes"] = getattr(value, "size", None)  # tamanho do upload, em bytes (o conteúdo não é guardado)
    try:
        img = handler(value)  # o BentoML abre a imagem aqui (validação original)
        img.load()            # força a leitura completa: detecta arquivo truncado ainda na validação
        return img
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        event["error_type"] = "invalid_image"  # causa específica do 400, para o log
        # ValueError é o sinal que o Pydantic entende como "dado inválido" -> o BentoML responde 400
        raise ValueError(
            "O arquivo enviado não é uma imagem válida ou está corrompido. Envie JPEG, PNG, WEBP ou BMP."
        ) from exc


ValidImage = Annotated[Image.Image, WrapValidator(decode_or_400)]


# ---------------------------------- Serviço ----------------------------------
@bentoml.service(traffic={"timeout": 30})  # requisição que passar de 30 s é abortada
class ImageClassifier:
    def __init__(self) -> None:
        # Roda UMA vez, na subida do servidor, e SÓ no processo que atende os pedidos.
        # Por isso o log é configurado aqui, e não no topo do arquivo: o processo supervisor
        # do BentoML também importa este módulo, e não deve abrir o arquivo de log.
        setup_inference_logger()
        self.processor = AutoImageProcessor.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
        self.model = AutoModelForImageClassification.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
        self.model.eval()

    @bentoml.api
    def classify(
        self,
        image: ValidImage,  # imagem enviada pelo cliente, já decodificada e validada
        top_k: Annotated[int, Field(ge=1, le=10, description="Quantas classes devolver (1 a 10)")] = 5,
    ) -> ClassificationResponse:
        """Recebe uma imagem e devolve as top_k classes mais prováveis do ImageNet-1k."""
        event = current_event()  # "prancheta" do pedido: o middleware grava no log ao final
        event["top_k"] = top_k
        try:
            # --- Entrada: características derivadas (a imagem não é guardada) ---
            t_feat = time.perf_counter()
            event.update(image_metadata(image))  # formato, modo e rotação: ANTES de girar
            # Fotos de celular vêm "deitadas" com a rotação no EXIF: aplica antes de tudo
            pil_image = ImageOps.exif_transpose(image).convert("RGB")
            event.update(pixel_features(pil_image))  # a mesma imagem que o modelo recebe
            event["features_ms"] = round((time.perf_counter() - t_feat) * 1000, 3)

            # --- Inferência (mesmo código da base) ---
            t0 = time.perf_counter()
            inputs = self.processor(images=pil_image, return_tensors="pt")
            with torch.inference_mode():
                probs = self.model(**inputs).logits.softmax(dim=-1)[0]
            top = torch.topk(probs, k=top_k)
            inference_ms = round((time.perf_counter() - t0) * 1000, 1)

            # --- Saída: resultado da inferência ---
            event.update(prediction_features(probs.numpy()))
            event["top1_label"] = self.model.config.id2label[top.indices[0].item()]
            event["inference_ms"] = inference_ms
        except Exception as exc:
            event["error_type"] = type(exc).__name__  # só a classe do erro, sem a mensagem (pode conter dados)
            raise

        return ClassificationResponse(
            model=ModelInfo(id=MODEL_ID, revision=MODEL_REVISION),
            inference_ms=inference_ms,
            predictions=[
                Prediction(label=self.model.config.id2label[idx.item()], score=round(score.item(), 4))
                for score, idx in zip(top.values, top.indices)
            ],
        )


# Grava uma linha de log por pedido ao /classify (inclusive os erros 400 de validação,
# que acontecem ANTES do método classify ser chamado)
ImageClassifier.add_asgi_middleware(
    InferenceLogMiddleware, paths=("/classify",), model_id=MODEL_ID, model_revision=MODEL_REVISION
)
