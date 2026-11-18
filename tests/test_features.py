"""Testes de sanidade das características: casos cujo resultado já sabemos."""

import io
import math
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageEnhance, ImageFilter

from monitoring.features import image_metadata, pixel_features, prediction_features

PRAIA = Path(__file__).parent.parent / "samples" / "06_praia.jpg"


@pytest.fixture
def praia() -> Image.Image:
    with Image.open(PRAIA) as img:
        return img.convert("RGB")


# ------------------------------- Metadados -------------------------------
def test_metadados_de_arquivo_jpeg():
    with Image.open(PRAIA) as img:
        assert image_metadata(img) == {"img_format": "JPEG", "img_mode": "RGB", "exif_rotated": False}


def test_rotacao_exif_detectada():
    # Cria em memória um JPEG com a tag de orientação 6 (= girar 90°), como uma foto de celular
    exif = Image.Exif()
    exif[0x0112] = 6
    buf = io.BytesIO()
    Image.new("RGB", (40, 20)).save(buf, "JPEG", exif=exif)
    buf.seek(0)
    with Image.open(buf) as img:
        assert image_metadata(img)["exif_rotated"] is True


# ---------------------------- Características de pixel ----------------------------
def test_valores_proximos_da_atividade_1(praia):
    # Atividade 1 (imagem inteira): mean_L = 115,43; nitidez = 2312,0; colorido = 46,2
    f = pixel_features(praia)
    assert f["mean_L"] == pytest.approx(115.43, abs=0.5)
    assert f["sharpness"] == pytest.approx(2312.0, rel=0.05)
    assert f["colorfulness"] == pytest.approx(46.2, rel=0.05)


def test_imagem_cinza_lisa_zera_contraste_nitidez_e_colorido():
    f = pixel_features(Image.new("RGB", (300, 200), (128, 128, 128)))
    assert f["std_L"] == 0 and f["sharpness"] == 0 and f["colorfulness"] == 0


def test_desfoque_derruba_a_nitidez(praia):
    borrada = praia.filter(ImageFilter.GaussianBlur(radius=3))
    assert pixel_features(borrada)["sharpness"] < 0.1 * pixel_features(praia)["sharpness"]


def test_escurecer_reduz_o_brilho(praia):
    escura = ImageEnhance.Brightness(praia).enhance(0.5)
    assert pixel_features(escura)["mean_L"] < 0.6 * pixel_features(praia)["mean_L"]


def test_imagem_original_nao_e_alterada(praia):
    tamanho_antes = praia.size
    pixel_features(praia)
    assert praia.size == tamanho_antes  # a redução é feita numa cópia


def test_geometria_e_da_imagem_original():
    f = pixel_features(Image.new("RGB", (1600, 900)))
    assert (f["width"], f["height"], f["aspect_ratio"]) == (1600, 900, 1.7778)


# ---------------------------- Características de saída ----------------------------
def test_modelo_em_duvida_total():
    f = prediction_features(np.full(1000, 1 / 1000))  # mesma probabilidade para as 1000 classes
    assert f["margin_top1_top2"] == 0
    assert f["entropy"] == pytest.approx(math.log(1000), abs=1e-3)  # entropia máxima


def test_modelo_com_certeza_total():
    p = np.zeros(1000)
    p[42] = 1.0
    assert prediction_features(p) == {"top1_score": 1.0, "margin_top1_top2": 1.0, "entropy": 0.0}
