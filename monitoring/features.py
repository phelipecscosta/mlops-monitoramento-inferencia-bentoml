"""Características derivadas para o monitoramento (entrada e saída do modelo).

Só números e rótulos: a imagem nunca sai daqui. As funções de pixel vêm da
Atividade 1 (https://github.com/phelipecscosta/mlops-bloco2-atividade1-processamento-imagens),
com dois ajustes: tudo é calculado sobre uma cópia reduzida (MAX_SIDE) e foi
acrescentado o contraste (std_L).
"""

import numpy as np
from PIL import Image

MAX_SIDE = 512             # px no maior lado da cópia usada nos cálculos (só reduz, nunca amplia)
EXIF_ORIENTATION = 0x0112  # tag EXIF que guarda a rotação da foto (1 = normal)


def image_metadata(image: Image.Image) -> dict:
    """Metadados da imagem COMO CHEGOU (antes da rotação EXIF): formato, modo e rotação."""
    orientation = image.getexif().get(EXIF_ORIENTATION, 1)
    return {
        "img_format": image.format or "unknown",  # JPEG, PNG, ... (None se a imagem não veio de arquivo)
        "img_mode": image.mode,                   # RGB, L (cinza), RGBA, P (paleta) ...
        "exif_rotated": orientation not in (0, 1),
    }


def pixel_features(rgb: Image.Image) -> dict:
    """Características de pixel da imagem JÁ EM PÉ e em RGB (a mesma que o modelo recebe)."""
    width, height = rgb.size

    small = rgb.copy()  # copy(): thumbnail() altera a imagem no lugar; a original segue para o modelo
    small.thumbnail((MAX_SIDE, MAX_SIDE), Image.Resampling.LANCZOS)

    pixels = np.asarray(small, dtype=np.float64)             # formato (altura, largura, 3)
    luma = np.asarray(small.convert("L"), dtype=np.float64)  # BT.601, igual ao repositório de referência
    R, G, B = pixels[:, :, 0], pixels[:, :, 1], pixels[:, :, 2]

    # Nitidez: variância do Laplaciano 3x3 [[0,1,0],[1,-4,1],[0,1,0]] (via deslocamentos do array)
    lap = luma[:-2, 1:-1] + luma[2:, 1:-1] + luma[1:-1, :-2] + luma[1:-1, 2:] - 4 * luma[1:-1, 1:-1]

    # Colorido de Hasler e Süsstrunk (2003): eixos de oposição de cor
    rg = R - G
    yb = 0.5 * (R + G) - B
    colorfulness = np.hypot(rg.std(), yb.std()) + 0.3 * np.hypot(rg.mean(), yb.mean())

    return {
        "width": width,
        "height": height,
        "aspect_ratio": round(width / height, 4),
        "mean_L": round(float(luma.mean()), 4),
        "std_L": round(float(luma.std()), 4),  # contraste
        "mean_R": round(float(R.mean()), 4),
        "mean_G": round(float(G.mean()), 4),
        "mean_B": round(float(B.mean()), 4),
        "sharpness": round(float(lap.var()), 4),
        "colorfulness": round(float(colorfulness), 4),
    }


def prediction_features(probs: np.ndarray) -> dict:
    """Confiança, margem e incerteza a partir do vetor softmax COMPLETO (1000 classes)."""
    p = np.asarray(probs, dtype=np.float64)
    top2 = np.sort(p)[-2:]   # as duas maiores probabilidades, em ordem crescente
    nonzero = p[p > 0]       # 0 * log(0) = 0 por definição; filtrar evita log(0)
    entropy = max(0.0, float(-(nonzero * np.log(nonzero)).sum()))  # em nats; máx. = ln(1000) ≈ 6,91
    return {
        "top1_score": round(float(top2[1]), 4),
        "margin_top1_top2": round(float(top2[1] - top2[0]), 4),
        "entropy": round(entropy, 4),
    }
