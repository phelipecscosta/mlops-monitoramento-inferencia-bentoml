"""Gera carga para o serviço: variações das imagens de samples/, em memória e com semente fixa.

Cenários (enviados no cabeçalho X-Scenario e gravados no log):
- reference: conjunto de referência (variações "normais" das 8 imagens);
- normal:    mesma distribuição da referência, com outra semente (não deve acusar drift);
- drift:     imagens escurecidas, borradas, dessaturadas e mais comprimidas (deve acusar drift).

Uso:
    uv run python scripts/generate_load.py                      # os 3 cenários, 300 pedidos cada
    uv run python scripts/generate_load.py --scenario drift --n 100
"""

import argparse
import io
import random
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
from PIL import Image, ImageEnhance, ImageFilter, ImageOps

SAMPLES = Path(__file__).resolve().parent.parent / "samples"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
SEEDS = {"reference": 101, "normal": 202, "drift": 303}  # semente fixa por cenário: carga reproduzível
INVALID_IMAGE_RATE = 0.02  # 2% dos pedidos levam um arquivo que não é imagem (gera 400)
INVALID_TOP_K_RATE = 0.01  # 1% dos pedidos levam top_k fora da faixa (gera 400)


def normal_variation(img: Image.Image, rng: random.Random) -> Image.Image:
    """Variações leves, como fotos diferentes da mesma fonte de imagens."""
    w, h = img.size
    scale = rng.uniform(0.7, 1.0)  # recorte de 70% a 100% de cada lado
    cw, ch = int(w * scale), int(h * scale)
    x, y = rng.randint(0, w - cw), rng.randint(0, h - ch)
    img = img.crop((x, y, x + cw, y + ch))
    if rng.random() < 0.5:
        img = ImageOps.mirror(img)  # espelhamento horizontal
    img = img.rotate(rng.uniform(-10, 10), resample=Image.Resampling.BILINEAR, expand=False)
    img = ImageEnhance.Brightness(img).enhance(rng.uniform(0.85, 1.15))
    img = ImageEnhance.Contrast(img).enhance(rng.uniform(0.85, 1.15))
    img = ImageEnhance.Color(img).enhance(rng.uniform(0.85, 1.15))  # saturação
    max_side = rng.randint(320, 800)  # resoluções variadas
    img.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    return img


def drift_variation(img: Image.Image, rng: random.Random) -> Image.Image:
    """Simula uma fonte de imagens pior: câmera escura, fora de foco e com cores desbotadas."""
    img = normal_variation(img, rng)
    img = ImageEnhance.Brightness(img).enhance(rng.uniform(0.35, 0.6))
    img = img.filter(ImageFilter.GaussianBlur(radius=rng.uniform(1.5, 3.0)))
    img = ImageEnhance.Color(img).enhance(rng.uniform(0.2, 0.5))
    return img


def build_requests(scenario: str, n: int) -> list[dict]:
    """Monta TODOS os pedidos antes de enviar: o conteúdo não depende da ordem de envio."""
    rng = random.Random(SEEDS[scenario])
    paths = sorted(p for p in SAMPLES.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
    originals = [Image.open(p).convert("RGB") for p in paths]
    vary = drift_variation if scenario == "drift" else normal_variation
    quality = (40, 70) if scenario == "drift" else (75, 95)  # drift: compressão JPEG mais forte

    requests = []
    for i in range(n):
        r = rng.random()
        if r < INVALID_IMAGE_RATE:
            content, top_k = b"isto nao e uma imagem", 5
        else:
            buf = io.BytesIO()
            vary(rng.choice(originals), rng).save(buf, "JPEG", quality=rng.randint(*quality))
            content = buf.getvalue()
            top_k = 50 if r < INVALID_IMAGE_RATE + INVALID_TOP_K_RATE else 5
        # Nome genérico: o nome do arquivo não é gravado no log, mas o multipart exige um
        requests.append({"content": content, "top_k": top_k, "name": f"img_{i:04d}.jpg"})
    return requests


def send(client: httpx.Client, url: str, scenario: str, req: dict) -> int:
    resp = client.post(
        url,
        files={"image": (req["name"], req["content"], "image/jpeg")},
        data={"top_k": str(req["top_k"])},
        headers={"X-Scenario": scenario},
    )
    return resp.status_code


def run(base_url: str, scenario: str, n: int, workers: int) -> Counter:
    print(f"[{scenario}] gerando {n} pedidos (semente {SEEDS[scenario]})...", flush=True)
    requests = build_requests(scenario, n)
    url = f"{base_url}/classify"
    t0 = time.perf_counter()
    statuses = Counter()
    with httpx.Client(timeout=60) as client, ThreadPoolExecutor(max_workers=workers) as pool:
        for i, status in enumerate(pool.map(lambda r: send(client, url, scenario, r), requests), 1):
            statuses[status] += 1
            if i % 50 == 0 or i == n:
                print(f"[{scenario}] {i}/{n} enviados", flush=True)
    print(f"[{scenario}] concluído em {time.perf_counter() - t0:.1f} s; respostas: {dict(sorted(statuses.items()))}")
    return statuses


def main() -> None:
    parser = argparse.ArgumentParser(description="Gera carga para o serviço de classificação.")
    parser.add_argument("--scenario", choices=["all", *SEEDS], default="all")
    parser.add_argument("--n", type=int, default=300, help="pedidos por cenário (padrão: 300)")
    parser.add_argument("--workers", type=int, default=2, help="pedidos simultâneos (padrão: 2)")
    parser.add_argument("--url", default="http://localhost:3000", help="endereço do serviço")
    args = parser.parse_args()

    try:  # falha cedo, com mensagem clara, se o serviço não estiver no ar
        httpx.get(f"{args.url}/readyz", timeout=5).raise_for_status()
    except httpx.HTTPError:
        sys.exit(f"Serviço indisponível em {args.url}. Suba-o antes, em outro terminal: just serve")

    scenarios = list(SEEDS) if args.scenario == "all" else [args.scenario]
    for scenario in scenarios:
        run(args.url, scenario, args.n, args.workers)
    print("Log em: logs/inference.jsonl")


if __name__ == "__main__":
    main()
    