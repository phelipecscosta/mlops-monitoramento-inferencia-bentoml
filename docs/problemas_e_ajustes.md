# Registro de problemas e ajustes

Problemas reais encontrados durante o desenvolvimento, como foram diagnosticados
e a decisão tomada. Ordem cronológica.

| # | Problema | Diagnóstico | Ajuste |
|---|---|---|---|
| 1 | `.gitignore` criado pelo PowerShell pode ser ignorado pelo git | PowerShell 5 grava `>`/`Out-File` em UTF-16; o git não lê | Arquivos de texto criados pelo VS Code (UTF-8) |
| 2 | PyTorch do PyPI em Linux traz CUDA (~2,5 GB) | Variantes diferentes por SO no PyPI | Índice `pytorch-cpu` explícito no `pyproject.toml` (`torch==...+cpu` em qualquer SO) |
| 3 | `requires-python = ">=3.11"` aceitava versões não testadas | Lock universal resolve para todas as versões declaradas | Restrito a `>=3.11,<3.12` |
| 4 | Modelo do Hugging Face pode mudar sem aviso | `from_pretrained` baixa sempre a versão mais recente | `revision` fixado no commit `34c2154c...` |
| 5 | Fotos de celular chegam "deitadas" ao modelo | Rotação fica no EXIF; o Pillow não a aplica sozinho | `ImageOps.exif_transpose` antes da inferência |
| 6 | Fotos de celular expõem localização (GPS no EXIF) — dado pessoal (LGPD) | Metadados EXIF incluem coordenadas | `prepare_samples.py` remove todo EXIF; originais em `data/` (ignorada) |
| 7 | `curl` no PowerShell não é o curl | É apelido de `Invoke-WebRequest` | Usar `curl.exe` nas instruções |
| 8 | Arquivo que não é imagem devolvia **500** (erro do servidor) | Traceback: decodificação ocorria dentro do BentoML, antes do método, sem conversão para 4xx | Parâmetro `image: Path` + `load_image()` própria: falha vira `InvalidArgument` → **400** |
| 9 | README prometia "3 a 6 minutos" de instalação | Teste em clone limpo, com caches vazios e sem Python local, mediu até ~10 min | Tempo corrigido no README com base na medição; registradas as duas versões de Python testadas (3.11.4 e 3.11.17) |
| 10 | Sugestão do professor: usar `Image.Image` (mais idiomático); com ele, arquivo inválido voltava a dar **500** | A decodificação do BentoML ocorre **dentro da validação do Pydantic**; o erro do Pillow não é um erro de validação e "vaza" como 500 | `WrapValidator` envolve a decodificação: a falha vira `ValueError` → erro de validação → **400**. Mantém `Image.Image` e unifica o formato de todos os erros de entrada |
| 11 | `RotatingFileHandler` padrão falha no Windows se outro programa mantém o log aberto (ex.: `Get-Content -Wait`, coletor do Loki) | Windows não permite renomear arquivo aberto: `PermissionError` na rotação | `ConcurrentRotatingFileHandler` (pacote `concurrent-log-handler`): trava de arquivo e, se a renomeação falhar, continua gravando sem perder eventos; coberto por teste automatizado |
| 12 | O `bentoml serve` importa o `service.py` em dois processos (supervisor e worker) | Diagnóstico com `os.getpid()`: import em 2 PIDs, `__init__` do serviço só no worker | Log configurado dentro do `__init__`, nunca no topo do módulo: só o processo que atende pedidos abre o arquivo |
| 13 | Erros 400 de validação não chegam ao método `classify`, então não seriam registrados | A validação do Pydantic ocorre antes da chamada do método | Middleware ASGI grava uma linha por pedido, inclusive erros; um `contextvars` leva as características do método até o middleware |
| 14 | Dois `TestClient` do serviço no mesmo processo de testes quebram com `DuplicateTimeseries` | As métricas do Prometheus são globais por processo | Fixture única `client` com `scope="session"` em `tests/conftest.py` |
| 15 | `Remove-Item .venv` apagou a pasta pela metade (`No pyvenv.cfg file`) | Mensagem de erro: `python.exe` e fontes do matplotlib "being used by another process"; o kernel do notebook continuava vivo após fechar o notebook | Encerrar só os processos do `.venv` do projeto: `Get-Process python* \| Where-Object { $_.Path -like "$PWD\.venv\*" } \| Stop-Process -Force`; depois apagar e `uv sync --locked` |

## Limitações observadas (não corrigidas, por decisão)

- **Erros do cliente (4xx) registrados como `ERROR` com traceback** no log do BentoML,
  poluindo o log. Comportamento do framework.


- **Dependências do framework usam APIs depreciadas**: os testes emitem avisos do
  BentoML (APIs do Pydantic a serem removidas no Pydantic 3), do Starlette e do pathspec.
  Não afetam o funcionamento hoje, mas uma instalação sem versões fixadas poderia quebrar
  numa atualização futura — risco mitigado pelo `uv.lock`, que trava as versões testadas.

