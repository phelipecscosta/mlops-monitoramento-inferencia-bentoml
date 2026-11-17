# Análise CRISP-DM — monitoramento do serviço de classificação de imagens

Objetivo desta análise: justificar **quais variáveis** o serviço grava no log e **em que formato**.
As fases 4 a 6 do CRISP-DM são adaptadas: o modelo (ResNet-50) já está pronto e não é treinado aqui;
o "modelo" deste projeto é a análise de drift que será feita sobre o log.

## Fases do CRISP-DM

O CRISP-DM (*Cross-Industry Standard Process for Data Mining*) é o processo padrão da indústria
para projetos de dados. Tem 6 fases, que são cíclicas: os resultados de uma fase podem levar a
rever as anteriores.

| Fase | Pergunta que responde | Neste projeto |
|---|---|---|
| 1. Entendimento do negócio | Qual problema resolver e como medir o sucesso? | Que perguntas o monitoramento deve responder |
| 2. Entendimento dos dados | Que dados existem e como são? | O que o endpoint recebe e devolve |
| 3. Preparação dos dados | Que dados usar e em que forma? | Variáveis gravadas no log e formato do arquivo |
| 4. Modelagem | Que técnica aplicar aos dados? | Análises previstas: histograma, boxplot, KS e PSI (o ResNet-50 não é treinado aqui) |
| 5. Avaliação | O resultado atende ao objetivo? | Os dados coletados bastam para as análises? |
| 6. Implantação | Como colocar em uso? | Log em produção: rotação, retenção e leitura |

## 1. Entendimento do negócio

**Problema.** Um serviço de inferência pode piorar sem gerar erro: se as imagens recebidas mudarem
(câmera diferente, iluminação, fotos borradas, outro tipo de conteúdo), o modelo continua respondendo
`200 OK`, mas com predições menos confiáveis. Sem registro do que entra e do que sai, essa degradação
é invisível e um incidente não pode ser investigado.

**Perguntas que o monitoramento deve responder:**

| # | Pergunta | Tipo |
|---|---|---|
| P1 | As imagens que chegam se parecem com as do conjunto de referência? | Drift de entrada |
| P2 | As respostas do modelo (classe e confiança) mudaram de padrão? | Drift de saída |
| P3 | O serviço está lento ou falhando? | Saúde operacional |
| P4 | Um pedido específico deu problema: o que entrou e o que saiu? | Auditoria |

**Critério de sucesso.** Com o arquivo de log, e só com ele, deve ser possível montar histogramas
e boxplots das variáveis de entrada e saída e calcular KS e PSI contra um conjunto de referência.

**Restrições.**
- Privacidade (LGPD): não gravar a imagem, o nome do arquivo nem o IP do cliente; apenas
  características numéricas/categóricas derivadas e o resultado da inferência.
- Reprodutibilidade: o professor sobe o serviço e gera carga só com o README, em qualquer máquina (CPU).
- Windows: gravação e rotação do arquivo devem funcionar de forma confiável.
- Custo: extrair as características não pode aumentar de forma relevante a latência da inferência.

## 2. Entendimento dos dados

**O que o endpoint `POST /classify` recebe:**
- `image`: arquivo de imagem (JPEG, PNG, WEBP ou BMP), enviado como multipart;
- `top_k`: inteiro de 1 a 10 (padrão 5).

**O que devolve:**
- `200`: identificação do modelo (`id`, `revision`), `inference_ms` e a lista das `top_k` classes com `score` (softmax);
- `400`: imagem inválida/corrompida ou `top_k` fora da faixa;
- `500`: erro interno; a requisição é abortada após 30 s (timeout).

**Observações sobre as 8 imagens de `samples/`:**
- tamanhos variados (640×426 a 800 px no maior lado), formatos JPEG e PNG;
- conteúdos bem diferentes (objetos, paisagens, ruído), o que gera características em faixas amplas;
- `05_ruido_sintetico.png` é um caso extremo (nitidez e colorido muito altos e baixa confiança do modelo):
  bom exemplo do que o monitoramento deve destacar.

**Consequência:** a imagem em si não pode ser gravada (privacidade e volume). Ela precisa ser
**resumida em números** que capturem brilho, cor, contraste, nitidez e geometria, os aspectos que
mudam quando a fonte das imagens muda.

## 3. Preparação dos dados: variáveis gravadas no log

Granularidade: **uma linha por pedido**, sucesso ou erro. KS e PSI exigem os valores individuais,
não médias agregadas.

### 3.1 Identificação do evento

| Variável | Tipo | Pergunta | Justificativa |
|---|---|---|---|
| ts | Data/hora ISO 8601 (µs + fuso) | P3, P4 | Ordenar eventos, filtrar janelas de tempo, medir taxa por período |
| event | Categórica | P4 | Tipo do evento (`inference`); permite outros tipos no futuro |
| schema_version | Categórica | P4 | Permite mudar o formato sem quebrar a leitura de logs antigos |
| request_id | Texto (UUID4 aleatório) | P4 | Identifica o pedido sem identificar a pessoa |
| scenario | Categórica | P1, P2 | Separa a carga de referência (`reference`) da carga avaliada (`normal`, `drift`); vem do cabeçalho `X-Scenario`, com lista fechada de valores |
| model_id, model_revision | Categórica | P2, P4 | Uma mudança de saída pode ser troca de modelo, não drift dos dados |

### 3.2 Entrada (características derivadas da imagem)

| Variável | Tipo | Pergunta | Justificativa |
|---|---|---|---|
| top_k | Numérica (inteiro) | P4 | Parâmetro do pedido; explica o tamanho da resposta |
| file_bytes | Numérica | P1, P3 | Tamanho do arquivo: muda com a câmera e a compressão; afeta a latência |
| img_format | Categórica | P1 | JPEG/PNG/…: mudança de formato indica nova fonte de imagens |
| img_mode | Categórica | P1 | RGB, L (cinza), RGBA: imagens em tons de cinza são um drift comum |
| width, height | Numérica | P1, P3 | Resolução de origem; afeta a latência e a qualidade |
| aspect_ratio | Numérica | P1 | Largura/altura: retrato × paisagem × quadrado |
| exif_rotated | Categórica (bool) | P1 | Fotos de celular com rotação no EXIF indicam a origem da imagem |
| mean_L | Numérica (0 a 255) | P1 | Brilho médio: detecta mudança de iluminação, câmera escura |
| std_L | Numérica (≥ 0) | P1 | Contraste: imagens "lavadas" ou com névoa têm desvio baixo |
| mean_R, mean_G, mean_B | Numérica (0 a 255) | P1 | Balanço de cor: detecta mudança de balanço de branco e de conteúdo |
| sharpness | Numérica (≥ 0) | P1 | Nitidez (variância do Laplaciano): detecta foto borrada ou comprimida demais |
| colorfulness | Numérica (≥ 0) | P1 | Colorido (Hasler e Süsstrunk): detecta imagem desbotada ou em preto e branco |

As características são calculadas sobre uma cópia reduzida a 512 px no maior lado (só reduz, nunca
amplia). Isso limita o custo para fotos grandes e torna os valores comparáveis entre resoluções. O que
importa para KS e PSI é usar **o mesmo cálculo** na referência e na carga avaliada.

### 3.3 Saída (resultado da inferência e do serviço)

| Variável | Tipo | Pergunta | Justificativa |
|---|---|---|---|
| status | Categórica (`ok`/`error`) | P3 | Base da taxa de erro |
| http_status | Categórica | P3 | Distingue erro do cliente (4xx) de erro do servidor (5xx) |
| error_type | Categórica | P3, P4 | Classe do erro (ex.: imagem inválida), sem a mensagem completa |
| top1_label | Categórica | P2 | Classe mais provável: mudança de frequência indica mudança de conteúdo |
| top1_score | Numérica (0 a 1) | P2 | Confiança do modelo: tende a cair com drift |
| margin_top1_top2 | Numérica (0 a 1) | P2 | Modelo dividido entre duas classes |
| entropy | Numérica (≥ 0) | P2 | Incerteza sobre as 1000 classes: sobe antes de a acurácia cair |
| features_ms | Numérica | P3 | Custo da extração de características (verifica a restrição de custo) |
| inference_ms | Numérica | P3 | Tempo de pré-processamento + inferência do modelo |
| latency_ms | Numérica | P3 | Tempo total do pedido no servidor |

### 3.4 O que NÃO é gravado, e por quê

| Dado | Motivo |
|---|---|
| Imagem (bytes ou pixels) | Privacidade (pode conter pessoas) e volume |
| Nome original do arquivo | Pode conter nome de pessoa ou outra informação pessoal |
| IP e User-Agent do cliente | Dados pessoais pela LGPD; não respondem a nenhuma pergunta P1 a P4 |
| Lista completa das top_k classes | A classe top-1, a margem e a entropia bastam para as distribuições; reduz o tamanho do log |
| Mensagem completa do erro e traceback | Pode conter dados do pedido; o `error_type` basta para a taxa de erro |

## 4. Modelagem: análises previstas sobre o log

Esta entrega só **grava** os dados; as análises abaixo definem o que o log precisa conter.

| Análise | Variáveis | Uso |
|---|---|---|
| Histograma e boxplot | Todas as numéricas de entrada e saída | Forma da distribuição, assimetria e outliers |
| KS | Numéricas contínuas (mean_*, std_L, sharpness, colorfulness, aspect_ratio, top1_score, margin, entropy, latências) | Distância entre a carga avaliada e a referência, com p-valor |
| PSI | Numéricas (em faixas definidas pela referência) e categóricas (img_format, img_mode, top1_label agrupada) | Estabilidade da distribuição, com limiares da indústria (0,1 e 0,25) |
| Taxa de erro | status, http_status, error_type | Saúde operacional por período |

**Conjunto de referência:** os pedidos com `scenario = reference`, gerados por script a partir das 8
imagens de `samples/`, com variações (recorte, brilho, desfoque, rotação) e semente fixa.

## 5. Avaliação: critérios de suficiência dos dados

Verificados num notebook rápido após gerar carga (etapa T5):

1. **Volume:** pelo menos 200 pedidos por cenário. O PSI com 10 faixas precisa de cerca de 20 observações
   por faixa para não ficar instável; o KS funciona com menos, mas perde poder de detecção.
2. **Completude:** nenhum campo nulo nas linhas de sucesso; nas de erro, os campos de saída ficam nulos
   de propósito e o `error_type` é preenchido.
3. **Separação:** o cenário `drift` deve produzir KS e PSI visivelmente maiores que o `normal`. Isso
   prova que as variáveis escolhidas detectam a mudança.
4. **Leitura direta:** o arquivo deve ser lido sem tratamento manual (`pandas.read_json(..., lines=True)`).

**Limitações conhecidas:**
- a referência deriva de apenas 8 imagens: tem pouca diversidade de conteúdo e não substitui um
  conjunto de treino real;
- o `scenario` vem do cliente (cabeçalho), o que só se justifica num ambiente de demonstração;
- `top1_label` tem até 1000 categorias: para o PSI, agrupar as mais frequentes e juntar o resto em
  "outras".

## 6. Implantação: formato e ciclo de vida do arquivo

**Formato: JSON Lines** (`logs/inference.jsonl`, UTF-8), um evento por linha.

| Critério | JSON Lines | CSV | Texto livre |
|---|---|---|---|
| Append-only seguro (linha corrompida não afeta as outras) | Sim | Sim | Sim |
| Aceita novos campos sem quebrar arquivos antigos | Sim | Não (cabeçalho fixo) | — |
| Tipos preservados (número, texto, nulo, booleano) | Sim | Não | Não |
| Leitura direta por pandas, Loki e Elasticsearch | Sim | Parcial | Exige regex |

**Data e hora:** ISO 8601 com microssegundos e fuso horário (ex.: `2026-11-17T17:07:00.123456-03:00`).
O `asctime` padrão do `logging` usa só milissegundos e não inclui o fuso, por isso o timestamp é gerado
por um formatador próprio.

**Rotação e retenção:** rotação por tamanho, com limite e número de arquivos guardados configuráveis.
O mecanismo deve funcionar com segurança no Windows (decisão justificada na etapa T3). Arquivos antigos
nunca são reescritos, só renomeados.

**Exemplo de linha (sucesso):**

```json
{"ts": "2026-11-17T17:07:00.123456-03:00", "event": "inference", "schema_version": "1", "request_id": "3f2c...", "scenario": "normal", "model_id": "microsoft/resnet-50", "model_revision": "34c2154c...", "top_k": 5, "file_bytes": 96772, "img_format": "JPEG", "img_mode": "RGB", "width": 640, "height": 480, "aspect_ratio": 1.3333, "exif_rotated": false, "mean_L": 115.43, "std_L": 61.2, "mean_R": 115.04, "mean_G": 116.68, "mean_B": 110.05, "sharpness": 2312.0, "colorfulness": 46.2, "status": "ok", "http_status": 200, "error_type": null, "top1_label": "seashore, coast, seacoast, sea-coast", "top1_score": 0.8123, "margin_top1_top2": 0.7011, "entropy": 0.92, "features_ms": 6.1, "inference_ms": 41.7, "latency_ms": 49.3}
```

Os valores de `std_L`, `top1_label`, `top1_score`, `margin_top1_top2` e `entropy` acima são
ilustrativos; os reais aparecem quando a carga for gerada.
