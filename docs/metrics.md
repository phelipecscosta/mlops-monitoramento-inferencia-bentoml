# Como analisar o GET /metrics

O serviço expõe em `GET /metrics` (porta 3000) as métricas nativas do BentoML no
[formato de exposição do Prometheus](https://prometheus.io/docs/instrumenting/exposition_formats/).
Este texto explica como ler essas métricas e calcular, a partir delas, a **taxa de erros** e o
**histograma** e o **boxplot da latência**, e por quanto tempo esses dados ficam no servidor.
Os cálculos estão em [`metrics_analise.xlsx`](metrics_analise.xlsx), com os números reais de
[`evidencias/metrics_apos_carga.txt`](evidencias/metrics_apos_carga.txt) (servidor recém-iniciado + `just load`).

## 1. Como ler o formato

Cada linha é `nome{rótulos} valor`. Linhas `# HELP` descrevem a métrica e `# TYPE` dizem o tipo.
Os rótulos separam a mesma métrica por endpoint e código HTTP:

```text
bentoml_service_request_total{endpoint="/classify",http_response_code="200",...} 873.0
```

Para o endpoint de inferência interessam três métricas:

| Métrica | Tipo | O que é |
|---|---|---|
| `bentoml_service_request_total` | counter | Pedidos atendidos, por código HTTP. Só sobe |
| `bentoml_service_request_duration_seconds` | histogram | Duração dos pedidos, em buckets cumulativos, por código HTTP |
| `bentoml_service_request_in_progress` | gauge | Pedidos em andamento agora. Sobe e desce |

## 2. Taxa de erros (aba "Contadores")

Com os contadores por código HTTP:

| Código | Pedidos |
|---|---|
| 200 | 873 |
| 400 | 27 |
| 500 | 0 |
| **Total** | **900** |

- **Taxa de erro total** = (400 + 500) / total = 27 / 900 = **3,0%** (são os pedidos inválidos de propósito do gerador de carga);
- **Taxa de erro do servidor** = 500 / total = **0,0%**. É o indicador de saúde do serviço: 4xx é erro do cliente, 5xx é do servidor.

Os contadores são **acumulados desde que o servidor subiu**. Para a taxa **de um período**, são
necessárias duas leituras: taxa = Δ erros / Δ total e vazão = Δ total / Δ tempo. É o que a função
`rate()` do Prometheus faz. Exemplo na planilha: 900 pedidos em 111,8 s = **8,05 pedidos/s**.

## 3. Histograma da latência (aba "Histograma")

O histograma do Prometheus não guarda cada duração: guarda **contadores cumulativos por faixa**.
`_bucket{le="0.25"} 850` significa que 850 pedidos levaram **até** 0,25 s. Para obter o histograma
comum, subtrai-se cada bucket do anterior:

| Faixa (s) | Acumulado (`le`) | Pedidos na faixa |
|---|---|---|
| até 0,05 | 2 | 2 |
| (0,05 ; 0,1] | 20 | 18 |
| (0,1 ; 0,25] | 850 | **830 (95%)** |
| (0,25 ; 0,5] | 873 | 23 |
| acima de 0,5 | 873 | 0 |

A **média exata** sai de `_sum / _count` = 155,97 s / 873 = **0,179 s**. É a única estatística
exata; todas as outras são estimadas a partir das faixas.

## 4. Boxplot da latência (aba "Boxplot")

Os quartis são **estimados** pelo mesmo método do `histogram_quantile` do Prometheus:

1. posição = q × `_count` (ex.: mediana = 0,5 × 873 = 436,5);
2. acha-se o primeiro bucket cujo acumulado ≥ posição (aqui, (0,1 ; 0,25], com acumulado 850);
3. interpola-se linearmente dentro dele, supondo os pedidos distribuídos por igual:
   0,1 + (0,25 − 0,1) × (436,5 − 20) / (850 − 20) = **0,175 s**.

| P1 | Q1 | Mediana | Q3 | P95 | P99 |
|---|---|---|---|---|---|
| 0,069 s | 0,136 s | 0,175 s | 0,215 s | 0,246 s | 0,405 s |

**Limitação:** os buckets padrão do BentoML vão de 5 ms a 180 s. Para um serviço que responde em
cerca de 0,2 s eles são grossos: 95% dos pedidos caem numa única faixa, e Q1, mediana, Q3 e P95
são interpolações dentro dela. O mínimo e o máximo não podem ser conhecidos, por isso os bigodes
usam P1 e P99. Para quantis exatos, usa-se o `latency_ms` do log em arquivo, que guarda a
duração de cada pedido.

## 5. Por quanto tempo os dados ficam no servidor (aba "Retenção")

- O BentoML guarda as métricas em arquivos do modo multiprocesso do `prometheus_client`
  (pasta `prometheus_multiproc_dir`, dentro do `BENTOML_HOME`).
- **Os valores valem desde que o servidor subiu e são apagados a cada reinício**: na subida, o
  BentoML limpa essa pasta (`ensure_prometheus_dir`, com `clean=True`).
- **Não há janela de tempo**: o `/metrics` é uma foto do acumulado, sem histórico nem horário
  de cada pedido.

**Evidência:** depois de reiniciar o servidor, o contador do `/classify` voltou a 0
([`evidencias/metrics_apos_reinicio.txt`](evidencias/metrics_apos_reinicio.txt)), enquanto o
log em arquivo continuou com as 1820 linhas.

Para ter histórico, um servidor Prometheus coleta o `/metrics` periodicamente e guarda as séries
com data e hora (retenção padrão de 15 dias). É o extra E2 deste projeto.

## 6. /metrics × log em arquivo

| | `/metrics` | Log em arquivo (`logs/inference.jsonl`) |
|---|---|---|
| Granularidade | Contadores agregados por faixa | Um registro por pedido |
| Latência | Média exata; quantis estimados | Valor exato de cada pedido |
| Variáveis de dados (entrada e saída) | Não tem | Tem (drift, KS, PSI) |
| Sobrevive a reinício | Não | Sim (append-only + rotação) |
| Custo e padronização | Baixíssimo; lido por qualquer ferramenta Prometheus | Exige leitura e análise próprias |

Os dois se complementam: o `/metrics` responde rápido a "o serviço está saudável agora?"
(alertas de taxa de erro e latência), e o log responde a "o que entrou, o que saiu e os dados
mudaram?".
