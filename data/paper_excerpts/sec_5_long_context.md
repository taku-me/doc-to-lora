# Doc-to-LoRA §5.1.2: Instant Zero-Shot Internalization of Long-Context Information

Source: arXiv:2602.15902 §5.1.2

We extend our evaluation to long-context scenarios, which pose a significant challenge for standard CD due to memory and computational constraints of distilling long context, using three document-grounded QA benchmarks from LongBench dataset (Bai et al., 2023): 2WikiMultihopQA (Ho et al., 2020), MultiFieldQA, and QASPER (Dasigi et al., 2021). The length of test samples can go up to 32K tokens. We note that D2L has never seen such long sequences during training. Specifically, the longest training sample is 2,344 tokens long (see Figure 5). In this experiment, CD uses generated queries from truncated documents since the base model performs worse without truncation.

The results in Table 1 and Figure 4 show that D2L can effectively internalize long-context documents without being explicitly trained to do so. Like the previous experiment, D2L outperforms CD with generated queries and almost reaches the upper-bound performance of the oracle CD on 2WikiMultihopQA. Even with 5 queries, CD uses up to 79 GB of VRAM to internalize the documents. Furthermore, the oracle CD requires more than 7 GB of VRAM during the update. In contrast, D2L with iterative LoRA generation uses 2x less memory compared to the oracle's update while maintaining sub-second internalization.

## Table 1: Performance on 2WikiMultihopQA

| Method | Normalized Performance ↑ | Additional Update Memory (GB) ↓ | Mean Update Latency (s) ↓ |
|---|---:|---:|---:|
| CD (oracle query) | 0.901 | 7.820 | 40.171 ± 0.351 |
| **D2L (batched)** | **0.857** | 11.522 | **0.209 ± 0.123** |
| **D2L (iterative)** | 0.844 | **3.791** | 0.551 ± 0.101 |
| CD (25 generated queries) | 0.745 | 59.925 | 465.454 ± 67.868 |
| CD (5 generated queries) | 0.704 | 79.371 | 72.537 ± 7.821 |

## Figure 4: Long Document QA Performance

D2L on 2WikiMultihopQA, MultiFieldQA, QASPER — measured against "Additional Memory Needed for Generation (MB)" axis. D2L achieves comparable performance to oracle CD at <100 MB additional memory, while ICL baselines need ~1 GB and CD with generated queries needs >40 GB.

Specifically, the ICL baseline requires around 1 GB of VRAM while all in-parameter knowledge methods require less than 100 MB.

## Key Numerical Facts (Memorize)

- Training sample distribution: **mean 277.3 tok, median 274 tok, max 2,344 tok** (Figure 5, App.B)
- D2L internalization latency: **~0.2-0.5 s** (batched/iterative)
- Oracle CD latency: **~40 s**
- D2L update memory: **3.8 GB (iterative)** / 11.5 GB (batched)
- Generation memory after D2L: **<100 MB additional**
- ICL baseline generation memory: **~1 GB**
