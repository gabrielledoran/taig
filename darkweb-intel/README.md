# TAIG — Threat Actor Intelligence Graph

NLP pipeline for extracting structured intelligence from threat actor communications using publicly available or leaked datasets.



## Research question

Can semantic features cluster dark web forum personas into threat actor groups actionable for cyber intel? What operational patterns emerge from their communications?

## Data sources

| Dataset | Contents | Access |
|---------|----------|--------|
| Conti ransomware leaks (2022) | ~60,000 internal chat messages, Russian/English | Archived publicly; see `notebooks/01_data_ingestion.ipynb` |
| Babuk ransomware leaks (2021) | Source code + internal communications | Publicly archived |
| Black Basta chat leaks (2024) | ~200,000 Matrix protocol chat messages, Russian/English, Sept 2022 – Sept 2024 | Released via threat intelligence community; search "Black Basta Matrix leak 2024" |
| LockBit leak corpus (2024) | Affiliate panel communications and operator messages following Operation Cronos | Partial release post-Operation Cronos (Feb 2024) and subsequent community dump (May 2024) |
| Hack Forums academic dataset | Forum posts, threat actor personas | Contact authors (Portnoff et al. 2017 — IEEE S&P) |
| Exploit.db offensive security DB | Public exploit posts with author metadata | exploit-db.com/gitlab |

## Methods

1. **NER** — extract entities: crypto addresses, malware tool names, victim org names, geographies (fine-tuned on CyNER or SecureBERT)
2. **Stylometric clustering** — Sentence-BERT embeddings → UMAP → HDBSCAN for persona grouping
3. **Topic modeling** — BERTopic for operational vocabulary discovery
4. **Geopolitical lens** — language detection, code-switching patterns (Russian/English), time-zone signal from message timestamps

## Notebooks

| Notebook | Purpose |
|----------|---------|
| `01_data_ingestion.ipynb` | Load and clean Conti/Babuk leak data |
| `02_eda_topics.ipynb` | BERTopic modeling, language distribution, timeline analysis |
| `03_clustering.ipynb` | Sentence-BERT embeddings, UMAP, HDBSCAN persona clustering |
| `04_ner.ipynb` | Named entity extraction for crypto, tools, victims |

## Setup

```bash
conda create -n darkwebintel python=3.11
conda activate darkwebintel
pip install -r requirements.txt
```

## Notes on data ethics

All data used here is either:
- Publicly leaked (Conti, Babuk, Black Basta, LockBit) and widely analyzed in academic and threat intelligence literature
- Publicly available (Exploit-DB)
- Available through academic data-sharing agreements (Hack Forums dataset)

No scraping of live dark web infrastructure is involved.
