

![image](https://github.com/natansr/adan/assets/4833993/0052d05c-f533-4cfd-9c13-eb8782e6cb56)


# Authomatic Disambiguation Author Name (ADAN) - v.1.0

**ADAN** is a modern system for **Author Name Disambiguation (AND)** in digital bibliographic repositories. It draws inspiration from two solid foundations — **ComMAND** and **FCAND** — and aims to combine their strengths into a unified and extensible pipeline.

---

## 🔍 About This Project

This project builds upon two foundational frameworks for Author Name Disambiguation (AND): ComMAND and FCAND.

- **ComMAND** introduces a modular AND pipeline combining SciBERT-based embeddings, heterogeneous graph construction, Graph Convolutional Networks (GCN), and Graph-enhanced Hierarchical Agglomerative Clustering (GHAC), all accessible through a graphical interface.

- **FCAND** provides a flexible system with user-configurable settings, including the choice of transformer-based embeddings, adjustable GCN depth, and clustering methods, designed to adapt to varying data characteristics.

ADAN inherits and extends both systems by supporting customizable NLP embeddings, enabling options for GCN configuration and clustering strategy selection, and integrating ongoing development of visualization tools for cluster exploration.

---

## 🆕 What's New in ADAN

-  **Embeddings Selection:** Choose from NLP models like SciBERT, TF-IDF, or Word2Vec, MiniLM. ✅ **(Implemented)**
-  **Clustering Algorithm Selection:** Clusters documents using. Support for future options (e.g., K-Means, DBSCAN). 🛠️ *(Under Development)*
-  **GCN Architecture Configuration:** Set number of layers and parameters.
-  **Cluster Visualization:** Visualize document groups and embeddings. 🛠️ *(Under Development)*

---

## 🚀 How to Use

### Prerequisites

- Python 3.10+ recommended

### Installation

Clone the repository:

```bash
git clone https://github.com/natansr/adan
cd adan
pip install -r requirements.txt
```

---

## Project Structure

```
├── data_process/
│   ├── pre_processing.py
│   └── pre_process_ghac.py
├── datasets/                    # Input and processed data
├── gcn/
│   └── embedding_extraction_gcn.py
├── ghac/
│   └── ghac.py
├── het_network/
│   └── network_creation.py
├── nlp/
│   └── nlp.py
├── gui.py                       # Main GUI script
└── README.md
```

---

## Modules Overview

- **Pre-processing:** Filters raw JSON files and structures data.
- **Embedding Extraction:** Generates document vectors using BERT,SciBERT, TF-IDF,Word2Vec, and others models.
- **Graph Construction:** Builds a heterogeneous graph with authors, papers, venues, etc.
- **GCN:** Learns contextual node representations from graph topology.
- **Clustering:** Clusters documents using GHAC or other algorithms (future).
- **Evaluation:** Computes standard AND metrics.
- **Visualization:** (Under development) to display cluster output graphically.

---

## Graphical User Interface (GUI)

Run the GUI:

```bash
python gui.py
```

The GUI supports:

- Feature selection
- Pre-processing
- Embedding extraction
- Graph construction
- GCN training
- Clustering and evaluation
- Cluster visualization *(coming soon)*

---

## Input Format

Example JSON structure:

```json
{
  "id": "doc1",
  "title": "Graph-based Methods for Disambiguation",
  "abstract": "We propose a novel framework...",
  "venue": "ICDM",
  "coauthors": ["Jane Smith", "John Doe"],
  "keywords": ["disambiguation", "graph", "bert"],
  "label": "author_001"
}
```

---

## Evaluation Metrics

The system provides:

- Pairwise Precision / Recall / F1
- ACP (Average Cluster Purity)
- AAP (Average Author Purity)
- K-Metric
- B-cubed

---

## Workflow (via GUI)

1. Select raw JSON folder
2. Preprocess files
3. Choose and extract embeddings
4. Construct heterogeneous graph
5. Train GCN
6. Cluster documents and evaluate
7. (Optional) Visualize results *(under development)*

---

## Citation

> _Please cite the original ComMAND and FCAND frameworks when using ADAN:_

```
[Rodrigues, N. S. and Ralha, C. G. A Novel Framework with ComMAND: A Combi-
ned Method for Author Name Disambiguation. Information Processing & Manage-
ment. ]

[N. D. S. Rodrigues and C. G. Ralha, "A Flexible and Configurable System to Author Name Disambiguation," in IEEE Access, doi: 10.1109/ACCESS.2025.3589957., ]
```

---

Stay tuned for upcoming features and improvements!
