

![image](https://github.com/natansr/adan/assets/4833993/0052d05c-f533-4cfd-9c13-eb8782e6cb56)



# Authomatic Disambiguation Author Name (ADAN) - v.1.0

**ADAN** is a modern system for **Author Name Disambiguation (AND)** in digital bibliographic repositories. This version derives from two robust frameworks — **ComMAND** and **FCAND** — and extends them with new features focused on **visualization**, **user configurability**, and **modularity**.

---

## 🔍 About This Project

This project builds upon two foundational frameworks for Author Name Disambiguation (AND): [ComMAND](#citation) and [FCAND](#citation).

- **ComMAND** introduces a modular AND pipeline combining SciBERT-based embeddings, heterogeneous graph construction, Graph Convolutional Networks (GCN), and Graph-enhanced Hierarchical Agglomerative Clustering (GHAC), all accessible through a graphical interface.

- **FCAND** provides a flexible system with user-configurable settings, including the choice of transformer-based embeddings (e.g., MiniLM), adjustable GCN depth, and clustering methods, designed to adapt to varying data characteristics.

ADAN inherits and extends both systems by supporting customizable NLP embeddings, enabling future options for GCN configuration and clustering strategy selection, and integrating ongoing development of visualization tools for cluster exploration.

---

## 🆕 What's New in ADAN

- 🧠 **Embeddings Selection:** Choose from NLP models like SciBERT, TF-IDF, or Word2Vec. ✅ **(Implemented)**
- 🌐 **Clustering Algorithm Selection:** Support for future options (e.g., GHAC, DBSCAN). 🛠️ *(Under Development)*
- 🧱 **GCN Architecture Configuration:** Set number of layers and parameters. 🛠️ *(Under Development)*
- 📊 **Cluster Visualization:** Visualize document groups and embeddings. 🛠️ *(Under Development)*

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
pip uninstall community
pip install python-louvain
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
- **Embedding Extraction:** Generates document vectors using SciBERT, TF-IDF, or Word2Vec.
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
[Placeholder for citation to “A Novel Framework with ComMAND: A Combined Method for Author Name Disambiguation”]

[Placeholder for citation to “A Flexible and Configurable System for Author Name Disambiguation”]
```

---

## 🧠 Based On

ADAN (this version) extends the following prior works:

- **ComMAND** – A Combined Method for Author Name Disambiguation.
- **FCAND** – A Flexible and Configurable System for Author Name Disambiguation.
- Qiao, Ziyue et al. “Unsupervised Author Disambiguation using Heterogeneous Graph Convolutional Network Embedding.” *2019 IEEE International Conference on Big Data (Big Data)*, pp. 910–919.

---

Stay tuned for upcoming features and improvements!
