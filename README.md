# Authomatic Disambiguation Author Name (ADAN)

The Authomatic Disambiguation Author Name (ADAN) is an advanced system for disambiguating author names in digital bibliographic repositories. This project employs natural language processing (NLP) techniques, BERT embeddings, Graph Convolutional Networks (GCN), random walk, clustering, and features a user-friendly web interface to achieve high accuracy in resolving ambiguities in author names.

## About This Project


This project is a derivative of the original ADAN project with improvements, and it can be found [here](https://gitlab.com/InfoKnow/SocialNetwork/sci_clan/adan).

In contrast to the original ADAN system, this enhanced version does not utilize multi-agents. Instead, it leverages BERT and GCN to enhance text representations and achieve superior disambiguation results.


Also, this project is built upon and extends the work conducted by Qiao. Qiao, Ziyue, Yi Du, Yanjie Fu, Pengfei Wang, and Yuanchun Zhou. "[Unsupervised Author Disambiguation using Heterogeneous Graph Convolutional Network Embedding.](https://ieeexplore.ieee.org/abstract/document/9005458)" In 2019 IEEE International Conference on Big Data (Big Data), pp. 910-919. IEEE, 2019.

In contrast to the techniques mentioned in Qiao's work, we have adopted the use of the BERT model as a pre-trained embedding to enhance the quality of text representations in our system. Additionally, we have implemented a user-friendly web interface to facilitate interaction with the system.

## How to Use

### Prerequisites

- Python 3.6
- Python libraries listed in `requirements.txt`

## Basic requirements

* python 3.6.5
* networkx 1.11
* gensim 3.4.0
* sklearn 0.20.1
* numpy 1.14.3
* pandas 0.23.0
* tensorflow 1.10.0


### Installation

1. Clone this repository:

   ```bash
    git clone https://github.com/....

2. Install and unninstall the dependencies

    ```bash
    pip install -r requirements.txt
    pip uninstall community
    pip install python-louvain



### Steps


```bash
# step 1: preprocess the data
python data_processing.py

# step 2: train the GRU based encoder to learn deep semantic representations
python DRLgru.py 

# step 3: construct a PHNet and generate random walks
python walks.py

# step 4: weighted heterogeneous network embedding
python WHNE.py

# step 5: generate clustering results
python evaluator.py
```
