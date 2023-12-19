import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Carregue os rótulos dos clusters e representações (embeddings) do ground-truth
with open('results/cluster_results/John Collins.xml_correct_labels.pkl', 'rb') as file:
    ground_truth_labels = pickle.load(file)

with open('results/cluster_results/John Collins.xml_embeddings.pkl', 'rb') as file:
    ground_truth_embeddings = pickle.load(file)

# Carregue os rótulos dos clusters e representações (embeddings) das predições
with open('results/cluster_results/John Collins.xml_predicted_labels.pkl', 'rb') as file:
    prediction_labels = pickle.load(file)

with open('results/cluster_results/John Collins.xml_embeddings.pkl', 'rb') as file:
    prediction_embeddings = pickle.load(file)

# Redução de dimensionalidade com t-SNE para visualização em 2D do ground-truth
tsne_ground_truth = TSNE(n_components=2, perplexity=30, n_iter=300, random_state=42)
tsne_ground_truth_result = tsne_ground_truth.fit_transform(ground_truth_embeddings)

# Redução de dimensionalidade com t-SNE para visualização em 2D das predições
tsne_prediction = TSNE(n_components=2, perplexity=30, n_iter=300, random_state=42)
tsne_prediction_result = tsne_prediction.fit_transform(prediction_embeddings)

# Crie duas subtramas para exibir os gráficos lado a lado
fig, axs = plt.subplots(1, 2, figsize=(12, 5))

# Plote o t-SNE do ground-truth na primeira subtrama
axs[0].scatter(tsne_ground_truth_result[:, 0], tsne_ground_truth_result[:, 1], c=ground_truth_labels, cmap='viridis', s=200)
axs[0].set_title('t-SNE do Ground-Truth')
axs[0].set_xlabel('t-SNE Dimension 1')
axs[0].set_ylabel('t-SNE Dimension 2')

# Plote o t-SNE das predições na segunda subtrama
axs[1].scatter(tsne_prediction_result[:, 0], tsne_prediction_result[:, 1], c=prediction_labels, cmap='viridis', s=200)
axs[1].set_title('t-SNE das Predições')
axs[1].set_xlabel('t-SNE Dimension 1')
axs[1].set_ylabel('t-SNE Dimension 2')

# Ajuste o layout para evitar sobreposição
plt.tight_layout()

# Exiba os gráficos
plt.show()