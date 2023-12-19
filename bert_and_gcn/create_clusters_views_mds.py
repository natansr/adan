import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import MDS

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

# Redução de dimensionalidade com MDS para visualização em 2D do ground-truth
mds_ground_truth = MDS(n_components=2, random_state=42)
mds_ground_truth_result = mds_ground_truth.fit_transform(ground_truth_embeddings)

# Redução de dimensionalidade com MDS para visualização em 2D das predições
mds_prediction = MDS(n_components=2, random_state=42)
mds_prediction_result = mds_prediction.fit_transform(prediction_embeddings)

# Crie duas subtramas para exibir os gráficos lado a lado
fig, axs = plt.subplots(1, 2, figsize=(12, 5))

# Plote o MDS do ground-truth na primeira subtrama
axs[0].scatter(mds_ground_truth_result[:, 0], mds_ground_truth_result[:, 1], c=ground_truth_labels, cmap='viridis', s=200)
axs[0].set_title('MDS do Ground-Truth')
axs[0].set_xlabel('MDS Dimension 1')
axs[0].set_ylabel('MDS Dimension 2')

# Plote o MDS das predições na segunda subtrama
axs[1].scatter(mds_prediction_result[:, 0], mds_prediction_result[:, 1], c=prediction_labels, cmap='viridis', s=200)
axs[1].set_title('MDS das Predições')
axs[1].set_xlabel('MDS Dimension 1')
axs[1].set_ylabel('MDS Dimension 2')

# Ajuste o layout para evitar sobreposição
plt.tight_layout()

# Exiba os gráficos
plt.show()
