import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import LocallyLinearEmbedding

def generate_author_plots(author):
    # Carregue os rótulos dos clusters e representações (embeddings) do ground-truth
    with open('results/cluster_results/' + author + '.xml_correct_labels.pkl', 'rb') as file:
        ground_truth_labels = pickle.load(file)

    with open('results/cluster_results/' + author + '.xml_embeddings.pkl', 'rb') as file:
        ground_truth_embeddings = pickle.load(file)

    # Carregue os rótulos dos clusters e representações (embeddings) das predições
    with open('results/cluster_results/' + author + '.xml_predicted_labels.pkl', 'rb') as file:
        prediction_labels = pickle.load(file)

    with open('results/cluster_results/' + author + '.xml_embeddings.pkl', 'rb') as file:
        prediction_embeddings = pickle.load(file)

    # Redução de dimensionalidade com LLE para visualização em 2D do ground-truth
    lle_ground_truth = LocallyLinearEmbedding(n_neighbors=10, n_components=2, method='standard')
    lle_ground_truth_result = lle_ground_truth.fit_transform(ground_truth_embeddings)

    # Redução de dimensionalidade com LLE para visualização em 2D das predições
    lle_prediction = LocallyLinearEmbedding(n_neighbors=10, n_components=2, method='standard')
    lle_prediction_result = lle_prediction.fit_transform(prediction_embeddings)

    # Crie duas subtramas para exibir os gráficos lado a lado
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    # Plote o LLE do ground-truth na primeira subtrama
    axs[0].scatter(lle_ground_truth_result[:, 0], lle_ground_truth_result[:, 1], c=ground_truth_labels, cmap='viridis', s=200)
    axs[0].set_title('LLE do Ground-Truth')
    axs[0].set_xlabel('LLE Dimension 1')
    axs[0].set_ylabel('LLE Dimension 2')

    # Plote o LLE das predições na segunda subtrama
    axs[1].scatter(lle_prediction_result[:, 0], lle_prediction_result[:, 1], c=prediction_labels, cmap='viridis', s=200)
    axs[1].set_title('LLE das Predições')
    axs[1].set_xlabel('LLE Dimension 1')
    axs[1].set_ylabel('LLE Dimension 2')

    # Ajuste o layout para evitar sobreposição
    plt.tight_layout()

    # Salvar a figura como uma única imagem
    plt.savefig('results/imgs/' + author + '_ground_truth_vs_prediction.png')

    plt.ioff()
    # Exiba a figura
    #plt.show(block=False)

# Lê os nomes dos autores de um arquivo de texto, um nome por linha
#with open('lista_autores.txt', 'r') as file:
    #authors = file.read().splitlines()

# Gera os gráficos para cada autor
#for author in authors:
    #generate_author_plots(author)



def main():
    generate_author_plots("Koichi Furukawa")

if __name__ == "__main__":
    main()
