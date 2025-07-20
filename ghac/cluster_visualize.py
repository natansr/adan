import os
import json
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.cluster import AgglomerativeClustering
from scipy.sparse.csgraph import connected_components
import networkx as nx
from tkinter import filedialog, Text
import ttkbootstrap as ttk
from ttkbootstrap.constants import *

class ClusterVisualizer(ttk.Window):
    def __init__(self):
        super().__init__(themename="morph")
        self.title("GHAC Visualization")
        self.geometry("800x600")

        self.embedding_path_var = ttk.StringVar()
        self.json_path_var = ttk.StringVar()
        self.output_dir_var = ttk.StringVar(value="visualizations")

        self.create_widgets()

    def create_widgets(self):
        frame = ttk.Frame(self, padding=10)
        frame.pack(fill="both", expand=True)

        ttk.Label(frame, text="Embedding .pkl File:", font=("Arial", 12)).pack(anchor="w")
        ttk.Entry(frame, textvariable=self.embedding_path_var, width=70).pack()
        ttk.Button(frame, text="Select Embedding", command=self.select_embedding).pack(pady=5)

        ttk.Label(frame, text="JSON File:", font=("Arial", 12)).pack(anchor="w")
        ttk.Entry(frame, textvariable=self.json_path_var, width=70).pack()
        ttk.Button(frame, text="Select JSON", command=self.select_json).pack(pady=5)

        ttk.Label(frame, text="Output Folder:", font=("Arial", 12)).pack(anchor="w")
        ttk.Entry(frame, textvariable=self.output_dir_var, width=70).pack()
        ttk.Button(frame, text="Select Folder", command=self.select_output_folder).pack(pady=5)

        ttk.Button(frame, text="Visualize Clusters", command=self.visualize).pack(pady=10)

        self.result_box = Text(frame, wrap="word", height=15)
        self.result_box.pack(fill="both", expand=True)

    def select_embedding(self):
        file_path = filedialog.askopenfilename(filetypes=[("PKL Files", "*.pkl")])
        if file_path:
            self.embedding_path_var.set(file_path)

    def select_json(self):
        file_path = filedialog.askopenfilename(filetypes=[("JSON Files", "*.json")])
        if file_path:
            self.json_path_var.set(file_path)

    def select_output_folder(self):
        folder_path = filedialog.askdirectory()
        if folder_path:
            self.output_dir_var.set(folder_path)

    def ghac_custom(self, mlist, n_clusters=-1):
        distance = []
        for i in range(len(mlist)):
            gtmp = []
            for j in range(len(mlist)):
                if i < j:
                    cosdis = np.dot(mlist[i], mlist[j]) / (np.linalg.norm(mlist[i]) * np.linalg.norm(mlist[j]))
                    gtmp.append(cosdis)
                elif i > j:
                    gtmp.append(distance[j][i])
                else:
                    gtmp.append(0)
            distance.append(gtmp)

        distance = np.array(distance)
        distance = np.multiply(distance, -1)

        if n_clusters == -1:
            best_m = -10000000
            n_components1, _ = connected_components(distance)

            distance[distance <= 0.5] = 0
            G = nx.from_numpy_matrix(distance)
            n_components, _ = connected_components(distance)

            for k in range(n_components, n_components1 - 1, -1):
                model_HAC = AgglomerativeClustering(linkage="average", metric='precomputed', n_clusters=k)
                model_HAC.fit(distance)
                labels = model_HAC.labels_

                mod = nx.algorithms.community.quality.modularity(
                    G,
                    [set(np.where(np.array(labels) == i)[0]) for i in range(len(set(labels)))]
                )
                if mod > best_m:
                    best_m = mod
                    best_labels = labels
            labels = best_labels
        else:
            model_HAC = AgglomerativeClustering(linkage='average', metric='precomputed', n_clusters=n_clusters)
            model_HAC.fit(distance)
            labels = model_HAC.labels_

        return labels

    def visualize(self):
        pkl_path = self.embedding_path_var.get()
        json_path = self.json_path_var.get()
        output_dir = self.output_dir_var.get()

        if not os.path.isfile(pkl_path) or not os.path.isfile(json_path):
            self.result_box.insert("end", "❌ Verifique se os arquivos foram selecionados corretamente.\n")
            return

        with open(pkl_path, "rb") as f:
            embeddings = pickle.load(f)

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        papers, labels, vectors = [], [], []
        for entry in data:
            pid = f"i{entry['id']}"
            if pid in embeddings and not np.all(embeddings[pid] == 0):
                papers.append(pid)
                labels.append(int(entry['label']))
                vectors.append(embeddings[pid])

        if len(papers) < 2:
            self.result_box.insert("end", "⚠️ Poucas publicações válidas para visualização.\n")
            return

        vectors = np.array(vectors)
        labels = np.array(labels)

        predicted = self.ghac_custom(vectors.tolist(), n_clusters=len(set(labels)))

        author_name = os.path.basename(json_path).replace(".json", "")

        self.visualize_tsne(vectors, labels, author_name, "Rotulos Reais", output_dir)
        self.visualize_tsne(vectors, predicted, author_name, "Rotulos Previstos", output_dir)

        self.result_box.insert("end", f"✅ Visualizações geradas para {author_name}\n")
        self.result_box.see("end")

    def visualize_tsne(self, vectors, labels, author_name, title, output_dir):

        plt.rcParams["font.family"] = "Times New Roman"  # Ou "Times New Roman"

        vectors = np.array(vectors)
        n_samples = vectors.shape[0]

        if n_samples < 3:
            self.result_box.insert("end", "❌ Número insuficiente de amostras para t-SNE.\n")
            return

        perplexity = min(30, max(2, n_samples - 1))

        tsne = TSNE(
            n_components=2,
            perplexity=perplexity,
            n_iter=1000,
            random_state=42,
            init="pca"
        )

        reduced = tsne.fit_transform(vectors)

        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(reduced[:, 0], reduced[:, 1], c=labels, cmap='tab10', s=400)

        for i, label in enumerate(labels):
            plt.annotate(label, (reduced[i, 0], reduced[i, 1]), fontsize=18, fontweight="bold")

        #Tirando o titulo da imagem ou colocando, basta comentar.
        plt.title(f"{title} - {author_name}", fontsize=16)

        plt.xticks([])
        plt.yticks([])
        plt.box(True)

        out_path = os.path.join(output_dir, f"{author_name}_{title.replace(' ', '_')}.pdf")
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(out_path, bbox_inches='tight')
        plt.close()

if __name__ == "__main__":
    app = ClusterVisualizer()
    app.mainloop()
