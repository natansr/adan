import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
import numpy as np
import pickle
import os
import ttkbootstrap as ttk
from ttkbootstrap.constants import *
from tkinter import filedialog, Text
from threading import Thread
from datetime import datetime
import psutil  # monitorar memória

# GPU/CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---- IO helpers -------------------------------------------------------------
def load_embeddings(embedding_path):
    try:
        with open(embedding_path, "rb") as file_obj:
            return pickle.load(file_obj)
    except FileNotFoundError:
        print(f"❌ Error: Embedding file {embedding_path} not found.")
        return None

def save_embeddings(embeddings, nodes, save_path):
    embeddings_dict = {node: embeddings[idx] for idx, node in enumerate(nodes)}
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    with open(save_path, "wb") as file_obj:
        pickle.dump(embeddings_dict, file_obj)

# ---- Graph / Data -----------------------------------------------------------
def prepare_features(G, embeddings, device):
    nodes = list(G.nodes)
    node_idx_map = {node: idx for idx, node in enumerate(nodes)}
    edges = [(node_idx_map[u], node_idx_map[v]) for u, v in G.edges]
    if len(edges) == 0:
        # evita tensor vazio inválido
        edge_index = torch.empty((2,0), dtype=torch.long).to(device)
    else:
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous().to(device)

    if embeddings:
        sample_embedding = next(iter(embeddings.values()))
        embedding_dim = sample_embedding.shape[0]
        features = [embeddings.get(node, np.zeros(embedding_dim)) for node in nodes]
    else:
        embedding_dim = 128
        features = np.random.normal(loc=0.0, scale=1.0, size=(len(nodes), embedding_dim))

    x = torch.tensor(features, dtype=torch.float).to(device)
    return Data(x=x, edge_index=edge_index), nodes, embedding_dim

# ---- Model ------------------------------------------------------------------
class GCN(torch.nn.Module):
    def __init__(self, input_dim, num_layers):
        super(GCN, self).__init__()
        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(input_dim, 512))
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(512, 512))
        self.convs.append(GCNConv(512, 512))
        self.fc = torch.nn.Linear(512, input_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
        x = self.fc(x)
        return x

# ---- Train ------------------------------------------------------------------
def train_gcn_once(data, input_dim, num_layers, epochs, progress_bar, progress_label, output_box, run_idx=None, run_total=None):
    # título do run
    run_prefix = f"[Run {run_idx}/{run_total}] " if (run_idx is not None and run_total is not None) else ""

    model = GCN(input_dim, num_layers).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

    def step():
        model.train()
        optimizer.zero_grad()
        out = model(data)
        loss = F.mse_loss(out, data.x)
        loss.backward()
        optimizer.step()
        return loss

    start_time = datetime.now()
    start_msg = f"{run_prefix}🟢 Treinamento iniciado: {start_time.strftime('%Y-%m-%d %H:%M:%S')}"
    print(start_msg); output_box.insert("end", start_msg + "\n"); output_box.see("end")

    process = psutil.Process(os.getpid())
    memory_logged = False

    for epoch in range(epochs):
        loss = step()
        progress = ((epoch + 1) / epochs) * 100
        progress_bar["value"] = progress
        progress_bar.update()
        progress_label.config(text=f"{run_prefix}⚙ Epoch {epoch + 1}/{epochs} - Loss: {loss.item():.4f}")
        output_box.insert("end", f"{run_prefix}Epoch {epoch + 1}, Loss: {loss.item():.4f}\n")
        output_box.see("end")

        if not memory_logged and epoch + 1 == max(1, epochs // 2):
            mem_usage_mb = process.memory_info().rss / (1024 ** 2)
            mem_msg = f"{run_prefix}🧠 Memória (metade das épocas): {mem_usage_mb:.2f} MB"
            print(mem_msg); output_box.insert("end", mem_msg + "\n"); output_box.see("end")
            memory_logged = True

    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    end_msg = f"{run_prefix}✅ Finalizado: {end_time.strftime('%Y-%m-%d %H:%M:%S')} | ⏱️ {duration:.2f}s"
    print(end_msg); output_box.insert("end", end_msg + "\n"); output_box.see("end")

    model.eval()
    with torch.no_grad():
        new_embeddings = model(data).cpu().numpy()

    return new_embeddings

# ---- GUI --------------------------------------------------------------------
class GCNApp(ttk.Window):
    def __init__(self):
        super().__init__(themename="morph")
        self.title("GCN Embedding Training")
        self.geometry("820x760")

        self.network_file_var = ttk.StringVar()
        self.embedding_file_var = ttk.StringVar()
        self.num_layers_var = ttk.IntVar(value=2)
        self.epochs_var = ttk.IntVar(value=1000)
        self.repeats_var = ttk.IntVar(value=10)   # <-- novo: número de repetições (X)

        self.create_widgets()

    def create_widgets(self):
        frame = ttk.Frame(self, padding=10)
        frame.pack(fill=BOTH, expand=True)

        ttk.Label(frame, text="Heterogeneous Graph File (.pkl):", font=("Arial", 12, "bold")).pack(anchor=W, pady=5)
        ttk.Entry(frame, textvariable=self.network_file_var, width=50, bootstyle="primary").pack(pady=5)
        ttk.Button(frame, text="Browse", bootstyle="info-outline",
                   command=lambda: self.network_file_var.set(filedialog.askopenfilename(filetypes=[("PKL Files", "*.pkl")], title="Select Graph File"))).pack(pady=5)

        ttk.Label(frame, text="NLP Embeddings File (.pkl):", font=("Arial", 12, "bold")).pack(anchor=W, pady=5)
        ttk.Entry(frame, textvariable=self.embedding_file_var, width=50, bootstyle="primary").pack(pady=5)
        ttk.Button(frame, text="Browse", bootstyle="info-outline",
                   command=lambda: self.embedding_file_var.set(filedialog.askopenfilename(filetypes=[("PKL Files", "*.pkl")], title="Select Embeddings File"))).pack(pady=5)

        row = ttk.Frame(frame); row.pack(fill=X, pady=5)
        ttk.Label(row, text="Number of Layers:", font=("Arial", 12, "bold")).pack(side=LEFT)
        ttk.Entry(row, textvariable=self.num_layers_var, width=8, bootstyle="primary").pack(side=LEFT, padx=8)

        row2 = ttk.Frame(frame); row2.pack(fill=X, pady=5)
        ttk.Label(row2, text="Number of Epochs:", font=("Arial", 12, "bold")).pack(side=LEFT)
        ttk.Entry(row2, textvariable=self.epochs_var, width=8, bootstyle="primary").pack(side=LEFT, padx=8)

        # novo: repetições
        row3 = ttk.Frame(frame); row3.pack(fill=X, pady=5)
        ttk.Label(row3, text="Repetições (X runs):", font=("Arial", 12, "bold")).pack(side=LEFT)
        ttk.Entry(row3, textvariable=self.repeats_var, width=8, bootstyle="primary").pack(side=LEFT, padx=8)

        ttk.Button(frame, text="Start Training", bootstyle="success", command=self.start_training).pack(pady=16)

        self.progress_bar = ttk.Progressbar(frame, orient="horizontal", length=500, mode="determinate", bootstyle="success")
        self.progress_bar.pack(pady=8)

        self.progress_label = ttk.Label(frame, text="")
        self.progress_label.pack(pady=4)

        result_frame = ttk.Labelframe(frame, text="Output", bootstyle="primary", padding=10)
        result_frame.pack(fill=BOTH, expand=True, pady=10)

        self.output_box = Text(result_frame, wrap="word", height=18, width=90)
        self.output_box.pack(side=LEFT, fill=BOTH, expand=True)

        scrollbar = ttk.Scrollbar(result_frame, command=self.output_box.yview)
        scrollbar.pack(side=RIGHT, fill=Y)
        self.output_box.config(yscrollcommand=scrollbar.set)

    def start_training(self):
        Thread(target=self.run_gcn_training_gui).start()

    def run_gcn_training_gui(self):
        network_file = self.network_file_var.get().strip()
        embedding_file = self.embedding_file_var.get().strip()
        num_layers = int(self.num_layers_var.get())
        epochs = int(self.epochs_var.get())
        repeats = max(1, int(self.repeats_var.get()))

        # carregar grafo e embeddings base
        with open(network_file, 'rb') as file:
            G = pickle.load(file)
        base_embeddings = load_embeddings(embedding_file)

        data, nodes, input_dim = prepare_features(G, base_embeddings, device)

        base_dir, base_name = os.path.split(embedding_file)
        stem, ext = os.path.splitext(base_name)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        for r in range(1, repeats + 1):
            # treina 1 vez
            new_embeddings = train_gcn_once(
                data=data,
                input_dim=input_dim,
                num_layers=num_layers,
                epochs=epochs,
                progress_bar=self.progress_bar,
                progress_label=self.progress_label,
                output_box=self.output_box,
                run_idx=r,
                run_total=repeats
            )

            # salva com nome único: <stem>_gcn_run{r}.pkl (e opcional timestamp)
            out_name = f"{stem}_gcn_run{r}.pkl"
            # se quiser incluir timestamp, use: out_name = f"{stem}_gcn_{timestamp}_run{r}.pkl"
            save_path = os.path.join(base_dir, out_name)
            save_embeddings(new_embeddings, nodes, save_path)

            done_msg = f"[Run {r}/{repeats}] 💾 Embeddings salvos em: {save_path}"
            print(done_msg); self.output_box.insert("end", done_msg + "\n"); self.output_box.see("end")


if __name__ == "__main__":
    app = GCNApp()
    app.mainloop()
