import os
import json
import numpy as np
import pandas as pd
import pickle
import ttkbootstrap as ttk
from ttkbootstrap.constants import *
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import pairwise_distances
from tkinter import filedialog, Text
from threading import Thread
from tqdm import tqdm
from datetime import datetime
import psutil

class GHAC:
    @staticmethod
    def load_gcn_embeddings(embedding_path):
        try:
            with open(embedding_path, "rb") as f:
                return pickle.load(f)
        except FileNotFoundError:
            print(f"❌ Embedding não encontrado: {embedding_path}")
            return None

    @staticmethod
    def GHAC(embeddings, n_clusters=-1):
        if len(embeddings) < 2:
            return np.array([])
        distance = pairwise_distances(embeddings, metric="cosine")
        if n_clusters != -1:
            model = AgglomerativeClustering(metric="precomputed", linkage="average", n_clusters=n_clusters)
        else:
            model = AgglomerativeClustering(metric="precomputed", linkage="average")
        return model.fit_predict(distance)

    @staticmethod
    def pairwise_evaluate(correct_labels, pred_labels):
        TP, TP_FP, TP_FN = 0.0, 0.0, 0.0
        n = len(correct_labels)
        if n < 2:
            return 0.0, 0.0, 0.0

        for i in range(n):
            for j in range(i + 1, n):
                same_true = (correct_labels[i] == correct_labels[j])
                same_pred = (pred_labels[i] == pred_labels[j])
                if same_true:
                    TP_FN += 1
                if same_pred:
                    TP_FP += 1
                if same_true and same_pred:
                    TP += 1

        pp = TP / TP_FP if TP_FP > 0 else 0.0
        pr = TP / TP_FN if TP_FN > 0 else 0.0
        pf1 = (2 * pp * pr) / (pp + pr) if (pp + pr) > 0 else 0.0
        return pp, pr, pf1

    @staticmethod
    def calculate_ACP_AAP(correct_labels, cluster_labels):
        correct_labels = np.asarray(correct_labels, dtype=int)
        cluster_labels = np.asarray(cluster_labels, dtype=int)

        acp = 0.0
        clusters = np.unique(cluster_labels)
        for c in clusters:
            idx = np.where(cluster_labels == c)[0]
            if len(idx) == 0: 
                continue
            labs = correct_labels[idx]
            acp += np.max(np.bincount(labs)) / len(idx)
        acp = acp / len(clusters) if len(clusters) > 0 else 0.0

        aap = 0.0
        authors = np.unique(correct_labels)
        for a in authors:
            idx = np.where(correct_labels == a)[0]
            if len(idx) == 0:
                continue
            preds = cluster_labels[idx]
            aap += np.max(np.bincount(preds)) / len(idx)
        aap = aap / len(authors) if len(authors) > 0 else 0.0

        return acp, aap

    @staticmethod
    def calculate_KMetric(ACP, AAP):
        return float(np.sqrt(ACP * AAP))

    @staticmethod
    def evaluate_one_embedding(embeddings_dict, json_dir, log_box=None):
        """Roda GHAC para TODOS os JSONs usando um único embeddings_dict; retorna DF por autor e um dict de médias."""
        results = []
        file_names = sorted([f for f in os.listdir(json_dir) if f.endswith('.json')])

        all_precision, all_recall, all_f1 = [], [], []
        all_acp, all_aap, all_k = [], [], []

        process = psutil.Process(os.getpid())
        mem_logged = False
        total_files = len(file_names)

        for idx, fname in enumerate(tqdm(file_names, desc="JSONs")):
            with open(os.path.join(json_dir, fname), 'r', encoding='utf-8') as file:
                data = json.load(file)

            unique_labels = list({entry['label'] for entry in data})
            label_mapping = {label: i for i, label in enumerate(unique_labels)}
            correct_all = [label_mapping[entry['label']] for entry in data]
            papers_all = [entry['id'] for entry in data]

            # alinhar com embeddings disponíveis
            X, y_true = [], []
            for k, pid in enumerate(papers_all):
                key = f"i{pid}"
                if key in embeddings_dict:
                    X.append(embeddings_dict[key])
                    y_true.append(correct_all[k])

            if len(y_true) < 2:
                if log_box:
                    log_box.insert("end", f"⚠️  Pulado (menos de 2 amostras): {fname}\n")
                    log_box.see("end")
                continue

            X = np.asarray(X)
            n_clusters = len(np.unique(y_true))
            y_pred = GHAC.GHAC(X, n_clusters=n_clusters)
            if y_pred.size == 0:
                continue

            pp, pr, pf1 = GHAC.pairwise_evaluate(y_true, y_pred)
            ACP, AAP = GHAC.calculate_ACP_AAP(y_true, y_pred)
            K = GHAC.calculate_KMetric(ACP, AAP)

            all_precision.append(pp); all_recall.append(pr); all_f1.append(pf1)
            all_acp.append(ACP); all_aap.append(AAP); all_k.append(K)
            results.append([fname, pp, pr, pf1, ACP, AAP, K])

            if log_box:
                log_box.insert("end", f"✔️  {fname}\n")
                log_box.see("end")

            if not mem_logged and (idx + 1) >= max(1, total_files // 2):
                mem_usage_mb = process.memory_info().rss / (1024 ** 2)
                msg = f"🧠 Memória (metade): {mem_usage_mb:.2f} MB"
                print(msg)
                if log_box:
                    log_box.insert("end", msg + "\n")
                    log_box.see("end")
                mem_logged = True

        if len(results) == 0:
            return None, None

        avg_row = ["AVERAGE",
                   float(np.mean(all_precision)),
                   float(np.mean(all_recall)),
                   float(np.mean(all_f1)),
                   float(np.mean(all_acp)),
                   float(np.mean(all_aap)),
                   float(np.mean(all_k))]
        results.append(avg_row)

        df = pd.DataFrame(results, columns=["Author", "Pairwise Precision", "Pairwise Recall", "Pairwise F1", "ACP", "AAP", "K"])
        avg_dict = {
            "pP": avg_row[1], "pR": avg_row[2], "pF1": avg_row[3],
            "ACP": avg_row[4], "AAP": avg_row[5], "K": avg_row[6]
        }
        return df, avg_dict


class GHACMultiRunApp(ttk.Window):
    def __init__(self):
        super().__init__(themename="morph")
        self.title("GHAC – Avaliar X embeddings de GCN (batch)")
        self.geometry("900x700")

        self.embeds_dir_var = ttk.StringVar()
        self.json_dir_var = ttk.StringVar()
        self.output_dir_var = ttk.StringVar()

        self.create_widgets()

    def create_widgets(self):
        frame = ttk.Frame(self, padding=10)
        frame.pack(fill=BOTH, expand=True)

        ttk.Label(frame, text="Pasta com embeddings (.pkl):", font=("Arial", 12, "bold")).pack(anchor=W, pady=4)
        ttk.Entry(frame, textvariable=self.embeds_dir_var, width=80).pack(pady=2)
        ttk.Button(frame, text="Selecionar pasta", bootstyle="info",
                   command=lambda: self.embeds_dir_var.set(filedialog.askdirectory())).pack(pady=4)

        ttk.Label(frame, text="Pasta dos JSONs (ground-truth):", font=("Arial", 12, "bold")).pack(anchor=W, pady=4)
        ttk.Entry(frame, textvariable=self.json_dir_var, width=80).pack(pady=2)
        ttk.Button(frame, text="Selecionar pasta", bootstyle="info",
                   command=lambda: self.json_dir_var.set(filedialog.askdirectory())).pack(pady=4)

        ttk.Label(frame, text="Pasta de saída (CSV por execução + summary_runs.csv):", font=("Arial", 12, "bold")).pack(anchor=W, pady=4)
        ttk.Entry(frame, textvariable=self.output_dir_var, width=80).pack(pady=2)
        ttk.Button(frame, text="Selecionar pasta", bootstyle="info",
                   command=lambda: self.output_dir_var.set(filedialog.askdirectory())).pack(pady=4)

        ttk.Button(frame, text="Iniciar batch", bootstyle="success", command=self.start_batch).pack(pady=10)

        self.log = Text(frame, wrap="word", height=22)
        self.log.pack(fill=BOTH, expand=True, pady=6)

    def start_batch(self):
        Thread(target=self.run_batch).start()

    def run_batch(self):
        embeds_dir = self.embeds_dir_var.get().strip()
        json_dir = self.json_dir_var.get().strip()
        out_dir = self.output_dir_var.get().strip()

        if not (os.path.isdir(embeds_dir) and os.path.isdir(json_dir) and os.path.isdir(out_dir)):
            self._log("❌ Caminhos inválidos. Confira as três pastas.")
            return

        embed_files = sorted([f for f in os.listdir(embeds_dir) if f.endswith(".pkl")])
        if len(embed_files) == 0:
            self._log("❌ Nenhum .pkl encontrado na pasta de embeddings.")
            return

        start = datetime.now()
        self._log(f"🟢 Início: {start.strftime('%Y-%m-%d %H:%M:%S')}")

        summary_rows = []

        for f in embed_files:
            emb_path = os.path.join(embeds_dir, f)
            self._log(f"\n📦 Carregando embeddings: {f}")
            embeddings = GHAC.load_gcn_embeddings(emb_path)
            if embeddings is None:
                self._log(f"⚠️  Pulando {f} (não carregou).")
                continue

            df, avg = GHAC.evaluate_one_embedding(embeddings, json_dir, log_box=self.log)
            if df is None:
                self._log(f"⚠️  Sem resultados para {f}.")
                continue

            base = os.path.splitext(f)[0]
            out_csv = os.path.join(out_dir, f"clustering_results__{base}.csv")
            df.to_csv(out_csv, index=False)
            self._log(f"💾 Salvo: {out_csv}")

            summary_rows.append({
                "run": base,
                "pP_avg": avg["pP"],
                "pR_avg": avg["pR"],
                "pF1_avg": avg["pF1"],
                "ACP_avg": avg["ACP"],
                "AAP_avg": avg["AAP"],
                "K_avg": avg["K"]
            })

        if len(summary_rows) > 0:
            summary_df = pd.DataFrame(summary_rows)
            summary_path = os.path.join(out_dir, "summary_runs.csv")
            summary_df.to_csv(summary_path, index=False)
            self._log(f"\n📊 RESUMO salvo: {summary_path}")
            self._log("   -> Use este arquivo para calcular média/desvio padrão entre as X execuções.")
        else:
            self._log("⚠️  Nada para resumir (confira se os JSONs e as chaves i<ID> batem com os embeddings).")

        end = datetime.now()
        self._log(f"\n✅ Fim: {end.strftime('%Y-%m-%d %H:%M:%S')}  ({(end-start).total_seconds():.2f}s)")

    def _log(self, msg):
        self.log.insert("end", msg + "\n")
        self.log.see("end")


if __name__ == "__main__":
    app = GHACMultiRunApp()
    app.mainloop()
