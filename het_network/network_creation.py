import os
import pickle
import networkx as nx
from tkinter import filedialog, Tk
from datetime import datetime
import psutil  # 🧠 Para medir memória usada pelo processo

class HeterogeneousNetworkGenerator:
    def __init__(self):
        self.G = nx.Graph()

    def read_data(self, dirpath, selected_features):
        # ⏱️ Início do processamento
        start_time = datetime.now()
        print(f"🕐 Início da geração da rede: {start_time.strftime('%H:%M:%S')}")

        print("Lendo dados dos arquivos...")
        try:
            if 'word' in selected_features:
                with open(os.path.join(dirpath, "paper_word.txt"), encoding='utf-8') as file:
                    for line in file:
                        toks = line.strip().split("\t")
                        if len(toks) == 2:
                            paper_id, word = toks
                            self.G.add_node(paper_id, type='paper')
                            self.G.add_node(word, type='word')
                            self.G.add_edge(paper_id, word, relationship='contains')

            if 'author' in selected_features:
                with open(os.path.join(dirpath, "paper_author.txt"), encoding='utf-8') as file:
                    for line in file:
                        toks = line.strip().split("\t")
                        if len(toks) == 2:
                            paper_id, author_id = toks
                            self.G.add_node(paper_id, type='paper')
                            self.G.add_node(author_id, type='author')
                            self.G.add_edge(paper_id, author_id, relationship='written_by')

            if 'venue_name' in selected_features:
                with open(os.path.join(dirpath, "paper_venue.txt"), encoding='utf-8') as file:
                    for line in file:
                        toks = line.strip().split("\t")
                        if len(toks) == 2:
                            paper_id, conf_id = toks
                            self.G.add_node(paper_id, type='paper')
                            self.G.add_node(conf_id, type='conference')
                            self.G.add_edge(paper_id, conf_id, relationship='presented_at')

            if 'title' in selected_features:
                with open(os.path.join(dirpath, "paper_title.txt"), encoding='utf-8') as file:
                    for line in file:
                        toks = line.strip().split("\t", 1)
                        if len(toks) == 2:
                            paper_id, title = toks
                            title_node_id = f'title_{paper_id}'
                            self.G.add_node(title_node_id, type='title', content=title)
                            self.G.add_edge(paper_id, title_node_id, relationship='has_title')

            if 'abstract' in selected_features:
                with open(os.path.join(dirpath, "paper_abstract.txt"), encoding='utf-8') as file:
                    for line in file:
                        toks = line.strip().split("\t", 1)
                        if len(toks) == 2:
                            paper_id, abstract = toks
                            abstract_node_id = f'abstract_{paper_id}'
                            self.G.add_node(abstract_node_id, type='abstract', content=abstract)
                            self.G.add_edge(paper_id, abstract_node_id, relationship='has_abstract')

            if 'author_names' in selected_features:
                with open(os.path.join(dirpath, "paper_author_names.txt"), encoding='utf-8') as file:
                    for line in file:
                        toks = line.strip().split("\t", 1)
                        if len(toks) == 2:
                            paper_id, author_names = toks
                            author_names_node_id = f'author_names_{paper_id}'
                            self.G.add_node(author_names_node_id, type='author_names', content=author_names)
                            self.G.add_edge(paper_id, author_names_node_id, relationship='has_author_names')

            org_path = os.path.join(dirpath, "paper_organization.txt")
            if 'organization' in selected_features and os.path.exists(org_path):
                with open(org_path, encoding='utf-8') as file:
                    for line in file:
                        toks = line.strip().split("\t")
                        if len(toks) == 2:
                            paper_id, organization = toks
                            org_node_id = f'org_{organization}'
                            self.G.add_node(org_node_id, type='organization', name=organization)
                            self.G.add_edge(paper_id, org_node_id, relationship='affiliated_with')

            print(f"📌 Nós adicionados: {self.G.number_of_nodes()}")
            print(f"🔗 Arestas adicionadas: {self.G.number_of_edges()}")

            # 🧠 Medição de memória ao final da construção
            process = psutil.Process(os.getpid())
            mem_mb = process.memory_info().rss / (1024 ** 2)
            print(f"🧠 Memória utilizada após leitura: {mem_mb:.2f} MB")

        except Exception as e:
            print(f"❌ Erro ao ler dados: {e}")

        # ⏱️ Fim do processamento
        end_time = datetime.now()
        print(f"✅ Fim da geração da rede: {end_time.strftime('%H:%M:%S')}")
        print(f"⏱️ Tempo total: {(end_time - start_time).total_seconds():.2f} segundos")

    def save_network(self, filepath):
        try:
            with open(filepath, 'wb') as file:
                pickle.dump(self.G, file)
            print("💾 Rede heterogênea salva com sucesso.")
        except Exception as e:
            print(f"❌ Erro ao salvar a rede: {e}")

def main(selected_features):
    root = Tk()
    root.withdraw()

    dirpath = filedialog.askdirectory(title="Selecione o diretório contendo os arquivos de entrada")
    if not dirpath:
        print("❌ Nenhum diretório selecionado.")
        return

    filepath = filedialog.asksaveasfilename(
        title="Selecione o local para salvar a rede heterogênea",
        defaultextension=".pkl",
        filetypes=[("Arquivos PKL", "*.pkl")]
    )
    if not filepath:
        print("❌ Nenhum caminho de saída selecionado.")
        return

    hng = HeterogeneousNetworkGenerator()
    hng.read_data(dirpath, selected_features)
    hng.save_network(filepath)

if __name__ == "__main__":
    selected_features = ['word', 'author', 'venue_name', 'title', 'abstract', 'author_names', 'organization']
    main(selected_features)
