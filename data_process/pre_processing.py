import json
import re
import os
from tkinter import filedialog, Tk, messagebox
from tqdm import tqdm
from datetime import datetime
import psutil  # 🧠 Para medir uso de memória

# Função principal de pré-processamento para processar vários arquivos JSON
def preprocess_data(input_dir, output_dir, selected_features):
    if not input_dir or not output_dir:
        raise ValueError("Os diretórios de entrada e saída não podem ser vazios.")

    os.makedirs(output_dir, exist_ok=True)

    # ⏱️ Início da medição de tempo
    start_time = datetime.now()
    print(f"🕐 Início do pré-processamento: {start_time.strftime('%H:%M:%S')}")

    # Dicionários para mapeamentos
    authors_map = {}
    venues_map = {}
    word_map = {}
    keyid = 0

    # Regex e stopwords
    r = r'[!“”"#$%&\'()*+,\-./:;<=>?@[\\]^_`{|}~—～]+'
    stopword = ['at', 'based', 'in', 'of', 'for', 'on', 'and', 'to', 'an', 'using', 'with', 'the']

    # Carrega arquivos JSON válidos
    all_data = []
    for file_name in os.listdir(input_dir):
        if file_name.lower().endswith(".json"):
            json_path = os.path.join(input_dir, file_name)
            with open(json_path, 'r', encoding='utf-8') as json_file:
                try:
                    file_data = json.load(json_file)
                    all_data.extend(file_data)
                except json.JSONDecodeError:
                    print(f"⚠️ Arquivo {file_name} ignorado: JSON inválido.")

    if not all_data:
        raise ValueError("Nenhum dado válido encontrado nos arquivos JSON do diretório.")

    # Abre arquivos de saída
    with open(os.path.join(output_dir, 'paper_author.txt'), 'w', encoding='utf-8') as f_author_id, \
         open(os.path.join(output_dir, 'paper_venue.txt'), 'w', encoding='utf-8') as f_venue_id, \
         open(os.path.join(output_dir, 'paper_word.txt'), 'w', encoding='utf-8') as f_word, \
         open(os.path.join(output_dir, 'paper_title.txt'), 'w', encoding='utf-8') as f_title, \
         open(os.path.join(output_dir, 'paper_author_names.txt'), 'w', encoding='utf-8') as f_author_names, \
         open(os.path.join(output_dir, 'paper_abstract.txt'), 'w', encoding='utf-8') as f_abstract, \
         open(os.path.join(output_dir, 'paper_venue_name.txt'), 'w', encoding='utf-8') as f_venue_name:

        for entry in tqdm(all_data, desc="Processando JSON(s)"):
            pid = entry.get('id')
            label = entry.get('label')

            if pid is None or label is None:
                continue

            # Campos textuais
            title = str(entry.get('title', '')).strip()
            abstract = str(entry.get('abstract', '')).strip()
            venue = str(entry.get('conf') or entry.get('venue', '')).strip()

            # Extrair autores considerando os diferentes formatos possíveis
            authors_str = entry.get('authors')  # String única separada por vírgula
            if authors_str:
                all_authors = [a.strip() for a in authors_str.split(',') if a.strip()]
            else:
                author_main = str(entry.get('author', '')).strip()
                coauthors = entry.get('coauthors', [])
                all_authors = [author_main] + coauthors if author_main else coauthors

            # Grava o abstract
            if 'abstract' in selected_features and abstract:
                f_abstract.write(f'i{pid}\t{abstract}\n')

            # Grava autor(es)
            if all_authors:
                for author_name in all_authors:
                    author_clean = author_name.replace(" ", "")
                    if author_clean not in authors_map:
                        authors_map[author_clean] = keyid
                        keyid += 1
                    f_author_id.write(f'i{pid}\t{authors_map[author_clean]}\n')

                f_author_names.write(f'i{pid}\t{",".join(all_authors)}\n')

            # Grava conferência/venue
            if 'venue_name' in selected_features and venue:
                if venue not in venues_map:
                    venues_map[venue] = keyid
                    keyid += 1
                f_venue_id.write(f'i{pid}\t{venues_map[venue]}\n')
                f_venue_name.write(f'i{pid}\t{venue}\n')

            # Limpa e grava título
            title_cleaned = re.sub(r, ' ', title).replace('\t', ' ').lower()
            f_title.write(f'i{pid}\t{title_cleaned}\n')

            # Contagem de palavras
            if 'word' in selected_features and title_cleaned:
                for w in title_cleaned.split():
                    if w and w not in stopword:
                        word_map[w] = word_map.get(w, 0) + 1

        # Segunda passada para escrever palavras com frequência ≥ 2
        if 'word' in selected_features:
            for entry in tqdm(all_data, desc="Escrevendo palavras frequentes"):
                pid = entry.get('id')
                if pid is None:
                    continue
                title = str(entry.get('title', '')).strip()
                title_cleaned = re.sub(r, ' ', title).replace('\t', ' ').lower()
                for w in title_cleaned.split():
                    if word_map.get(w, 0) >= 2:
                        f_word.write(f'i{pid}\t{w}\n')

    # ⏱️ Fim do processamento
    end_time = datetime.now()
    print(f"✅ Fim do pré-processamento: {end_time.strftime('%H:%M:%S')}")
    print(f"⏱️ Tempo total: {(end_time - start_time).total_seconds():.2f} segundos")

    # 🧠 Memória usada
    process = psutil.Process(os.getpid())
    mem_used_mb = process.memory_info().rss / (1024 ** 2)
    print(f"🧠 Memória utilizada: {mem_used_mb:.2f} MB")

    messagebox.showinfo("Concluído", "Processamento de múltiplos JSON concluído com sucesso!")


# Função principal para seleção de caminhos e execução
def run_pre_processing(input_dir, output_dir, selected_features):
    if selected_features is None:
        selected_features = []
    preprocess_data(input_dir, output_dir, selected_features)


if __name__ == "__main__":
    root = Tk()
    root.withdraw()
    input_dir = filedialog.askdirectory(title="Selecione a pasta com os arquivos JSON")
    output_dir = filedialog.askdirectory(title="Selecione o diretório de saída")
    selected_features = ['abstract', 'venue_name', 'word']  # Você pode mudar aqui

    if input_dir and output_dir:
        run_pre_processing(input_dir, output_dir, selected_features)
    else:
        messagebox.showerror("Erro", "Seleção de diretório cancelada.")
