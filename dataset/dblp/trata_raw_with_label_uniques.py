import pandas as pd
import random
import json

# Ler os dados do arquivo CSV
data = pd.read_csv('dataset/low_raw.csv', delimiter='\t')

# Preencher valores NaN nos abstracts com uma string vazia
data['Abstract'] = data['Abstract'].fillna('')

# Preencher valores NaN nos coautores com uma lista vazia
data['Author List'] = data['Author List'].fillna('')

# Criar um dataset vazio
dataset = []

# Criar uma lista para armazenar os labels únicos
unique_labels = []

# Iterar sobre cada linha dos dados
for index, row in data.iterrows():
    label = row['Label']
    author = row['Name']
    abstract = row['Abstract']
    coauthors = row['Author List']
    title = row['Title']
    
    # Remover NaN dos coautores se presentes
    coauthors = coauthors.split(';') if coauthors else []
    
    # Verificar se o label já está na lista de labels únicos
    if label not in unique_labels:
        # Se ainda não estiver, adicionar à lista de labels únicos
        unique_labels.append(label)
    
    # Obter o valor do label único (índice na lista + 1)
    unique_label = unique_labels.index(label) + 1
    
    # Criar o dicionário de dados
    data = {
        'label': unique_label,
        'author': author,
        'title': title,
        'abstract': abstract,
        'coauthors': coauthors
    }
    
    # Adicionar os dados ao dataset
    dataset.append(data)

# Embaralhar o dataset
random.shuffle(dataset)

# Dividir o dataset em treinamento (70%), teste (20%) e validação (10%)
train_size = int(0.7 * len(dataset))
test_size = int(0.2 * len(dataset))
train_data = dataset[:train_size]
test_data = dataset[train_size:train_size+test_size]
validation_data = dataset[train_size+test_size:]

# Exportar o conjunto de treinamento para um arquivo JSON com quebras de linha
with open('dataset/train_data.json', 'w') as f:
    json.dump(train_data, f, indent=4, separators=(',', ': '))

# Exportar o conjunto de teste para um arquivo JSON com quebras de linha
with open('dataset/test_data.json', 'w') as f:
    json.dump(test_data, f, indent=4, separators=(',', ': '))

# Exportar o conjunto de validação para um arquivo JSON com quebras de linha
with open('dataset/validation_data.json', 'w') as f:
    json.dump(validation_data, f, indent=4, separators=(',', ': '))

# Imprimir o total de labels
total_labels = len(unique_labels)
print("Total de labels:", total_labels)
