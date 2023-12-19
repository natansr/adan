import os
import pandas as pd
from lxml import etree

# Caminho do diretório com arquivos XML
xml_dir = 'xmls'

# Inicializar um array para armazenar todos os dados
data = []

# Inicializar o contador de documentos
doc_counter = 0

# Inicializar a variável para a maior label
max_label = -1

# Loop sobre todos os arquivos XML no diretório
for filename in os.listdir(xml_dir):
    if filename.endswith('.xml'):
        # Carrega o arquivo XML
        parser = etree.XMLParser(recover=True)
        tree = etree.parse(os.path.join(xml_dir, filename), parser)
        root = tree.getroot()
        
        # Loop sobre todas as publicações na tag 'publications'
        for pub in root.find('publications'):
            # Substitua ',' por ';' na lista de autores
            authors = pub.find('authors').text.replace(',', ';')

            # Confira se a tag 'conference' existe antes de tentar acessar o texto
            if pub.find('conference') is not None:
                venue = pub.find('conference').text
            else:
                venue = ''  # Ou algum outro valor padrão que faça sentido

            # Aqui, nós adicionamos o contador do documento ao valor da label
            label = int(pub.find('label').text) + doc_counter * 10

            # Atualiza a maior label, se necessário
            if label > max_label:
                max_label = label

            # Adiciona os dados ao conjunto de dados
            data.append({
                'Label': str(label),
                'Name': root.find('FullName').text,
                'Author List': authors,
                'Title': pub.find('title').text,
                'Abstract': '',  # Como o exemplo do XML não tem abstract, deixei vazio
                'Keywords': '',  # Mesmo caso para Keywords
                'Venue': venue,
                'Download links': '',  # Mesmo caso para Download links
                'Publish Year': pub.find('year').text
            })
        
        # Incrementa o contador de documentos após processar cada arquivo XML
        doc_counter += 1

# Imprime a maior label
print("A maior label é:", max_label)

# Converta os dados para um DataFrame e salve em um arquivo CSV
df = pd.DataFrame(data)
df.to_csv('raw.csv', index=False, sep='\t')
