import re
import os
import json

path = 'raw-data_custand/'
file_names = os.listdir(path)

r = '[!“”"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~—～]+'

keyid = 0
papers = {}
authors = {}
jconfs = {}
word = {}
author1 = {}
ambiguous_names = 0

for fname in file_names:
    with open(path + fname, 'r', encoding='utf-8') as file:
        data = json.load(file)

        for item in data:
            pid = item['label']
            paper = item['title']
            authorlist = item['coauthors']
            jconf = item['keywords']

            # Processar os dados relacionados a papers, authors e jconfs
            papers[pid] = paper

            for author in authorlist:
                author = author.replace(" ", "")
                if author not in authors:
                    authors[author] = keyid
                    keyid += 1
                else:
                    ambiguous_names += 1

            jconf = jconf.strip().replace(" ", "")
            if jconf not in jconfs:
                jconfs[jconf] = keyid
                keyid += 1

            # Processar os dados de palavras
            line = re.sub(r, ' ', paper)
            line = line.replace('\t', ' ')
            line = line.lower()

            split_cut = line.split(' ')
            for j in split_cut:
                if len(j) > 1:  # removido a condição que verifica se a palavra está na lista de stopwords
                    if j not in word:
                        word[j] = 1
                    else:
                        word[j] += 1

print(len(author1), "ambiguous names.")
