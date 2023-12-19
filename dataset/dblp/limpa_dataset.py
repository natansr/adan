import csv
import re
import unicodedata

def normalize_text(text):
    normalized_text = unicodedata.normalize('NFKD', text)
    normalized_text = normalized_text.encode('ascii', 'ignore').decode('utf-8')
    return normalized_text

def clean_csv(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as file:
        reader = csv.reader(file, delimiter='\t')
        headers = next(reader)
        rows = list(reader)

    cleaned_rows = []

    for row in rows:
        if row:  # Ignora linhas vazias
            label = row[0]
            if re.match(r'^\d+$', label):  # Verifica se a linha começa com um numeral (label)
                if len(row) > 1 and any(row[1:]):
                    normalized_row = [normalize_text(cell) for cell in row]
                    cleaned_rows.append(normalized_row)

    with open(output_file, 'w', newline='') as file:
        writer = csv.writer(file, delimiter='\t')
        writer.writerow(headers)
        writer.writerows(cleaned_rows)


# Exemplo de uso:
input_file = 'dataset/full_raw.csv'
output_file = 'dataset/raw.csv'
clean_csv(input_file, output_file)
