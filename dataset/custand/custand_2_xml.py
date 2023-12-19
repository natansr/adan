import csv
import xml.etree.ElementTree as ET
import pandas as pd

# Ler o arquivo CSV e carregar os dados em um DataFrame
path = "full_custand.csv"
try:
    df = pd.read_csv(path, encoding='utf-8', delimiter=';', error_bad_lines=False)
except pd.errors.ParserError as e:
    print(f"Erro na leitura do CSV: {e}")
    exit(1)

# Criar a estrutura básica do XML
root = ET.Element("persons")

# Iterar sobre cada linha do DataFrame para criar os elementos XML
for index, row in df.iterrows():
    person = ET.SubElement(root, "person")
    
    personID = ET.SubElement(person, "personID")
    personID.text = str(row["personID"])
    
    FullName = ET.SubElement(person, "FullName")
    FullName.text = row["FullName"]
    
    FirstName, LastName = row["FullName"].split(maxsplit=1)
    FirstNameElement = ET.SubElement(person, "FirstName")
    FirstNameElement.text = FirstName
    
    LastNameElement = ET.SubElement(person, "LastName")
    LastNameElement.text = LastName
    
    publication = ET.SubElement(person, "publication")
    
    title = ET.SubElement(publication, "title")
    title.text = row["title"]
    
    year = ET.SubElement(publication, "year")
    year.text = str(row["year"])
    
    authors = ET.SubElement(publication, "authors")
    authors.text = row["authors"]
    
    jconf = ET.SubElement(publication, "jconf")
    jconf.text = row["jconf"]
    
    publication_id = ET.SubElement(publication, "id")
    publication_id.text = str(row["id"])
    
    label = ET.SubElement(publication, "label")
    label.text = str(row["label"])
    
    organization = ET.SubElement(publication, "organization")
    organization.text = row["organization"]
    
# Criar o objeto ElementTree para salvar o XML
tree = ET.ElementTree(root)

# Salvar o XML em um arquivo
xml_file = "output.xml"
tree.write(xml_file, encoding="utf-8", xml_declaration=True)

print(f"XML gerado e salvo em {xml_file}.")
