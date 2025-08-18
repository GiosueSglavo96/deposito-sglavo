import string
from collections import Counter

def conta_righe(testo):
    return len(testo.splitlines())

def conta_parole(testo):
    testo = testo.translate(str.maketrans('', '', string.punctuation)).lower()
    parole = testo.split()
    return len(parole)

def parole_frequenti(testo, top_n=5):
    testo = testo.translate(str.maketrans('', '', string.punctuation)).lower()
    parole = testo.split()
    frequenze = Counter(parole).most_common(top_n)
    return frequenze

nome_file = "C:\\Users\\KB316GR\\OneDrive - EY\\Desktop\\file.txt"

try:
    with open(nome_file, 'r', encoding='utf-8') as f:
        contenuto = f.read()
except FileNotFoundError:
    print(f"Errore: il file '{nome_file}' non è stato trovato.")
else:
    righe = conta_righe(contenuto)
    parole = conta_parole(contenuto)
    frequenti = parole_frequenti(contenuto)
