import string
from collections import Counter

def conta_righe(testo):
    return len(testo.splitlines())

def conta_parole(testo):
    testo = testo.translate(str.maketrans('', '', string.punctuation)).lower()
    parole = testo.split()
    return len(parole)