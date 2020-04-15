#!/usr/bin/env python3

# Moduli za numeriku
import numpy as np
from numpy import inf
import random

# Modul za grafiku
import matplotlib.pyplot as plt

# Modul za jedinku
from jedinka import Jedinka

# Generisanje sest slucajnih skupova
def generisi():
  random.seed(0)
  for i in range(1, 7):
    broj = random.randint(6, 8)
    tacke = [(random.randint(1, 100),
              random.randint(1, 100))
              for j in range(broj)]
    boje = random.choices(('r', 'b'), k=broj)

    with open(f'../Primeri/random{i}.txt', 'w') as dat:
      dat.write('\n'.join((repr(tacke), repr(boje))))

# Racunanje skorova
def izracunaj():
  for i in range(1, 7):
    tacke, boje, crvenih, plavih = \
           Jedinka.ucitaj(f'../Primeri/random{i}.txt')

    # Iteracija kroz domen
    n = len(tacke)
    jed = [Jedinka(rez, tacke, boje, crvenih, plavih, False).skor
                   for rez in Jedinka.domen(n)]

    # Cuvanje rezultata
    with open(f'../Skorovi/skorrand{i}.txt', 'w') as dat:
      dat.write(repr(jed))

# Prikazivanje rezultata
def prikazi():
  fig = plt.figure(figsize=(7,5))
  
  for i in range(1, 7):
    with open(f'../Skorovi/skorrand{i}.txt', 'r') as dat:
      skor = eval(dat.read())

    ukupno = len(skor)
    skor = np.array([*filter(lambda x: x!=inf, skor)])
    dopustivo = len(skor)
    najbolje = min(skor)
    sredina = np.mean(skor)
    print(ukupno, dopustivo, najbolje, sredina)

    fig.add_subplot(230+i).hist(skor)
  
  fig.tight_layout()
  fig.show()

# Glavna funkcija
def main():
  #generisi()
  #izracunaj()
  prikazi()

# Ispitivanje nacina pokretanja
if __name__ == '__main__':
  main()
