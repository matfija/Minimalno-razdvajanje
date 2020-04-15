#!/usr/bin/env python3

# Moduli za numeriku
import numpy as np

# Modul za grafiku
import matplotlib.pyplot as plt

# Modul za jedinku
from jedinka import Jedinka

# Racunanje skorova
def izracunaj():
  for i in range(1, 7):
    tacke, boje, crvenih, plavih = \
           Jedinka.ucitaj(f'../Primeri/random{i}.txt')

    # Iteracija kroz domen
    n = len(tacke)
    jed = [Jedinka(rez, tacke, boje, crvenih, plavih).skor
                   for rez in Jedinka.domen(n)]

    # Cuvanje rezultata
    with open(f'../Skorkazne/skorkazna{i}.txt', 'w') as dat:
      dat.write(repr(jed))

# Prikazivanje rezultata
def prikazi():
  fig = plt.figure(figsize=(7,5))
  
  for i in range(1, 7):
    with open(f'../Skorkazne/skorkazna{i}.txt', 'r') as dat:
      skor = eval(dat.read())

    najbolje = min(skor)
    sredina = np.mean(skor)
    print(najbolje, sredina)

    fig.add_subplot(230+i).hist(skor)
  
  fig.tight_layout()
  fig.show()

# Glavna funkcija
def main():
  #izracunaj()
  prikazi()

# Ispitivanje nacina pokretanja
if __name__ == '__main__':
  main()
