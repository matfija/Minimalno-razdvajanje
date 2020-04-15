#!/usr/bin/env python3

# Modul za numeriku
import numpy as np
from numpy import inf

# Modul za grafiku
import matplotlib.pyplot as plt

# Modul za jedinku
from jedinka import Jedinka

# Racunanje skorova
def izracunaj():
  # Citanje tacaka
  tacke, boje, crvenih, plavih = Jedinka.ucitaj('../Primeri/ulaz9.txt')

  # Iteracija kroz domen
  n = len(tacke)
  jed = [Jedinka(rez, tacke, boje, crvenih, plavih, False).skor
                 for rez in Jedinka.domen(n)]

  # Cuvanje rezultata
  with open('../Skorovi/skor9.txt', 'w') as dat:
    dat.write(repr(jed))

# Prikazivanje rezultata
def prikazi():
  # Citanje podataka
  with open('../Skorovi/skor9.txt', 'r') as dat:
    skor = eval(dat.read())

  ukupno = len(skor) # 62814
  skor = np.array([*filter(lambda x: x!=inf, skor)])
  dopustivo = len(skor) # 1277
  sredina = np.mean(skor) # 39.46604267796484
  print(ukupno, dopustivo, sredina)

  plt.hist(skor)
  plt.show()

# Glavna funkcija
def main():
  #izracunaj()
  prikazi()

# Ispitivanje nacina pokretanja
if __name__ == '__main__':
  main()
