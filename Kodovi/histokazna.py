#!/usr/bin/env python3

# Modul za numeriku
import numpy as np

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
  jed = [Jedinka(rez, tacke, boje, crvenih, plavih).skor
                 for rez in Jedinka.domen(n)]

  # Cuvanje rezultata
  with open('../Skorkazne/kazna9.txt', 'w') as dat:
    dat.write(repr(jed))

# Prikazivanje rezultata
def prikazi():
  # Citanje podataka
  with open('../Skorkazne/kazna9.txt', 'r') as dat:
    skor = eval(dat.read())

  najbolje = min(skor) # 23.445853886544874
  sredina = np.mean(skor) # 1246.5120897629688
  print(najbolje, sredina)

  plt.hist(skor)
  plt.show()

# Glavna funkcija
def main():
  #izracunaj()
  prikazi()

# Ispitivanje nacina pokretanja
if __name__ == '__main__':
  main()
