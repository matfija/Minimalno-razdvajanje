#!/usr/bin/env python3

# Modul za jedinku
from jedinka import Jedinka

# Iscrpna pretraga
def iscrpna(ime, opt, kazna=False):
  # Iscrpna pretraga fajla
  jed = Jedinka.iscrpna(ime, kazna)

  # Prikazivanje rezultata
  print(jed.skor, Jedinka.rastojanje(jed, opt))
  #jed.nacrtaj()

# Slucajna pretraga
def slucajna(ime, opt, niter=500, kazna=False, seme=0):
  jed = Jedinka.slucajna(ime, niter, kazna, seme)

  # Prikazivanje rezultata
  print(jed.skor, Jedinka.rastojanje(jed, opt))
  #jed.nacrtaj()

# Slucajna pretraga
def lokalna(ime, opt, pocetno=None, niter=500, kazna=False, seme=100):
  jed = Jedinka.lokalna(ime, pocetno, niter, kazna, seme)

  # Prikazivanje rezultata
  print(jed.skor, Jedinka.rastojanje(jed, opt))
  #jed.nacrtaj()

# Simulirano kaljenje
def simkal(ime, opt, pocetno=None, niter=500, kazna=False, seme=100):
  jed = Jedinka.simkal(ime, pocetno, niter, kazna, seme)
  
  # Prikazivanje rezultata
  print(jed.skor, Jedinka.rastojanje(jed, opt))
  #jed.nacrtaj()

# Genetski algoritam
def genetski(ime, opt, npop=20, niter=23, nsk=7,
             pc=0.9, pm=0.3, kazna=True, seme=3):
  jed = Jedinka.genetski(ime, npop, niter, nsk, pc, pm, kazna, seme)

  # Prikazivanje rezultata
  print(jed.skor, Jedinka.rastojanje(jed, opt))
  #jed.nacrtaj()

# Jato ptica
def jato(ime, opt, npop=10, niter=30, nsk=7,
         lok=2, glob=2, kazna=True, seme=7):
  jed = Jedinka.jato(ime, npop, niter, nsk, lok, glob, kazna, seme)

  # Prikazivanje rezultata
  print(jed.skor, Jedinka.rastojanje(jed, opt))
  #jed.nacrtaj()

# Glavna funkcija
def main():
  # Svi implementirani algoritmi
  algoritmi = iscrpna, slucajna, lokalna, simkal, genetski, jato

  # Svi generisani test primeri
  datoteke = ('../Primeri/ulaz9.txt',
              *(f'../Primeri/random{i}.txt' for i in range(1, 7)))

  # Optimalna resenja svih primera
  opt = '027653', '034', '235', '134', '157', '0145', '025'

  # Primena svakog algoritma na
  # svaki generisani test primer
  for algoritam in algoritmi:
    print(algoritam.__name__)
    for datoteka in zip(datoteke, opt):
      algoritam(*datoteka)
    print()

# Ispitivanje nacina pokretanja
if __name__ == '__main__':
  main()
