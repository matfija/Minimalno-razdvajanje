#!/usr/bin/env python3

# Modul za slucajnost
import random

# Modul za crtanje
import matplotlib.pyplot as plt

# Modul za jedinku
from jedinka import Jedinka
Jedinka.random = Jedinka.punrand

# Generisanje slucajnog skupa
def generisi(broj, naj, seme=0):
  random.seed(0)
  broj, naj = 30, 1000
  tacke = [(random.randint(1, naj),
            random.randint(1, naj)) for i in range(broj)]
  boje = random.choices(('r', 'b'), k=broj)

  # Upisavanje tacaka u datoteku
  with open(f'../Primeri/ulaz{broj}.txt', 'w') as dat:
    dat.write('\n'.join((repr(tacke), repr(boje))))

# Uporedjivanje rezultata
def uporedi(broj, seme=0):
  # Ime velike datoteke
  ime = f'../Primeri/ulaz{broj}.txt'

  # Upisivanje rezultata u datoteku
  with open(f'../Skorovi/skor{broj}.txt', 'w') as dat:
    # Primena slucajne pretrage
    jed = Jedinka.slucajna(ime, niter=100000,
                           kazna=False, seme=0)
    print('Slucajna:', jed.skor, jed.kod, file=dat)

    # Primena lokalne pretrage
    jed = Jedinka.lokalna(ime, niter=100000,
                           kazna=False, seme=0)
    print('Lokalna:', jed.skor, jed.kod, file=dat)
    
    # Primena simuliranog kaljenja
    jed = Jedinka.simkal(ime, niter=100000,
                           kazna=False, seme=0)
    print('Kaljenje:', jed.skor, jed.kod, file=dat)

    # Primena genetskog algoritma
    jed = Jedinka.genetski(ime, npop=500, niter=250, nsk=10,
                           pc=0.9, pm=0.3, kazna=False, seme=0)
    print('Genetski:', jed.skor, jed.kod, file=dat)

    # Primena jata ptica
    jed = Jedinka.jato(ime, npop=500, niter=250, nsk=10,
                       lok=2, glob=2, kazna=False, seme=0)
    print('Jato:', jed.skor, jed.kod, file=dat)

# Prikazivanje rezultata
def prikazi(broj):
  # Ucitavanje primera
  primer = Jedinka.ucitaj(f'../Primeri/ulaz{broj}.txt')
  
  # Citanje dobijenih resenja
  with open(f'../Skorovi/skor{broj}.txt', 'r') as dat:
    resenja = [eval(''.join(rez.split(' ')[2:])) for rez in dat]
  
  # Prikazivanje svakog
  fig = plt.figure(figsize=(7,7))
  for i, res in enumerate(resenja, 2):
    Jedinka(res, *primer, False).nacrtaj(fig, i)
  fig.tight_layout()
  fig.show()

# Glavna funkcija
def main():
  broj, naj = 30, 1000
  #generisi(broj, naj)
  #uporedi(broj)
  prikazi(broj)

# Ispitivanje nacina pokretanja
if __name__ == '__main__':
  main()
