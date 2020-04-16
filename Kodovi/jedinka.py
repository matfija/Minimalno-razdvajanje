#!/usr/bin/env python3

# Modul za gresku
from sys import exit as greska

# Modul za kopiranje
from copy import deepcopy

# Modul za matematiku
import numpy as np
norm = lambda x: np.array(x)/sum(x)

# Moduli za slucajnost
from random import seed, random, sample,\
                   randrange, randint, choice, choices
shuffled = lambda x: sample(x, len(x))
from numpy.random import choice as npsample, seed as npseed

# Moduli za iteraciju
from itertools import chain, combinations, permutations
from more_itertools import ilen, take, sample as isample,\
                           unique_everseen as unique

# Modul za rastojanje
from Levenshtein import editops

# Modul za geometriju
from shapely.geometry import Point, Polygon

# Modul za crtanje
import matplotlib.pyplot as plt

# Sve rotacije liste
def sverot(lista):
  for i in range(len(lista)):
    tek = lista[i:]+lista[:i]
    yield tek
    yield tek[::-1]

# Zamena slova u stringu
def swap(niz, i, j, elem=None):
  # Pretvaranje u listu
  niz = list(niz)

  # Zamena (swap) slova
  if j != -1:
    niz[i], niz[j] = niz[j], niz[i]
  else:
    niz[i] = elem

  # Vracanje rezultujuce niske
  return niz if isinstance(niz[0], int)\
             else ''.join(niz)

# Klasa za izmene
class Izmene():
  # Konstruktor klase
  def __init__(self, kod):
    self.kod = kod

  # Konkatenacija izmena
  def __add__(self, dr):
    return Izmene(self.kod + dr.kod)

  # Prosirivanje izmena
  def __iadd__(self, dr):
    self.kod += dr.kod
    return self

  # Umnozavanje izmena
  def __rmul__(self, koef):
    koef, ceo = np.modf(koef)
    koef = int(np.ceil(koef*len(self.kod)))
    return Izmene(int(ceo) * self.kod + self.kod[:koef])

  # Stringovni prikaz
  def __str__(self):
    return f'Izmene{self.kod}'

# Klasa za jedinke
class Jedinka(Polygon):
  # Brojac popunjenih jedinki tj.
  # evaluacije fje prilagodjenosti
  BROJAC = 0

  # Dekorator za odlozenu inicijalizaciju
  def odlozena(metod):
    def omotac(*args):
      for self in args:
        if isinstance(self, Jedinka):
          self.popuni()
      return metod(*args)
    return omotac
  
  # Potvrda crvene boje
  @staticmethod
  def crvena(boja):
    return boja == 'r'

  # Potvrda plave boje
  @staticmethod
  def plava(boja):
    return boja == 'b'

  # Ucitavanje iz datoteke
  @staticmethod
  def ucitaj(ime):
    with open(ime, 'r') as dat:
      tacke = np.array(eval(dat.readline()))
      boje = np.array(eval(dat.readline()))
      crvenih = ilen(filter(Jedinka.crvena, boje))
      plavih = ilen(filter(Jedinka.plava, boje))
    return tacke, boje, crvenih, plavih

  # Cuvanje u datoteku
  def sacuvaj(self, ime):
    with open(ime, 'w') as dat:
      dat.write('\n'.join((repr(self.tacke.tolist()),
                           repr(self.boje.tolist()))))

  # Priprema za rastojanje
  @staticmethod
  def pripremi(jedinka):
    # Izvlacenje koda
    if isinstance(jedinka, Jedinka):
      jedinka = jedinka.kod

    # Izvlacenje brojeva
    if isinstance(jedinka, str):
      jedinka = [*map(int, jedinka)]

    # Izvlacenje niske
    if isinstance(jedinka, list):
      jedinka = ''.join(map(chr, jedinka))

    # Vracanje rezultata
    return jedinka

  # Rastojanje permutacija
  @staticmethod
  def rastojanje(prva, druga):
    return len(Jedinka.sveops(prva, druga))

  # Operacije rekonstrukcije
  @staticmethod
  def rastops(prva, druga):
    # Priprema jedinki
    prva = Jedinka.pripremi(prva)
    druga = Jedinka.pripremi(druga)

    # Odredjivanje liste operacija
    operacije = []
    tekuce = [*editops(prva, druga)]
    while tekuce:
      # Razmatranje samo prve
      op, i, j = tekuce[-1]
      
      # Ako je zamena
      if op == 'replace':
        # Slova koja se menjaju
        x1, x2 = prva[i], druga[j]
        operacije.append(('zameni', (ord(x1), ord(x2))))

        # Zamena slova u nisci
        k = prva.find(x2)
        prva = swap(prva, i, k, x2)
      
      # Ako je dodavanje
      elif op == 'insert':
        # Indeks pozicije i slovo koje se dodaje
        operacije.append(('dodaj', (i, ord(druga[j]))))

        # Dodavanje slova na mesto
        prva = prva[:i] + druga[j] + prva[i:]

      # Ako je brisanje
      else: #op == 'delete'
        # Slovo koje se brise; nije indeks
        # kako bi radilo kod jata ptica
        operacije.append(('obrisi', ord(prva[i])))

        # Brisanje suvisnog slova
        prva = prva[:i] + prva[i+1:]

      # Azuriranje tekuce liste
      tekuce = [*editops(prva, druga)]
    
    # Vracanje operacija
    return operacije

  # Iscrpna rekonstrukcija
  @staticmethod
  def sveops(prva, druga):
    # Izvlacenje koda
    if isinstance(druga, Jedinka):
      druga = druga.kod

    # Minimizacija duzine puta
    return min((Jedinka.rastops(prva, dr) for dr
                in sverot(druga)), key=len)

  # Primena operacija
  @staticmethod
  def primeni(jedinka, operacije, postupno=False):
    # Priprema jedinke
    jedinka = [*map(ord, Jedinka.pripremi(jedinka))]

    # Popunjavanje liste
    if postupno:
      postupno = [jedinka.copy()]

    # Primena operacija
    for op, ind in operacije:
      # Zamena brojeva pre i posle
      if op == 'zameni':
        pre, posle = ind
        i = jedinka.index(pre) if pre in jedinka else -1
        j = jedinka.index(posle) if posle in jedinka else -1
        jedinka = [*swap(jedinka, i, j, posle)]

      # Dodavanje broja na indeks
      elif op == 'dodaj':
        i, slovo = ind
        if slovo not in jedinka:
          jedinka.insert(i, slovo)

      # Brisanje sa indeksa;
      # elif op == 'obrisi'
      elif ind in jedinka and len(jedinka) > 3:
        jedinka.remove(ind)

      # Popunjavanje liste
      if postupno:
        postupno.append(jedinka.copy())

    # Vracanje rezultata
    return postupno if postupno else jedinka

  # Dohvatanje preciscenog domena;
  # preporuceno za opste potrebe
  @staticmethod
  def domen(n):
    # Za svaki broj temena k
    for k in range(3, n+1):
      # Za svaku k-kombinaciju indeksa
      for komb in combinations(range(n), k):
        prvi = komb[0]
        # Emitovanje svake validne permutacije
        for perm in permutations(komb[1:]):
          if perm[0] < perm[-1]:
            yield [prvi, *perm]
  
  # Dohvatanje punog domena; za
  # slucaj da je stvarno potrebno
  @staticmethod
  def pundom(n):
    return chain(*(permutations(range(n), k) for k in range(3, n+1)))

  # Generisanje slucajnih jedinki iz
  # preciscenog domena; preporuceno
  @staticmethod
  def random(n, k=None):
    return isample(Jedinka.domen(n), 1)[0] if k is\
           None else isample(Jedinka.domen(k), n)

  # Generisanje slucajnih jedinki;
  # za slucaj da je bas potrebno
  @staticmethod
  def punrand(n, k=None):
    return Jedinka.standardizuj(sample(range(n), randint(3, n)))\
                                if k is None else\
          [Jedinka.standardizuj(sample(range(k), randint(3, k)))
                                for i in range(n)]

  # Odabir pocetnog resenja
  @staticmethod
  def pocetno(tacke, boje, crvenih=None, plavih=None, kazna=True):
    return Jedinka(Jedinka.random(len(tacke)),
                   tacke, boje, crvenih, plavih, kazna)

  # Odabir pocetne populacije
  @staticmethod
  def initpop(ime, npop=20, kazna=True):
    # Citanje tacaka iz datoteke
    tacke, boje, crvenih, plavih = Jedinka.ucitaj(ime)

    # Uzorkovanje iz domena
    return sorted(Jedinka(kod, tacke, boje, crvenih, plavih, kazna)
                          for kod in Jedinka.random(npop, len(tacke)))

  # Iscrpna pretraga nad datotekom
  @staticmethod
  def iscrpna(ime, kazna=True):
    # Citanje tacaka iz datoteke
    tacke, boje, crvenih, plavih = Jedinka.ucitaj(ime)

    # Iteracija kroz domen
    return min(Jedinka(kod, tacke, boje, crvenih, plavih, kazna)
                       for kod in Jedinka.domen(len(tacke)))

  # Slucajna pretraga nad datotekom
  @staticmethod
  def slucajna(ime, niter=500, kazna=True, seme=None):
    # Inicijalizacija pseudoslucajnosti
    seed(seme)
    
    # Citanje tacaka iz datoteke
    tacke, boje, crvenih, plavih = Jedinka.ucitaj(ime)

    # Iteracija kroz slucajnost
    return min(Jedinka(kod, tacke, boje, crvenih, plavih, kazna)
                       for kod in Jedinka.random(niter, len(tacke)))

  # Lokalna pretraga nad jedinkom
  def lokalizuj(self, niter=500, seme=None):
    # Inicijalizacija pseudoslucajnosti
    seed(seme)

    # Cuvanje polazne reference
    jed = self

    # Iteracija kroz susedstvo
    for i in range(1, niter):
      # Uzima se samo bolji sused
      jed = min(jed, jed.sused())

    # Popunjavanje polazne jedinke
    self.azuriraj(jed.kod)
    self.popuni()

  # Lokalna pretraga nad datotekom
  @staticmethod
  def lokalna(ime, pocetno=None, niter=500, kazna=True, seme=None):
    # Inicijalizacija pseudoslucajnosti
    seed(seme)
    
    # Odabir pocetnog resenja
    jed = Jedinka.pocetno(*Jedinka.ucitaj(ime), kazna) if pocetno\
          is None else Jedinka(pocetno, *Jedinka.ucitaj(ime), kazna)

    # Lokalna pretraga nad njim
    jed.lokalizuj(niter, seme)

    # Vracanje resenja
    return jed

  # Lokalna pretraga nad jedinkom
  def simkali(self, niter=500, seme=None):
    # Inicijalizacija pseudoslucajnosti
    seed(seme)

    # Cuvanje polazne reference
    jed = self

    # Cuvanje reference na najbolju
    naj = jed
  
    # Iteracija kroz susedstvo
    for i in range(1, niter):
      sus = jed.sused()

      # Uvek se uzima bolji sused
      if sus < jed:
        jed = sus
      
      # A ponekad moze i losiji
      elif 1/i**0.5 > random():
        jed = sus

      # Azuriranje najboljeg
      naj = min(naj, sus)

    # Popunjavanje polazne jedinke
    self.azuriraj(naj.kod)
    self.popuni()

  # Lokalna pretraga nad datotekom
  @staticmethod
  def simkal(ime, pocetno=None, niter=500, kazna=True, seme=None):
    # Inicijalizacija pseudoslucajnosti
    seed(seme)
    
    # Odabir pocetnog resenja
    jed = Jedinka.pocetno(*Jedinka.ucitaj(ime), kazna) if pocetno\
          is None else Jedinka(pocetno, *Jedinka.ucitaj(ime), kazna)

    # Lokalna pretraga nad njim
    jed.simkali(niter, seme)

    # Vracanje resenja
    return jed

  # Slucajna selekcija
  @staticmethod
  def slucsel(sortpop, k=None):
    return choices(sortpop)[0] if k is None\
           else choices(sortpop, k=k)

  # Turnirska selekcija
  @staticmethod
  def turnirska(sortpop, k=None, vel=4):
    sel = [min(sample(sortpop, vel)) for i in range(k or 1)]
    return sel[0] if k is None else sel

  # Dohvatanje skora
  @staticmethod
  @odlozena
  def skor(jedinka):
    return jedinka.skor

  # Ruletska selekcija
  @staticmethod
  def ruletska(sortpop, k=None):
    # Verovatnoca odabira je obrnuto proporcionalna
    # skoru; bolje su jedinke sa manjim skorom
    probs = norm([*map(Jedinka.skor, reversed(sortpop))])
    sel = [npsample(sortpop, p=probs) for i in range(k or 1)]
    return sel[0] if k is None else sel

  # Rangovska selekcija
  @staticmethod
  def rangovska(sortpop, k=None):
    # Verovatnoca odabira je obrnuto proporcionalna
    # indeksu; bolje su jedinke sa manjim indeksima
    probs = norm(range(len(sortpop), 0, -1))
    sel = [npsample(sortpop, p=probs) for i in range(k or 1)]
    return sel[0] if k is None else sel

  # Odabir jedinki za uskrtanje
  # nad sortiranom populacijom
  @staticmethod
  def selekcija(sortpop, n=None):
    # Odabir tipa selekcije
    k = randrange(4) if sortpop[0].kazna else randrange(3)

    # Primena odabrane selekcije
    if k == 0:
      return Jedinka.slucsel(sortpop, n)
    elif k == 1:
      return Jedinka.turnirska(sortpop, n)
    elif k == 2:
      return Jedinka.rangovska(sortpop, n)
    else:
      return Jedinka.ruletska(sortpop, n)

  # Ukrstanje prvog reda
  @staticmethod
  def prvogreda(prva, druga):
    # Izdvajanje kodova roditelja
    kodp, kodd = sorted((prva.kod, druga.kod), key=len)

    # Odredjivanje duzine dece
    duzinap = randint(len(kodp), len(np.union1d(kodp, kodd)))
    duzinad = randint(len(kodp), len(np.union1d(kodp, kodd)))

    # Odredjivanje indeksa
    ip, jp = sorted(sample(range(len(kodp)+1), 2))
    id, jd = sorted(sample(range(len(kodd)+1), 2))

    # Inicijalizacija dece
    detep = kodp[ip:jp]
    deted = kodd[id:jd]

    # Nesadrzani u deci
    nesadrzp = filter(lambda x: x not in detep, (*kodd[jp:], *kodd[:jp]))
    nesadrzd = filter(lambda x: x not in deted, (*kodp[jd:], *kodp[:jd]))

    # Popunjavanje dece
    detep.extend(take(max(0, duzinap-len(detep)), nesadrzp))
    deted.extend(take(max(0, duzinad-len(deted)), nesadrzd))

    # Vracanje dece
    return prva.kopija(detep), druga.kopija(deted)

  # Poziciono ukrstanje
  @staticmethod
  def poziciono(prva, druga):
    # Izdvajanje kodova roditelja
    kodp, kodd = prva.kod, druga.kod

    # Odredjivanje indeksa prekida
    i = choice(range(len(kodp)))
    j = choice(range(len(kodd)))

    # Generisanje dece
    detep = [*kodp[:i], *kodd[j:]]
    deted = [*kodp[i:], *kodd[:j]]

    # Vracanje dece
    return prva.kopija(detep), druga.kopija(deted)

  # Rekombinacija ivica
  @staticmethod
  def rekombivic(prva, druga):
    # Izdvajanje kodova roditelja
    kodp, kodd = prva.kod, druga.kod
    
    # Pravljenje mape susedstva
    sused = {}

    # Popujavanje prvom jedinkom
    n = len(kodp)
    for i in range(n):
      sused[kodp[i]] = [kodp[i-1], kodp[(i+1)%n]]

    # Dopunjavanje drugom jedinkom
    n = len(kodd)
    for i in range(n):
      if kodd[i] in sused:
        if kodd[i-1] not in sused[kodd[i]]:
          sused[kodd[i]].append(kodd[i-1])
        if kodd[(i+1)%n] not in sused[kodd[i]]:
          sused[kodd[i]].append(kodd[(i+1)%n])
      else:
        sused[kodd[i]] = [kodd[i-1], kodd[(i+1)%n]]

    # Prvi geni za razmatranje
    genp, gend = kodp[0], kodd[0]

    # Inicijalizacija dece
    detep, deted = [genp], [gend]
    duzinap = randint(len(kodp), len(sused))
    duzinad = randint(len(kodp), len(sused))

    # Kopiranje mape susedstva
    suskopija = deepcopy(sused)

    # Popunjavanje prvog deteta
    while len(detep) < duzinap:
      # Izbacivanje tekuceg gene iz suseda
      for i in sused:
        if genp in sused[i]:
          sused[i].remove(genp)

      # U slucaju neprazne liste suseda tekuceg gena
      if sused[genp]:
        # Odredjivanje novog gena sa najmanje suseda
        genp = min(sused[genp], key=lambda x: len(sused[x]))
      else:
        # U suprotnom uzimanje slucajnog novog gena
        genp = choice([*filter(lambda x: x not in detep, sused)])

      # Dodavanje tekuceg gena
      detep.append(genp)

    # Vracanje kopije mape susedstva
    sused = suskopija

    # Popunjavanje drugog deteta
    while len(deted) < duzinad:
      # Izbacivanje tekuceg gene iz suseda
      for i in sused:
        if gend in sused[i]:
          sused[i].remove(gend)

      # U slucaju neprazne liste suseda tekuceg gena
      if sused[gend]:
        # Odredjivanje novog gena sa najmanje suseda
        gend = min(sused[gend], key=lambda x: len(sused[x]))
      else:
        # U suprotnom uzimanje slucajnog novog gena
        gend = choice([*filter(lambda x: x not in deted, sused)])

      # Dodavanje tekuceg gena
      deted.append(gend)

    # Vracanje dece
    return prva.kopija(detep), druga.kopija(deted)

  # Ukrstanje jedinki
  @staticmethod
  def ukrstanje(prva, druga, p=0.9):
    # Ukrstanje ima zadatu verovatnocu
    if random() > p:
      return prva, druga
      
    # Odabir tipa ukrstanja
    k = randrange(3)

    # Primena odabranog ukrstanja
    if k == 0:
      return Jedinka.prvogreda(prva, druga)
    elif k == 1:
      return Jedinka.poziciono(prva, druga)
    else:
      return Jedinka.rekombivic(prva, druga)

  # Smena generacija (reprodukcija)
  @staticmethod
  def smena(populacija, pc=0.9, pm=0.01):
    # Za svaki par dvoje po dvoje
    opseg = iter(range(len(populacija)))
    for i, j in zip(opseg, opseg):
      # Eventualno ukrstanje roditelja
      populacija[i], populacija[j] = \
                     Jedinka.ukrstanje(populacija[i], populacija[j], pc)

      # Eventualna mutacija potomaka; ovde
      # realizovana kao zamena sa susedom
      populacija[i] = populacija[i].sused(pm)
      populacija[j] = populacija[j].sused(pm)

  # Genetski algoritam
  @staticmethod
  def genetski(ime, npop=20, niter=25, nsk=10,
               pc=0.9, pm=0.01, kazna=True, seme=None):
    # Inicijalizacija pseudoslucajnosti
    seed(seme)
    npseed(seme)

    # Elitna petina populacije
    elit = npop//5
    norm = npop-elit
    
    # Inicijalizacija populacije
    populacija = Jedinka.initpop(ime, npop, kazna)

    # Iterativna smena generacija
    for i in range(1, niter):
      # Odabir jedinki za ukrstanje
      odabrani = Jedinka.selekcija(populacija, norm)

      # Smena generacija odabranih
      Jedinka.smena(odabrani, pc, pm)

      # Ubacivanje nove generacije
      # na mesta iza elitnih jedinki
      populacija[elit:] = odabrani

      # Uredjivanje po skoru
      populacija.sort()

      # Kaljenje najbolje jedinke
      populacija[0].simkali(nsk, seme)

    # Vracanje najbolje jedinke
    return populacija[0]

  # Optimizacija jatom ptica
  @staticmethod
  def jato(ime, npop=10, niter=50, nsk=10,
           lok=2, glob=2, kazna=True, seme=None):
    # Inicijalizacija pseudoslucajnosti
    seed(seme)

    # Inicijalizacija populacije
    ptice = Jedinka.initpop(ime, npop, kazna)
    loknaj = ptice.copy()
    globnaj = min(ptice)
    brzine = [Izmene([]) for i in range(npop)]

    # Iterativno 'letenje' jedinki
    for i in range(1, niter):
      # Za svaku pticu iz jata
      for j in range(npop):
        # Azuriranje brzine
        brzine[j] += lok * random() * (loknaj[j] - ptice[j]) \
                   + glob * random() * (globnaj - ptice[j])

        # Azuriranje pozicije
        ptice[j] = ptice[j] + brzine[j]

        # Azuriranje minimuma
        loknaj[j] = min(loknaj[j], ptice[j])
        globnaj = min(globnaj, ptice[j])

      # Kaljenje najbolje jedinke
      globnaj.simkali(nsk, seme)

    # Vracanje najbolje jedinke
    return globnaj

  # Standardizacija koda jedinke; preciscenom domenu
  # pripadaju samo one ciji je prvi element minimalan,
  # a drugi manji od poslednjeg, npr. samo x = (1,2,3)
  # kao trougao, posto je min(x) = 1, a 2 < 3
  @staticmethod
  def standardizuj(kod):
    # Eliminisanje duplikata
    kod = [*unique(map(int, kod))]
    
    # Obrada izuzetka
    if len(kod) < 3:
      return []
    
    # Postavljanje minimuma na pocetak
    najmanji = np.argmin(kod)
    if najmanji != 0:
      kod = [*kod[najmanji:], *kod[:najmanji]]

    # Obrtanje ako je drugi veci od poslednjeg
    if kod[1] > kod[-1]:
      kod[1:] = reversed(kod[1:])

    # Vracanje standardizovanog koda
    return kod
  
  # Popunjavanje neicinijalizovane jedinke
  def popuni(self):
    if not self.inic:
      super().__init__(self.tacke[self.kod])
      self.inic = True

      # Boje tacaka
      self.crvenih = self.crvenih if self.crvenih else\
                     ilen(filter(Jedinka.crvena, self.boje))
      self.plavih = self.plavih if self.plavih else\
                    ilen(filter(Jedinka.plava, self.boje))

      # Boje temena
      self.nacrvenih = ilen(filter(Jedinka.crvena, self.boje[self.kod]))
      self.naplavih = ilen(filter(Jedinka.plava, self.boje[self.kod]))

      # Boje unutrasnjosti
      self.ucrvenih = ilen(filter(self.contains,
                                  (Point(self.tacke[i])
                                   for i in range(len(self.tacke))
                                   if Jedinka.crvena(self.boje[i]))))
      self.uplavih = ilen(filter(self.contains,
                                 (Point(self.tacke[i])
                                  for i in range(len(self.tacke))
                                  if Jedinka.plava(self.boje[i]))))

      # Boje spoljasnjosti
      self.vancrvenih = self.crvenih - self.nacrvenih - self.ucrvenih
      self.vanplavih = self.plavih - self.naplavih - self.uplavih

      # Minimalan broj promasenih tacaka;
      # greska je da su unutra crvene a van plave
      # ili da su van crvene a unutra plave
      self.promasenih = min(self.ucrvenih + self.vanplavih,
                            self.vancrvenih + self.uplavih)

      # Skor sa kaznom za promasene tacke
      self.skor = (1 + 10 * self.promasenih) * self.length

      # Dodatna kazna za nevalidne poligone
      if not self.is_valid:
        self.skor *= 10

      # Potpuno odbacivanje resenja ukoliko
      # nije dozvoljena manja kazna
      if not self.kazna and (not self.is_valid or self.promasenih):
         self.skor = float('inf')

      # Azuriranje brojaca jedinki
      Jedinka.BROJAC += 1
  
  # Konstruktor jedinke
  def __init__(self, kod, tacke, boje, crvenih=None, plavih=None, kazna=True):
    self.kod = [*map(int, kod)]
    self.tacke = tacke
    self.boje = boje
    self.crvenih = crvenih
    self.plavih = plavih
    self.kazna = kazna
    self.inic = False

  # Kopija jedinke
  def kopija(self, kod=[], k=None):
    kod = Jedinka.standardizuj(kod)
    return Jedinka(kod if kod else self.kod, self.tacke, self.boje,
                   self.crvenih, self.plavih, k if k else self.kazna)

  # Azuriranje koda; zbog njega je neophodno
  # kopiranje u implementaciji mutacija
  def azuriraj(self, kod=[], k=None):
    std = Jedinka.standardizuj(kod)
    if (std == self.kod):
      return False
    else:
      self.kod = std if std else Jedinka.standardizuj(self.kod)
      self.kazna = k if k else self.kazna
      self.inic = False
      return True

  # Mutacija umetanjem
  def umetni(self, i, j):
    kod = self.kod.copy()
    kod[i+1:] = [kod[j], *kod[i+1:j], *kod[j+1:]]
    return self.azuriraj(kod)

  # Mutacija zamenom
  def zameni(self, i, j=None):
    kod = self.kod.copy()
    if j is None:
      kod[i] = choice(np.setdiff1d(range(len(self.tacke)), kod))
    else:
      kod[i], kod[j] = kod[j], kod[i]
    return self.azuriraj(kod)

  # Mutacija obrtanjem
  def obrni(self, i, j):
    kod = self.kod.copy()
    kod[i:j+1] = reversed(kod[i:j+1])
    return self.azuriraj(kod)

  # Mutacija mesanjem
  def mesaj(self, i, j):
    kod = self.kod.copy()
    kod[i:j+1] = shuffled(kod[i:j+1])
    return self.azuriraj(kod)

  # Mutacija dodavanjem
  def dodaj(self, i):
    self.kod.insert(i,
                    choice(np.setdiff1d(range(len(self.tacke)),
                                        self.kod)))
    return self.azuriraj()

  # Mutacija brisanjem
  def obrisi(self, i):
    self.kod.pop(i)
    return self.azuriraj()

  # Operator mutacije
  def mutiraj(self, p=0.01):
    # Mutacija ima zadatu verovatnocu
    if random() > p:
      return

    # Pokusaj pravljenja mutacije
    # dok ne dodje do promene
    n, promena = len(self.kod), False
    while not promena:
      # Odabir tipa mutacije; mora dodavanje ili zamena
      # elementa ako je trougao, a ne sme ako je n-tougao
      k = randint(5, 6) if n == 3 else randrange(5) if\
          n == len(self.tacke) else randrange(7)

      # Odabir indeksa za algoritme mutacije
      i, j = sorted(sample(range(n), 2))

      # Primena odabrane mutacije
      if k == 0:
        promena = self.umetni(i, j)
      elif k == 1:
        promena = self.zameni(i, j)
      elif k == 2:
        promena = self.obrni(i, j)
      elif k == 3:
        promena = self.mesaj(i, j)
      elif k == 4:
        promena = self.obrisi(randrange(n))
      elif k == 5:
        promena = self.zameni(randrange(n))
      else:
        promena = self.dodaj(randrange(n+1))
  
  # Sused jedinke; mutacija nad kopijom
  def sused(self, p=1):
    kopija = self.kopija()
    kopija.mutiraj(p)
    return kopija

  # Crtanje jedinke
  @odlozena
  def nacrtaj(self, fig=None, i=None):
    # Izdvajanje koordinata
    x = [t[0] for t in self.tacke]
    y = [t[1] for t in self.tacke]

    # Prikaz samo problema
    if i == 2:
      fig.add_subplot(321).scatter(x, y, c=self.boje)

    # Prikaz problema i resenja
    if fig is None:
      plt.scatter(x, y, c=self.boje)
      plt.plot(*self.exterior.xy)
      plt.show()
    else:
      fig = fig.add_subplot(320+i)
      fig.scatter(x, y, c=self.boje)
      fig.plot(*self.exterior.xy)

  # Dodavanje izmena
  def __add__(self, izmene):
    return self.kopija(Jedinka.primeni(self, izmene.kod))

  # Razlika jedinki
  def __sub__(self, dr):
    return Izmene(Jedinka.sveops(dr, self))

  # Poredjenje jedinki
  @odlozena
  def __lt__(self, dr):
    return (self.skor, self.promasenih, self.length) <\
           (  dr.skor,   dr.promasenih,   dr.length)

  # Detaljan prikaz jedinke
  @odlozena
  def __str__(self):
    return f'Kod {self.kod}, obim {self.length:.2f},\n'\
           f'promasenih {self.promasenih}, skor {self.skor:.2f},\n'\
           f'nacrvenih {self.nacrvenih}, naplavih {self.naplavih},\n'\
           f'ucrvenih {self.ucrvenih}, uplavih {self.uplavih},\n'\
           f'vancrvenih {self.vancrvenih}, vanplavih {self.vanplavih}'

# Ispitivanje nacina pokretanja
if __name__ == '__main__':
  greska('Jedinka nije samostalan program! Pokrenite main!')
