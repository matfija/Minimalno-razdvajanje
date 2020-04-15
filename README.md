#### Metaheuristike
<img width="700" src="https://github.com/matfija/Minimalno-razdvajanje/blob/master/Slike/poredjenje30.png">

## Minimalno razdvajanje :triangular_ruler:
Seminarski rad na kursu Računarska inteligencija. Cilj je bio osmisliti optimizacioni algoritam za određivanje [minimalnog mnogougaonog razdvajanja dva konačna skupa crvenih i plavih tačaka u ravni](https://www.csc.kth.se/~viggo/wwwcompendium/node272.html). Tražen je najmanji (u smislu obima) mnogougao koji ispunjava postavljeni zahtev – temena su mu iz zadatih skupova obojenih tačaka i njihova boja se zanemaruje, pri čemu on deli ravan na dva dela, unutrašnjost i spoljašnjost, tako da su sve tačke koje se nalaze u jednom delu iste boje, dok je boja tačaka u različitim delovima različita. Izložen je kontrolni algoritam iscrpne pretrage koji garantovano pronalazi optimum, zatim neka heuristička rešenja iz literature, slučajna i lokalna pretraga sa simuliranim kaljenjem, a na kraju je predložena metaheuristika zasnovana na genetskom algoritmu.

## Detalji i podešavanje :memo:
Svi programi su napisani u jeziku Python (verzija 3.7.5), na operativnom sistemu Windows, uz upotrebu zvaničnog [IDLE](https://docs.python.org/3/library/idle.html)-a kao integrisanog razvojnog okruženja, ali, usled fleksibilnosti samog Pythona, rade i na drugim operativnim sistemima, kao što su Ubuntu i druge distribucije Linuxa.

Od nestandardnih biblioteka, korišćene su [numpy](https://numpy.org/) i [matplotlib](https://matplotlib.org/) za matematička izračunavanja i crtanje, [more-itertools](https://more-itertools.readthedocs.io/en/stable/) za naprednu iteraciju, [Shapely](https://shapely.readthedocs.io/en/latest/manual.html) za komforan rad sa mnogouglovima i [python-Levenshtein-wheels](https://pypi.org/project/python-Levenshtein-wheels/) za rad sa Levenštajnovim putem i rastojanjem uređivanja, a najlakše ih je podesiti pomoću [pip](https://pip.pypa.io/en/stable/)-a i postavljene datoteke sa zavisnostima komandom poput `pip install -r reqs.txt`, kojom će svaki nedostajući biti instaliran.

Nakon kloniranja (`git clone https://github.com/matfija/Minimalno-razdvajanje`) tj. bilo kog načina preuzimanja repozitorijuma, željeni skript se pokreće pozivanjem Pajtonovog interpretatora nad fajlom, što je moguće učiniti komandom poput `python3 veliki.py` ili `python veliki.py`. Omogućeno je i direktno pokretanje, pošto se na početku svake datoteke nalazi shebang koji sugeriše operativnom sistemu gde se nalazi neophodni interpretator. Naravno, za ovaj pristup je neophodno prethodno učiniti fajl izvršivim komandom poput `chmod u+x main.py`.