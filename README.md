#Návod na použití

1. Je nutno mit nainstalovanu Anacondu z
https://www.anaconda.com/distribution/#download-section.
Anaconda je package manager ktery se postara
o dotahnuti potrebnych balicku. Duvod proc jsme nepouzili
defaultni package manazer pip je ten, ze 
pip kompiluje balicky ze zdrojovych souboru
a v nasem pripade pada. Pokud se conda neprida do 
PATH, zkuste https://stackoverflow.com/questions/44597662/conda-command-is-not-recognized-on-windows-10.

2. V kořenovém adresáři programu (`./Program`)
je soubor requirements.txt. Tento soubor obsahuje seznam
potřebných balíčků, ktere je potreba mit
v soucasnem prostredi Pythonu, aby byl program spustitelny.
Nejlepsi zpusob jak toho dosahnout je vytvoreni 
conda prostredi prikazem
`conda create --name soc_Biroscak --file requirements.txt`
ve slozce `./Program`.

3. Dale je potreba aktivovat vytvorene
prostredi prikazem `conda activate soc_Biroscak`
(Nutno provest vzdy po spusteni nove konzole)

4. Pote staci spustit program, tedy v adresari `Program` 
prikaz `python svm.py`.




