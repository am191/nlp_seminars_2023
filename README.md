# NLP seminārs 2023

- **data** - satur visas nepieciešamās train un dev kopas + UD 2.11 test kopa. "text" folderī ir train kopas konvertētas uz .txt failu ar noņemtām .conllu anotācijām (teikumi, dokumentu references utml.) - to izmantoja MarMot modeļu apmācības laikā

- **models** - satur katra modeļa pēdējo (UD 2.11) versiju, jo tā ir ar visprecīzākajiem rezultātiem. Mape models/spacy/lv_udtb_latvian satur spacy_test.py failu, kas izsauc konkrētā modeļa teksta marķēšanas funkcijas (ārpus konkrētās mapes izsaukt modeli nav izdevies - neatrod).

- **results** - teksta faili ar marķētajām vienībām.

Par failiem:
- flair_train.py - Flair apmācības skripts
- [spaCy](https://spacy.io/usage/training0) un [MarMot](https://github.com/muelletm/cistern/blob/wiki/marmot.md) tika apmācīti caur komandrindu
- scikit_scores.py - izmantots novērtēšanas stadijā
