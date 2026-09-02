# docu/_extraction/

**Cache de travail, entièrement régénérable.** Un fichier texte par chapitre du
cours, produit par
[../../scripts/youtube/split_course.py](../../scripts/youtube/split_course.py).

```powershell
.venv\Scripts\python.exe scripts\youtube\split_course.py 2 3 5 7 8 9 11 12
```

Les arguments sont des numéros de Partie. Le périmètre retenu couvre 35
chapitres, environ 59 000 mots.

## Statut

Ce n'est **pas un livrable**. C'est de la matière première extraite d'un PDF
sous droits appartenant à l'utilisateur, conservée localement le temps de
rédiger la synthèse. Les livrables sont les notes de [../synthese/](../synthese/)
et [../methode/](../methode/), rédigées avec leurs propres mots et leur propre
structure.

Ne rien publier depuis ce dossier, ne pas le versionner, ne pas y créer de
fichier à la main : il se reconstruit d'une commande.

## Nommage

`p<partie>-ch<chapitre>-<slug>.md`, ce qui rend l'ordre de lecture évident et
permet de retrouver un chapitre depuis une note qui le cite.

Chaque fichier porte en tête un commentaire indiquant les pages source, pour
remonter au PDF quand une formulation demande vérification.

## Limite de l'extraction

Le découpage s'appuie sur la table des matières en texte brut des trois
premières pages du PDF — le document ne porte pas de signets. Un chapitre
s'étend jusqu'au début du suivant, ce qui inclut parfois quelques lignes de
transition. Sans importance pour la lecture, à savoir si on compte des mots.
