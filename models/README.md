# models/

Dossier des modèles entraînés. Son contenu est **généré** par `python -m src.train` et n'est pas
versionné (sauf ce fichier), car il se reconstruit en quelques minutes et dépend des versions
des bibliothèques (`requirements.txt`).

| Fichier | Contenu |
|---|---|
| `model.joblib` | modèle retenu (pipeline complet : variables dérivées → prétraitement → régresseur, cible log1p) |
| `model_metadata.json` | modèle choisi, hyperparamètres, noms des variables transformées, catégories connues, plages observées, période, versions, empreinte des données |
| `shap_background.csv` | 200 maisons tirées du jeu d'**entraînement**, servant de référence SHAP |
| `candidate_<modèle>.joblib` | meilleur réglage de chaque candidat (utilisé par `src.evaluate` pour comparer sur le test) |

L'application et `src.evaluate` chargent ces fichiers sans ré-entraîner ; s'ils sont absents,
un message indique la commande à lancer.
