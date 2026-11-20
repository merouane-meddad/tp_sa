# Résultats — Dataset 5, sentiments des tweets

Synthèse des trois tâches, à reprendre dans le rapport. Tous les chiffres viennent des
tables de `reports/tables/` (reproductibles avec les scripts, graine 42).

## Données et protocole

- **Train** : 50 306 tweets (2011 – déc. 2015). **Test** : 12 284 tweets (nov.–déc. 2016),
  dates retrouvées à partir des identifiants Twitter (« snowflake IDs »).
- Le chargeur répare des artefacts d'encodage présents dans **37 %** des textes bruts du train :
  guillemets CSV (`"…""…""…"`), échappements JSON (`\u2019`, `\u002c`), entités HTML (`&amp;`).
  C'est ce qui faisait apparaître `u002c` parmi les mots les plus fréquents. Dans le test, 10 % des
  textes sont en « mojibake » (UTF-8 lu comme Windows-1252 : `ðŸ˜‚` au lieu de 😂), aussi réparé.
- **87 textes** ont des étiquettes contradictoires et ~1,5 % des lignes sont dupliquées.
  Pour la modélisation, `deduplicate()` retire les conflits et ne garde qu'une copie des doublons.
- Protocole : train dédupliqué (49 421 tweets) découpé en **train 90 % / validation 10 %**
  stratifiés. Prétraitements et hyperparamètres sont choisis par validation croisée à 5 plis sur
  le train ; le fichier de test n'est utilisé qu'une fois, pour la comparaison finale.
- Métriques : **macro-F1** (principale), **macro-rappel** (métrique officielle de SemEval-2017
  tâche 4A), F1 moyen des classes positive/négative (F1-PN). L'accuracy n'est qu'indicative.

## Tâche 1 — Analyse descriptive

| | Négatif | Neutre | Positif |
|---|---|---|---|
| Train | 15,6 % | 44,9 % | 39,6 % |
| Test | **32,3 %** | 48,3 % | **19,3 %** |

- **Déséquilibre et décalage de distribution** : le test contient deux fois plus de négatifs et deux
  fois moins de positifs que le train, d'où le choix de poids de classes équilibrés et de métriques
  macro.
- **Longueur** : quasi identique entre classes (ε² < 0,006 malgré un test de Kruskal-Wallis
  significatif à cause de la taille de l'échantillon).
- **Mots fréquents** : surtout liés à la collecte (dates, événements : *tomorrow*, *may*, *1st*).
  Les **négations** (*not*, *no*, *don't*) sont 2 à 3 fois plus fréquentes dans les tweets négatifs,
  elles ne doivent donc pas être supprimées comme mots vides.
- **Delta TF-IDF amélioré** (cours, α = 0,5) : mots positifs évaluatifs (*enjoying*, *excellent*,
  *amazing*, *congrats*) ; mots négatifs à la fois évaluatifs (*horrible*, *worst*, *racist*) et liés à
  l'actualité violente ou judiciaire (*manslaughter*, *terror*, *prison*, *syria*). Une partie de la
  polarité passe donc par le sujet du tweet.
- **Correction** : la version précédente comptait 23–35 % de tweets avec émoticône. L'expression
  régulière détectait `:/` dans chaque `https://`. Après correction : 2–8 %. Le fichier d'entraînement
  ne contient **aucun emoji** (alors que 4–15 % des tweets de test en ont).

## Tâche 2 — Modèles d'apprentissage automatique

### Prétraitement (cours Ch3), régression logistique, validation croisée 5 plis

| Prétraitement | macro-F1 CV | Vocabulaire |
|---|---|---|
| Nettoyage seul | 0,638 | 96 615 |
| Mots vides retirés | 0,633 | 66 889 |
| Mots vides + lemmatisation | 0,641 | 67 448 |
| Mots vides + racinisation | 0,640 | 66 960 |
| **Lemmatisation (mots vides conservés)** | **0,643** | 91 874 |

Retirer les mots vides réduit le vocabulaire d'un tiers mais n'améliore pas le score : sur des textes
de 20 mots, des mots comme *but*, *so* ou *too* portent de l'information. La lemmatisation apporte un
léger gain. Les écarts restent de l'ordre de l'écart-type entre plis (±0,003 à 0,006).

### Comparaison des modèles (hyperparamètres choisis par grid search)

| Modèle | Val macro-F1 | Test macro-F1 | Test macro-rappel | Test F1-PN |
|---|---|---|---|---|
| **SVM linéaire, Delta TF-IDF (un-contre-tous)** | **0,658** | **0,617** | 0,614 | 0,600 |
| SVM linéaire, TF-IDF | 0,656 | 0,606 | 0,606 | 0,586 |
| Vote majoritaire NB + SVM + ME | 0,656 | 0,610 | 0,611 | 0,596 |
| ME (régression logistique), TF-IDF | 0,654 | 0,610 | 0,613 | 0,599 |
| CNN + LSTM (embeddings word2vec) | 0,651 | 0,612 | **0,622** | 0,612 |
| LSTM (embeddings word2vec) | 0,650 | 0,609 | 0,620 | **0,613** |
| MLP, 1 couche cachée (TF-IDF) | 0,621 | 0,583 | 0,592 | 0,577 |
| RNN simple (embeddings word2vec) | 0,617 | 0,572 | 0,585 | 0,571 |
| SVM linéaire, word2vec pondéré par Delta TF-IDF | 0,615 | 0,566 | 0,593 | 0,589 |
| Naive Bayes, TF-IDF | 0,610 | 0,585 | 0,597 | 0,583 |

À retenir :

- **Delta TF-IDF** donne le meilleur modèle linéaire (+1,1 point de macro-F1 en test par rapport au
  TF-IDF classique), en pondérant chaque n-gramme par sa capacité à séparer les classes.
- **RNN → LSTM → CNN+LSTM** : le RNN simple est nettement en dessous (0,572) et instable d'une
  époque à l'autre. Le LSTM, réponse du cours à la disparition du gradient, gagne près de 4 points.
  La convolution du modèle hybride ajoute un léger gain. Les modèles récurrents ont le meilleur
  **rappel sur les négatifs** (0,71–0,73 contre 0,57–0,64 pour les modèles linéaires), d'où le meilleur
  macro-rappel. Ils sur-apprennent vite : le meilleur score de validation est atteint dès la 2ᵉ époque.
- Le **vote majoritaire** n'améliore pas son meilleur membre : NB, SVM et ME utilisent les mêmes
  attributs et font les mêmes erreurs.
- Le **MLP** sur 30 000 n-grammes sur-apprend (arrêt précoce après 7 itérations) et reste en dessous
  des modèles linéaires.
- Le **word2vec pondéré** (formule du cours) perd l'information des n-grammes. Entraîné sur
  seulement 44 000 tweets, word2vec donne des vecteurs peu discriminants pour la classe neutre
  (rappel neutre 0,44).
- **Écart validation/test** d'environ 4 points pour tous les modèles : c'est le décalage temporel et
  thématique du test (2016, autres sujets), pas un sur-apprentissage d'un modèle particulier.
- **Erreurs** : la confusion principale est *neutre ↔ polarisé* (21–37 % des négatifs et positifs
  prédits neutres). Les confusions négatif ↔ positif sont plus rares (5–14 %). Un échantillon d'erreurs se
  trouve dans `task2_error_sample.csv`.
- À titre de repère, les meilleurs systèmes de SemEval-2017 (réseaux profonds avec embeddings
  pré-entraînés sur des millions de tweets) atteignaient un macro-rappel d'environ 0,68 sur ce test.

## Tâche 3 — Métadonnées

Associations avec la polarité (train, V de Cramér : 0,1 = faible, 0,3 = moyen) :

| Métadonnée | Négatif | Neutre | Positif | V de Cramér |
|---|---|---|---|---|
| Point d'exclamation | 12,8 % | 12,5 % | **33,1 %** | **0,25** |
| Juron non censuré | **13,4 %** | 1,8 % | 2,2 % | **0,22** |
| Émoticône positive | 0,8 % | 1,4 % | 7,3 % | 0,16 |
| `!!` répétés | 2,5 % | 2,0 % | 7,4 % | 0,13 |
| URL | 19,7 % | **32,8 %** | 25,5 % | 0,11 |
| Émoticône négative | 2,8 % | 0,5 % | 0,3 % | 0,10 |
| Point d'interrogation | 13,7 % | 14,6 % | 8,9 % | 0,08 |
| Injure censurée (f\*\*k…) | 0,24 % | 0,06 % | 0,08 % | 0,02 |
| Marqueur explicite d'ironie | 0,09 % | 0,04 % | 0,13 % | 0,01 |

- **URLs** : surtout dans les tweets neutres (titres d'articles, liens partagés).
- **Émoticônes** : polarité très nette (`:)` 82 % positif, `:(` 67 % négatif ; Delta-IDF ±3,6),
  mais elles sont rares. Les emoji n'existent que dans le test (15 % des positifs, 4–5 % des négatifs
  et des neutres) : `😍` et `❤️` sont presque toujours positifs, mais `😂` se répartit entre
  négatif (23), neutre (43) et positif (41), et `😭` à parts égales entre les trois classes. Le rire
  sert aussi à se moquer et les pleurs à exprimer une émotion positive : le sens d'un emoji dépend
  du contexte.
- **Ponctuation** : l'exclamation est la métadonnée la plus liée au sentiment (positif).
- **Injures censurées** : très rares (48 occurrences sur 50 000 tweets), plus fréquentes en négatif.
  Les jurons non censurés sont bien plus fréquents (13 % des négatifs). La détection valide chaque
  forme masquée contre un lexique de jurons, ce qui écarte `A$AP`, `#rap` ou `is@the`.
- **Ironie** : les marqueurs explicites sont quasi absents (0,04–0,13 %). Sur un échantillon de
  12 tweets détectés, environ 4 sont vraiment ironiques. Des expressions comme *so much fun* ou
  *I love how* sont surtout littérales. Les tweets négatifs avec émoticône positive expriment plutôt un
  sentiment mitigé ou adouci que de l'ironie (`task3_validation_samples.csv`).
- **Apport au classifieur** (régression logistique TF-IDF + métadonnées normalisées) :
  - toutes les métadonnées : +1,0 point de macro-F1 en validation, +0,3 en test ;
  - la **ponctuation** apporte presque tout le gain (+1,2 en validation, +0,5 en test) ;
  - les émoticônes n'apportent rien, car elles sont déjà dans le texte sous forme de jetons
    `EMOPOS`/`EMONEG` ;
  - les métadonnées seules : macro-F1 0,45 (contre 0,33 pour un tirage au hasard).
- **Décalage train/test** : URLs dans 53 % des tweets de test contre 28 % en train, emoji absents du
  train, hashtags deux fois plus fréquents en test. Les métadonnées apprises sur le train se
  transfèrent mal.
